"""Bounded Stage 3 atom-physics training entry point.

This script trains the direct atom-force distribution only.  It does not run
transition training and never passes a force target into the model forward.
The default CLI is intentionally bounded; a reported run should first pass
the target audit and tiny-overfit gate and then explicitly increase limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow import (
    ESMCConfig,
    ForceHead,
    ForceLossConfig,
    HeavyFlowContextConfig,
    HeavyFlowContextEncoder,
    HeavyFlowGeometryConfig,
    HeavyFlowGraphConfig,
    HeavyFlowPhysicsModel,
    PhysicsPredictor,
    PhysicsPredictorConfig,
    audit_force_targets,
    collate_heavy_flow,
    force_loss,
)
from force_md.heavy_flow.force_losses import ForceNormalizer
from force_md.heavy_flow.history import HistoryConfig
from force_md.heavy_flow.checkpoint import (ARCHITECTURE_VERSION, NORMALIZER_DEFINITION, architecture_config, load_normalizer_artifact, warm_start_stage3, finite_optimizer_gradients)
from force_md.heavy_flow.physics_dataset import HeavyFlowPhysicsFrameDataset, PhysicsSplitManifest, build_physics_splits


def _dataclass_kwargs(value: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value[key] for key in allowed if key in value}


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def build_stage3_model(config: dict[str, Any], *, device: Optional[str] = None) -> HeavyFlowPhysicsModel:
    config = architecture_config(config)
    sequence = dict(config.get("sequence_encoder", {}))
    if "family" in sequence:
        sequence["model_family"] = sequence.pop("family")
    esm_keys = {"model_family", "model_name", "revision", "backend", "cache_dir", "model_cache_dir", "device", "cache_only", "projected_dim", "stub_dim", "stub_seed"}
    esm_config = ESMCConfig(**_dataclass_kwargs(sequence, esm_keys))
    geometry = HeavyFlowGeometryConfig(**_dataclass_kwargs(config.get("geometry", {}), {"num_blocks", "lmax", "hidden_irreps", "radial_basis", "radial_width", "history_scale_angstrom", "max_atomic_number", "scalar_embedding_dim", "edge_embedding_dim", "edge_chunk_size", "activation_checkpoint"}))
    graph = HeavyFlowGraphConfig(**_dataclass_kwargs(config.get("graph", {}), {"spatial_cutoff_angstrom", "max_spatial_neighbors", "include_covalent_edges", "include_chain_adjacent_edges"}))
    context = HeavyFlowContextConfig(**_dataclass_kwargs(config.get("context", {}), {"joint_scalar_dim", "fusion_blocks", "attention_heads", "pair_bias", "dropout", "geometry_modality_dropout", "atom_refine_blocks", "atom_refine_lmax", "atom_refine_radial_basis", "atom_refine_edge_embedding_dim", "atom_refine_edge_chunk_size", "atom_refine_activation_checkpoint", "attention_hidden_dim", "identity_embedding_dim"}))
    context_encoder = HeavyFlowContextEncoder(
        geometry_config=geometry,
        graph_config=graph,
        esmc_config=esm_config,
        context_config=context,
    )
    physics_data = dict(config.get("physics", {}))
    predictor_config = PhysicsPredictorConfig(**_dataclass_kwargs(physics_data, {"physics_blocks", "lmax", "scalar_dim", "vector_channels", "axial_channels", "pair_dim", "radial_basis", "edge_embedding_dim", "spatial_cutoff_angstrom", "predict_force_mean", "predict_isotropic_force_variance", "use_pair_latent", "edge_chunk_size", "activation_checkpoint"}))
    predictor = PhysicsPredictor(context_encoder.output_irreps, config=predictor_config)
    head_data = dict(config.get("force_head", {}))
    head = ForceHead(
        predictor_config.scalar_dim,
        vector_channels=predictor_config.vector_channels,
        predict_force_mean=predictor_config.predict_force_mean,
        predict_isotropic_force_variance=predictor_config.predict_isotropic_force_variance,
        **_dataclass_kwargs(head_data, {"logvar_min", "logvar_max"}),
    )
    context_encoder.required_history = HistoryConfig.from_config(config)
    model = HeavyFlowPhysicsModel(context_encoder, predictor, head)
    if device is not None:
        model.to(device)
    return model


def fit_force_normalizer(dataset: Iterable[Any]) -> ForceNormalizer:
    return ForceNormalizer.fit_from_samples(dataset)


def train_physics_steps(
    model: HeavyFlowPhysicsModel,
    samples: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    normalizer: Optional[ForceNormalizer] = None,
    steps: int = 100,
    batch_size: int = 1,
    loss_config: Optional[ForceLossConfig] = None,
) -> list[float]:
    materialized = list(samples)
    if not materialized:
        raise ValueError("cannot train Stage 3 with no samples")
    losses: list[float] = []
    model.train()
    for step in range(steps):
        start = (step * batch_size) % len(materialized)
        batch = [materialized[(start + offset) % len(materialized)] for offset in range(min(batch_size, len(materialized)))]
        sample = collate_heavy_flow(batch).to(next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        output = model(sample.condition)
        objective = force_loss(output.force_distribution, sample.targets, sample.condition, normalizer=normalizer, config=loss_config)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite Stage 3 loss at step {step}: {objective}")
        objective.backward()
        finite_optimizer_gradients(model.parameters())
        optimizer.step()
        losses.append(float(objective.detach()))
    return losses


def train_physics_stream_steps(
    model: HeavyFlowPhysicsModel,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    *,
    normalizer: Optional[ForceNormalizer] = None,
    steps: int = 100,
    batch_size: int = 1,
    loss_config: Optional[ForceLossConfig] = None,
    step_offset: int = 0,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    start_step: int = 0,
    prefetch: int = 1,
) -> list[float]:
    """Train from a dataset without materializing a whole domain chunk.

    ``HeavyFlowPhysicsFrameDataset`` reads HDF5 frames on demand.  This entry
    point keeps a bounded number of CPU batches alive, which makes a
    500-domain rotation possible even when a chunk contains many thousands of
    frames.  The frame order restarts at the beginning of each local epoch;
    ``step_offset`` identifies the chunk's initial global step. ``start_step``
    resumes at an exact optimizer-step boundary. Under DDP, batch_size is per
    rank; the tail is weighted without repeating frames in the objective.
    """
    from experiments.heavy_flow.physics_runtime import train_stream

    return train_stream(model, dataset, optimizer, normalizer=normalizer,
                        steps=steps, batch_size=batch_size, loss_config=loss_config,
                        step_offset=step_offset, progress_callback=progress_callback,
                        start_step=start_step, prefetch=prefetch)


def save_stage3_checkpoint(
    path: str | Path,
    model: HeavyFlowPhysicsModel,
    *,
    normalizer: ForceNormalizer,
    split_manifest: PhysicsSplitManifest,
    config: dict[str, Any],
    audit: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    chunk_rotation: Optional[dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    upstream_state = model.state_dict()
    payload: dict[str, Any] = {
        "stage": "stage3_atom_physics",
        "architecture_version": ARCHITECTURE_VERSION,
        "history": HistoryConfig.from_config(config).as_dict(),
        "normalizer_definition": NORMALIZER_DEFINITION,
        "upstream_training": {"status": "force_optimized" if step and step > 0 else "fixture", "optimizer_steps": int(step or 0)},
        "provenance": {"esmc": dict(model.context_encoder.geometry_encoder.sequence_encoder.provenance),
                       **getattr(model, "training_provenance", {})},
        "model_state": upstream_state,
        # Keep the Stage 3 handoff explicit.  ``HeavyFlowPhysicsModel`` is the
        # complete upstream consumed by Stage 4: context encoder (including
        # ESM-C projection, geometry, pooling, fusion, and atom refinement),
        # physics predictor, and force head.
        "upstream": upstream_state,
        "normalizer": normalizer.as_dict(),
        "split_manifest": split_manifest.to_dict(),
        "config": architecture_config(config),
        "target_audit": audit,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if step is not None:
        if step < 0:
            raise ValueError("step must be non-negative")
        payload["step"] = int(step)
    if chunk_rotation is not None:
        payload["chunk_rotation"] = chunk_rotation
    torch.save(payload, path)


def _build_reader(args: argparse.Namespace) -> MdCathDataset:
    return MdCathDataset(MdCathConfig(
        data_dir=args.data_dir,
        frames_per_trajectory=args.frames_per_trajectory,
        max_domains=args.max_domains,
        esm2_cache_dir=args.esm2_cache_dir,
        allow_fake_plm=args.allow_fake_plm,
        quarantine_path=args.quarantine_path,
        coord_quarantine_path=args.coord_quarantine_path,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/heavy_flow/stage3.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--esm2-cache-dir", default=None)
    parser.add_argument("--quarantine-path", default=None)
    parser.add_argument("--coord-quarantine-path", default=None)
    parser.add_argument("--allow-fake-plm", action="store_true")
    parser.add_argument("--frames-per-trajectory", type=int, default=2)
    parser.add_argument("--max-domains", type=int, default=30)
    parser.add_argument("--max-train-frames", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--normalizer-path", default="outputs/heavy_flow/stage3/chunk_rotation_normalizer.json")
    parser.add_argument("--fit-normalizer", action="store_true", help="explicitly fit only selected small train samples")
    parser.add_argument("--warm-start", default=None, help="explicit compatible-weight reuse; Stage 3 force training is still required")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="outputs/heavy_flow/stage3_v2/checkpoint.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = architecture_config(load_yaml(args.config))
    reader = _build_reader(args)
    selected_indices = reader.index[:args.max_train_frames * 2]
    manifest = build_physics_splits(
        type("IndexReader", (), {"index": selected_indices})(),
        same_domain_fraction=float(config["data"]["same_domain_validation_fraction"]),
        unseen_domain_fraction=float(config["data"]["unseen_domain_validation_fraction"]),
        temporal_block_size=int(config["data"]["temporal_block_size"]),
        seed=int(config["data"]["seed"]),
    )
    train_data = HeavyFlowPhysicsFrameDataset(reader, manifest.train[:args.max_train_frames], history_config=HistoryConfig.from_config(config))
    train_samples = [train_data[i] for i in range(len(train_data))]
    audit = audit_force_targets(train_samples, split_name="train").to_dict()
    normalizer, normalizer_provenance = (fit_force_normalizer(train_samples), {"source": "selected_train_samples"}) if args.fit_normalizer else load_normalizer_artifact(args.normalizer_path)
    model = build_stage3_model(config, device=args.device)
    if args.warm_start:
        report = warm_start_stage3(model, args.warm_start)
        report_path = Path(args.output).with_suffix(".warm_start.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"warm_start_report": str(report_path), "loaded": len(report["loaded"]), "not_loaded": len(report["not_loaded"]), "requires_stage3_force_training": True}))
    model.training_provenance = {**getattr(model, "training_provenance", {}), "normalizer": normalizer_provenance}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    losses = train_physics_steps(model, train_samples, optimizer, normalizer=normalizer, steps=args.steps, batch_size=args.batch_size, loss_config=ForceLossConfig(**config.get("loss", {})))
    save_stage3_checkpoint(args.output, model, normalizer=normalizer, split_manifest=manifest, config=config, audit=audit, optimizer=optimizer, step=len(losses), metrics={"loss_first": losses[0], "loss_last": losses[-1], "steps": len(losses)})
    print(json.dumps({"output": args.output, "train_frames": len(train_samples), "audit_passed": audit["passed"], "loss_first": losses[0], "loss_last": losses[-1]}, indent=2))


if __name__ == "__main__":
    main()
