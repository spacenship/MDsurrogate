"""Bounded Stage 4 training entry point for direct heavy-atom rectified flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow import (
    HeavyAtomFlowModel,
    collate_heavy_flow,
)
from force_md.heavy_flow.flow_decoder import FlowDecoder, FlowDecoderConfig
from force_md.heavy_flow.geometry_losses import (
    GeometryLossConfig,
    combined_flow_loss,
    compute_geometry_regularization,
    geometry_gradient_norms,
)
from force_md.heavy_flow.physics_dataset import HeavyFlowTemporalPairDataset
from force_md.heavy_flow.history import HistoryConfig
from force_md.heavy_flow.checkpoint import (ARCHITECTURE_VERSION, NORMALIZER_DEFINITION, architecture_config, validate_checkpoint, finite_optimizer_gradients)
from force_md.heavy_flow.rectified_flow import build_flow_path, endpoint_from_velocity, rectified_flow_loss

from experiments.heavy_flow.train_physics import build_stage3_model, load_yaml

__all__ = [
    "build_from_stage3_config",
    "load_stage3_upstream_checkpoint",
    "build_stage4_model",
    "train_flow_steps",
    "save_stage4_checkpoint",
    "load_stage4_checkpoint",
]


def _kwargs(values: dict[str, Any], names: set[str]) -> dict[str, Any]:
    return {name: values[name] for name in names if name in values}


_UPSTREAM_COMPONENT_PREFIXES = {
    "ESM-C trainable projection": "context_encoder.geometry_encoder.sequence_encoder.projection.",
    "atom geometry encoder": "context_encoder.geometry_encoder.",
    "atom-to-residue pooling": "context_encoder.pool.",
    "residue global context": "context_encoder.fusion.",
    "residue-to-atom refinement": "context_encoder.refiner.",
    "physics predictor": "predictor.",
    "force-trained pair projection": "predictor.pair_projection.",
    "force head": "force_head.",
}


def _config_for_device(config: Mapping[str, Any], device: Optional[str]) -> dict[str, Any]:
    """Copy an upstream config and prevent a stale checkpoint device setting."""
    result = dict(config)
    sequence = dict(result.get("sequence_encoder", {}))
    if device is not None:
        sequence["device"] = device
    result["sequence_encoder"] = sequence
    return result


def build_from_stage3_config(
    config: Mapping[str, Any],
    *,
    device: Optional[str] = None,
):
    """Build the complete Stage 3 upstream from its serialized config.

    The returned object is intentionally the same composite used by Stage 3,
    rather than a Stage 4-specific reimplementation of its submodules.
    """
    return build_stage3_model(_config_for_device(config, device), device=device)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Optional[Path]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Stage 3 checkpoint must contain a mapping: {path}")
        return dict(payload), path
    if isinstance(source, Mapping):
        return dict(source), None
    raise TypeError("stage3_checkpoint must be a path or checkpoint mapping")


def _upstream_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("upstream")
    if state is None:
        # Both aliases are accepted only after architecture/metadata validation.
        state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("Stage 3 checkpoint has no mapping under 'upstream' or 'model_state'")
    if not state:
        raise ValueError("Stage 3 upstream state is empty")
    return state


def _validate_upstream_state(state: Mapping[str, torch.Tensor]) -> None:
    missing = [
        name for name, prefix in _UPSTREAM_COMPONENT_PREFIXES.items()
        if not any(str(key).startswith(prefix) for key in state)
    ]
    if missing:
        required = ", ".join(missing)
        raise ValueError(f"Stage 3 upstream is incomplete; missing components: {required}")


def load_stage3_upstream_checkpoint(
    source: str | Path | Mapping[str, Any],
    *,
    device: Optional[str] = None,
    allow_untrained_fixture: bool = False,
):
    """Build, strict-load, freeze, and validate the Stage 3 upstream.

    The returned tuple is ``(upstream, config, provenance)``.  ``upstream``
    contains the learned ESM-C projection, all context hierarchy modules, the
    physics predictor, and the force head.
    """
    payload, source_path = _checkpoint_payload(source)
    validate_checkpoint(payload, require_trained=True, allow_untrained_fixture=allow_untrained_fixture)
    stage = payload.get("stage")
    if stage is not None and stage != "stage3_atom_physics":
        raise ValueError(f"expected a Stage 3 checkpoint, got stage={stage!r}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage 3 checkpoint has no mapping under 'config'")
    state = _upstream_state(payload)
    _validate_upstream_state(state)
    upstream = build_from_stage3_config(config, device=device)
    upstream.load_state_dict(state, strict=True)
    upstream.requires_grad_(False)
    upstream.eval()
    sequence_encoder = upstream.context_encoder.geometry_encoder.sequence_encoder
    provenance: dict[str, Any] = {
        "stage": "stage3_atom_physics",
        "source": str(source_path) if source_path is not None else None,
        "sha256": _checkpoint_sha256(source_path) if source_path is not None else None,
        "state_key_count": len(state),
        "components": list(_UPSTREAM_COMPONENT_PREFIXES),
        "esmc": dict(sequence_encoder.provenance),
        "architecture_version": payload["architecture_version"],
        "history": payload["history"],
        "normalizer": payload["normalizer"],
        "normalizer_definition": payload["normalizer_definition"],
        "upstream_training": payload["upstream_training"],
        "training_provenance": payload.get("provenance", {}),
        "split_manifest": payload.get("split_manifest"),
        "target_audit": payload.get("target_audit"),
        "metrics": payload.get("metrics"),
    }
    if "restored_upstream_provenance" in payload:
        provenance = dict(payload["restored_upstream_provenance"])
    return upstream, dict(config), provenance


def build_stage4_model(
    config: dict[str, Any],
    *,
    stage3_checkpoint: str | Path | Mapping[str, Any] | None = None,
    device: Optional[str] = None,
    allow_untrained_fixture: bool = False,
) -> HeavyAtomFlowModel:
    """Load the complete Stage 3 upstream and attach a fresh Stage 4 decoder."""
    if stage3_checkpoint is None:
        raise ValueError(
            "Stage 4 requires a Stage 3 checkpoint; pass stage3_checkpoint "
            "to build_stage4_model or use --stage3-checkpoint"
        )
    config = architecture_config(config)
    upstream, upstream_config, upstream_provenance = load_stage3_upstream_checkpoint(
        stage3_checkpoint,
        device=device,
        allow_untrained_fixture=allow_untrained_fixture,
    )
    if HistoryConfig.from_config(config) != HistoryConfig.from_config(upstream_config):
        raise ValueError("Stage 4 history must match its pretrained upstream checkpoint")
    context = upstream.context_encoder
    predictor = upstream.predictor
    force_head = upstream.force_head
    context_config = context.context_config
    physics_config = predictor.config
    flow_data = dict(config.get("flow", {}))
    if not bool(flow_data.get("freeze_upstream", True)):
        raise ValueError("Stage 4 requires freeze_upstream=True for the Stage 3 handoff")
    flow_config = FlowDecoderConfig(**_kwargs(flow_data, {"flow_blocks", "lmax", "hidden_irreps", "conditioning_layers", "radial_basis", "edge_embedding_dim", "spatial_cutoff_angstrom", "max_spatial_neighbors", "spatial_recompute_every", "time_embedding_dim", "atom_identity_dim", "max_atomic_number"}))
    decoder = FlowDecoder(
        context.output_irreps,
        context_config.joint_scalar_dim,
        physics_config.scalar_dim,
        physics_vector_channels=physics_config.vector_channels,
        physics_axial_channels=physics_config.axial_channels,
        physics_pair_dim=physics_config.pair_dim if physics_config.use_pair_latent else 0,
        config=flow_config,
    )
    model = HeavyAtomFlowModel(
        context, predictor, force_head, decoder,
        freeze_upstream=bool(flow_data.get("freeze_upstream", True)),
        sigma_scale=float(flow_data.get("sigma_scale", 0.1)),
        remove_center_of_mass_noise=bool(flow_data.get("remove_center_of_mass_noise", True)),
        upstream_config=upstream_config,
        upstream_provenance=upstream_provenance,
    )
    # Make the handoff order explicit even though HeavyAtomFlowModel repeats
    # the freeze for its three registered upstream children.
    upstream.requires_grad_(False)
    upstream.eval()
    if device is not None:
        model.to(device)
    return model


def train_flow_steps(
    model: HeavyAtomFlowModel,
    samples: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 100,
    batch_size: int = 1,
    sigma_scale: Optional[float] = None,
    geometry_config: Optional[GeometryLossConfig] = None,
    geometry_scale: float = 1.0,
    seed: Optional[int] = None,
    step_offset: int = 0,
    report_gradient_norms: bool = True,
) -> list[dict[str, float]]:
    """Train only adapters/decoder on target-constructed RF paths.

    ``x_future`` is read in this function solely to make ``FlowPath``.  It is
    never passed to ``model.encode_condition`` or ``FlowDecoder.forward``.
    """
    # Datasets remain lazy: only the current optimizer batch is read.
    materialized = samples if hasattr(samples, "__getitem__") and hasattr(samples, "__len__") else list(samples)
    if not materialized:
        raise ValueError("cannot train Stage 4 with no samples")
    if steps < 1 or batch_size < 1 or step_offset < 0:
        raise ValueError("steps and batch_size must be positive and step_offset non-negative")
    first_parameter = next(model.parameters(), None)
    model_device = (
        first_parameter.device
        if first_parameter is not None
        else materialized[0].condition.current_positions.device
    )
    if sigma_scale is None:
        sigma_scale = model.sigma_scale
    geometry_config = geometry_config or GeometryLossConfig()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=model_device)
        generator.manual_seed(int(seed))
    history: list[dict[str, float]] = []
    model.train()
    for step in range(steps):
        global_step = step_offset + step
        start = (global_step * batch_size) % len(materialized)
        batch = [materialized[(start + offset) % len(materialized)] for offset in range(min(batch_size, len(materialized)))]
        # Dataset samples are intentionally allowed to remain on CPU.  Move
        # the complete collated contract together so ESM-C, geometry, targets,
        # and the RF noise generator all use the model's device.
        sample = collate_heavy_flow(batch).to(model_device)
        condition, target = sample.condition, sample.targets
        bundle = model.encode_condition(condition)
        future_mask = target.future_mask if target.future_mask is not None else condition.atom_mask
        path = build_flow_path(
            condition.current_positions,
            target.x_future,
            atom_mask=condition.atom_mask,
            future_mask=future_mask,
            lag=condition.lag,
            sigma_scale=sigma_scale,
            remove_center_of_mass_noise=model.remove_center_of_mass_noise,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        velocity = model.velocity(bundle, path.x_s, path.flow_time, recompute_spatial=True)
        flow = rectified_flow_loss(velocity, path.target_velocity, path.mask)
        endpoint = endpoint_from_velocity(path.x_s, path.flow_time, velocity)
        endpoint = torch.where(path.mask[..., None], endpoint, condition.current_positions)
        geometry = compute_geometry_regularization(endpoint, path.x1, bundle.atom_topology, config=geometry_config)
        objective = combined_flow_loss(flow, geometry, geometry_scale=geometry_scale)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite Stage 4 loss at step {global_step}: {objective}")
        diagnostics = geometry_gradient_norms(geometry, model.decoder.parameters()) if report_gradient_norms else {}
        objective.backward()
        finite_optimizer_gradients(model.decoder.parameters())
        optimizer.step()
        metrics = {"step": float(global_step), "loss": float(objective.detach()), "flow_loss": float(flow.detach())}
        metrics.update({f"geometry_{name}": value for name, value in geometry.as_dict().items()})
        metrics.update({f"geometry_grad_{name}": value for name, value in diagnostics.items()})
        history.append(metrics)
    return history


def save_stage4_checkpoint(
    path: str | Path,
    model: HeavyAtomFlowModel,
    *,
    config: dict[str, Any],
    metrics: Optional[dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
) -> None:
    """Save a Stage 4 model and, when supplied, resumable optimizer state."""
    if step is not None and step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if model.upstream_config is None or model.upstream_provenance is None:
        raise ValueError("Stage 4 checkpoint requires a Stage 3-loaded upstream with provenance")
    if not model.upstream_is_frozen:
        raise ValueError("cannot save Stage 4 checkpoint with a trainable upstream")
    model_state = model.state_dict()
    payload: dict[str, Any] = {
        "stage": "stage4_heavy_atom_flow",
        "architecture_version": ARCHITECTURE_VERSION,
        "history": HistoryConfig.from_config(model.upstream_config).as_dict(),
        "normalizer": model.upstream_provenance["normalizer"],
        "normalizer_definition": NORMALIZER_DEFINITION,
        "upstream_training": model.upstream_provenance["upstream_training"],
        "model_state": model_state,
        "upstream": {
            key: value for key, value in model_state.items()
            if key.startswith(("context_encoder.", "predictor.", "force_head."))
        },
        "upstream_config": model.upstream_config,
        "upstream_provenance": model.upstream_provenance,
        "config": architecture_config(config),
        "metrics": metrics,
        "upstream_frozen": model.upstream_is_frozen,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = int(step)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_stage4_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    stage3_checkpoint: str | Path | Mapping[str, Any] | None = None,
    allow_untrained_fixture: bool = False,
) -> tuple[HeavyAtomFlowModel, dict[str, Any]]:
    """Load a Stage 4 checkpoint onto ``device`` for inference or resuming.

    Version 2 checkpoints are self-contained; legacy architectures are rejected.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("stage") != "stage4_heavy_atom_flow":
        raise ValueError("checkpoint is not a Stage 4 heavy-atom flow checkpoint")
    validate_checkpoint(checkpoint, require_trained=True, allow_untrained_fixture=allow_untrained_fixture)
    config = checkpoint["config"]
    embedded_upstream = (
        "upstream" in checkpoint
        and isinstance(checkpoint.get("upstream_config"), Mapping)
        and checkpoint.get("upstream_frozen") is True
    )
    if embedded_upstream:
        stage3_source: str | Path | Mapping[str, Any] = {
            "stage": "stage3_atom_physics",
            "config": checkpoint["upstream_config"],
            "upstream": checkpoint["upstream"],
            "architecture_version": checkpoint["architecture_version"],
            "history": checkpoint["history"],
            "normalizer": checkpoint["normalizer"],
            "normalizer_definition": checkpoint["normalizer_definition"],
            "upstream_training": checkpoint["upstream_training"],
            "restored_upstream_provenance": checkpoint["upstream_provenance"],
        }
    elif stage3_checkpoint is not None:
        stage3_source = stage3_checkpoint
    else:
        raise ValueError(
            "Stage 4 checkpoint does not embed a Stage 3 upstream/config; "
            "pass stage3_checkpoint (or --stage3-checkpoint) to resume it"
        )
    model = build_stage4_model(config, stage3_checkpoint=stage3_source, device=device, allow_untrained_fixture=allow_untrained_fixture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _build_reader(args: argparse.Namespace) -> MdCathDataset:
    return MdCathDataset(MdCathConfig(
        data_dir=args.data_dir,
        esm2_cache_dir=args.esm2_cache_dir,
        represented_scope="heavy_atom",
        frames_per_trajectory=args.frames_per_trajectory,
        max_domains=args.max_domains,
        allow_fake_plm=args.allow_fake_plm,
        check_pbc=True,
        quarantine_path=args.quarantine_path,
        coord_quarantine_path=args.coord_quarantine_path,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/heavy_flow/stage4.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--esm2-cache-dir", default=None)
    parser.add_argument("--quarantine-path", default=None)
    parser.add_argument("--coord-quarantine-path", default=None)
    parser.add_argument("--allow-fake-plm", action="store_true")
    parser.add_argument("--frames-per-trajectory", type=int, default=2)
    parser.add_argument("--max-domains", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="outputs/heavy_flow/stage4_v2/checkpoint.pt")
    parser.add_argument("--resume", default=None, help="resume from a Stage 4 checkpoint")
    parser.add_argument("--stage3-checkpoint", default=None, help="learned Stage 3 upstream checkpoint for a new Stage 4 run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-gradient-diagnostics", action="store_true")
    args = parser.parse_args()
    config = architecture_config(load_yaml(args.config))
    checkpoint: Optional[dict[str, Any]] = None
    if args.resume is not None:
        model, checkpoint = load_stage4_checkpoint(
            args.resume,
            device=args.device,
            stage3_checkpoint=args.stage3_checkpoint,
        )
        config = checkpoint["config"]
    if checkpoint is None:
        stage3_checkpoint = args.stage3_checkpoint or config.get("stage3_checkpoint")
        model = build_stage4_model(config, stage3_checkpoint=stage3_checkpoint, device=args.device)
    reader = _build_reader(args)
    dataset = HeavyFlowTemporalPairDataset(reader, reader.index, history_config=HistoryConfig.from_config(model.upstream_config), future_lag_frames=config.get("data", {}).get("future_lag_frames", 1))
    samples = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=args.lr)
    start_step = 0
    if checkpoint is not None:
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
    flow_data = config.get("flow", {})
    geometry = GeometryLossConfig(**_kwargs(flow_data.get("geometry", {}), set(GeometryLossConfig.__dataclass_fields__)) if isinstance(flow_data.get("geometry"), dict) else {})
    logs = train_flow_steps(model, samples, optimizer, steps=args.steps, batch_size=args.batch_size, geometry_config=geometry, geometry_scale=float(flow_data.get("geometry_scale", 1.0)), step_offset=start_step, report_gradient_norms=not args.no_gradient_diagnostics)
    save_stage4_checkpoint(
        args.output,
        model,
        config=config,
        metrics={"steps": len(logs), "loss_first": logs[0]["loss"], "loss_last": logs[-1]["loss"]},
        optimizer=optimizer,
        step=start_step + len(logs),
    )
    print(json.dumps({"output": args.output, "samples": len(samples), "loss_first": logs[0]["loss"], "loss_last": logs[-1]["loss"]}, indent=2))


if __name__ == "__main__":
    main()
