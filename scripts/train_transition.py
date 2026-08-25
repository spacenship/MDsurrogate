#!/usr/bin/env python
"""Train one arm of the Phase 1.5 transition probe.

    python scripts/train_transition.py --config configs/phase1_5_smoke.yaml --arm physics_latent

The arm is the only thing that may differ between runs of one ablation. Every
other input -- manifest, split, seed, batch order, optimiser, schedule, step
budget, backbone -- comes from the config and is recorded, hashed, in the
checkpoint and in the results row.

GPU policy on this machine: short runs on 0-3 (and only when nothing else is on
them), long training runs on 4-7. Pass ``CUDA_VISIBLE_DEVICES`` explicitly;
``--device`` names a *visible* index, never a physical one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_md.data.adapters.lag_pairs import (  # noqa: E402
    LagPairConfig,
    LagPairDataset,
    LagPairManifest,
    MdCathConfig,
    build_lag_pair_manifest,
    collate_lag_pairs,
    restore_phase1_split,
)
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.training.transition_module import (  # noqa: E402
    TransitionTrainConfig,
    TransitionTrainer,
)
from force_md.transition import (  # noqa: E402
    ConditionerConfig,
    FrozenPhase1Extractor,
    MetricConfig,
    TransitionLossWeights,
    TransitionProbe,
    TransitionProbeConfig,
)


def build_configs(raw: dict, arm: str | None):
    data, model, train = raw["data"], raw["model"], raw["train"]
    mdcath = MdCathConfig(
        data_dir=data["data_dir"],
        esm2_cache_dir=data.get("esm2_cache_dir"),
        quarantine_path=data.get("quarantine_path"),
        coord_quarantine_path=data.get("coord_quarantine_path"),
        represented_scope=data.get("represented_scope", "heavy_atom"),
        temperatures=_tuple(data.get("temperatures")),
        replicas=_tuple(data.get("replicas")),
        ps_per_frame=data["ps_per_frame"],
        max_residues=data.get("max_residues"),
        allow_fake_plm=data.get("allow_fake_plm", False),
        check_pbc=data.get("check_pbc", True),
    )
    pairs = LagPairConfig(
        mdcath=mdcath,
        lags_ps=tuple(data.get("lags_ps", (1000.0, 4000.0))),
        history_length=data.get("history_length", 2),
        require_current_force_labels=data.get("require_current_force_labels", True),
        pairs_per_trajectory=data.get("pairs_per_trajectory"),
        max_trajectories_per_domain=data.get("max_trajectories_per_domain"),
        max_domains=data.get("max_domains"),
        max_pairs=data.get("max_pairs"),
        selection=data.get("selection", "even"),
        seed=data.get("split_seed", 0),
    )
    probe = TransitionProbeConfig(
        arm=arm or model.get("arm", "structure_only"),
        conditioner=ConditionerConfig(**model.get("conditioner", {})),
        irreps=IrrepsConfig(**model.get("irreps", {})),
        num_blocks=model.get("num_blocks", 3),
        history_length=data.get("history_length", 2),
        residue_knn=model.get("residue_knn", 16),
        sequence_max_offset=model.get("sequence_max_offset", 2),
        backbone_cutoff=model.get("backbone_cutoff", 13.0),
        use_plm=model.get("use_plm", True),
        use_temperature=model.get("use_temperature", True),
    )
    training = TransitionTrainConfig(
        loss_weights=TransitionLossWeights(**train.pop("loss_weights", {})),
        metrics=MetricConfig(**train.pop("metrics", {})),
        **train,
    )
    return pairs, probe, training, data


def _tuple(value):
    return None if value is None else tuple(value)


def build_datasets(pairs: LagPairConfig, data: dict, out_dir: str, log=print):
    """Restore Phase 1's split, then build one manifest per side."""
    train_domains, val_domains = restore_phase1_split(
        pairs.mdcath,
        snapshot_path=data.get("phase1_split_snapshot"),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 0),
    )
    if data.get("max_domains"):
        keep = int(data["max_domains"])
        train_domains, val_domains = train_domains[:keep], val_domains[: max(1, keep // 4)]
    log(f"domains: {len(train_domains)} train / {len(val_domains)} val (Phase 1 split)")

    train_manifest = build_lag_pair_manifest(pairs, train_domains, split="train")
    val_manifest = build_lag_pair_manifest(pairs, val_domains, split="val")
    train_manifest.assert_disjoint(val_manifest)
    log(
        f"pairs: {len(train_manifest)} train / {len(val_manifest)} val "
        f"| by lag {train_manifest.counts_by_lag()} "
        f"| manifest {train_manifest.content_hash()[:12]}"
    )
    os.makedirs(out_dir, exist_ok=True)
    train_manifest.save(os.path.join(out_dir, "manifest_train.json"))
    val_manifest.save(os.path.join(out_dir, "manifest_val.json"))
    return (
        LagPairDataset(pairs, train_manifest),
        LagPairDataset(pairs, val_manifest),
        train_manifest,
        val_manifest,
    )


def make_loader(dataset, batch_size, *, shuffle, workers, seed):
    kwargs = dict(collate_fn=collate_lag_pairs, num_workers=workers,
                  persistent_workers=workers > 0)
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, **kwargs
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(root / "configs" / "phase1_5_smoke.yaml"))
    parser.add_argument("--arm", default=None, help="overrides model.arm")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="build everything and report sizes without training")
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    pairs, probe_config, train_config, data = build_configs(raw, args.arm)
    if args.seed is not None:
        train_config = type(train_config)(**{**train_config.__dict__, "seed": args.seed})
        pairs = type(pairs)(**{**pairs.__dict__, "seed": args.seed})
    if args.max_steps is not None:
        train_config = type(train_config)(**{**train_config.__dict__,
                                             "max_steps": args.max_steps})
    if args.device is not None:
        train_config = type(train_config)(**{**train_config.__dict__, "device": args.device})

    out_dir = args.out_dir or str(
        root / "runs" / f"phase1_5_{probe_config.arm}_seed{train_config.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    log = print

    train_ds, val_ds, train_manifest, _ = build_datasets(pairs, data, out_dir, log=log)
    batch_size = data.get("batch_size", 4)
    workers = data.get("num_workers", 0)
    train_loader = make_loader(train_ds, batch_size, shuffle=True, workers=workers,
                               seed=train_config.seed)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, workers=workers,
                             seed=train_config.seed)

    extractor = FrozenPhase1Extractor.from_checkpoint(
        raw["phase1"]["checkpoint"],
        device=train_config.device,
        expect=raw["phase1"].get("expect_contract"),
    )
    probe = TransitionProbe(
        probe_config, latent_irreps=extractor.contract["physics_latent_irreps"]
    )
    trainer = TransitionTrainer(probe, extractor, train_config, manifest=train_manifest)

    provenance = trainer.provenance()
    log(
        f"arm {probe_config.arm} | parameters {provenance['parameter_count']:,} "
        f"(conditioner {provenance['parameter_breakdown']['conditioner']:,}, "
        f"backbone {provenance['parameter_breakdown']['blocks']:,}) "
        f"| device {train_config.device}"
    )
    Path(out_dir, "provenance.json").write_text(json.dumps(provenance, indent=1, default=str))

    if args.resume:
        trainer.load_state(args.resume)
        log(f"resumed from {args.resume} at step {trainer.step}")

    if args.dry_run:
        log("dry-run: nothing trained")
        train_ds.close()
        val_ds.close()
        return 0

    checkpoint = os.path.join(out_dir, "last.pt")
    history = trainer.fit(train_loader, val_loader, log=log, checkpoint_path=checkpoint)
    Path(out_dir, "history.json").write_text(json.dumps(history, indent=1))

    summary, records = trainer.evaluate(val_loader, split="val")
    Path(out_dir, "val_records.json").write_text(json.dumps(records, indent=1))
    Path(out_dir, "val_summary.json").write_text(
        json.dumps({"provenance": provenance, "summary": summary}, indent=1, default=str)
    )
    log(f"\nfinal validation ({probe_config.arm}):")
    for row in summary.get("rows", []):
        log(
            f"  lag {row['lag_ns']:g} ns | Ca RMSD {row['ca_rmsd_micro']:.4f} "
            f"(identity {row['ca_rmsd_identity_micro']:.4f}) "
            f"| rotation {row['rotation_geodesic_deg_micro']:.2f} deg "
            f"(identity {row['rotation_geodesic_deg_identity_micro']:.2f})"
        )
    log(f"\ncheckpoint -> {checkpoint}")
    train_ds.close()
    val_ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
