#!/usr/bin/env python
"""Run the A-E Phase 1.5 ablation and write one tidy results table.

    python scripts/run_phase1_5_ablation.py --config configs/phase1_5_smoke.yaml

Every arm is trained in this one process, in order, from the **same** manifest,
the same seed, the same batch order, the same optimiser and the same step budget.
The runner asserts that rather than assuming it: each arm's manifest hash and
Phase 1 checkpoint hash are compared against the first arm's, and a mismatch
aborts. An ablation whose arms saw different data is not an ablation.

Outputs, in ``--out-dir``:

    results.csv / results.json   one row per (arm, seed, lag, split)
    <arm>/                       provenance, history, per-pair records, checkpoint

Nothing here declares a winner. The decision gates in ``docs/phase1_5_design.md``
need three seeds and domain-level confidence intervals; a single-seed ordering is
an observation, not a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import (  # noqa: E402
    CONDITIONER_ARMS,
    FrozenPhase1Extractor,
    TransitionProbe,
)
from train_transition import build_configs, build_datasets, make_loader  # noqa: E402

#: Order matters only for readability; the arms are independent.
DEFAULT_ARMS = (
    "structure_only",
    "force_torque",
    "physics_latent",
    "force_pattern_shape",
    "oracle_force",
)

#: Columns the Phase 1.5 plan requires, in order.
COLUMNS = [
    "arm", "seed", "lag_ns", "split", "step",
    "parameter_count", "trainable_parameter_count", "conditioner_parameters",
    "domain_count", "pair_count",
    "ca_rmsd", "ca_rmsd_identity", "ca_rmsd_relative",
    "translation_rmse", "translation_rmse_identity",
    "rotation_geodesic_deg", "rotation_geodesic_deg_identity", "rotation_relative",
    "pair_distance_mae", "pair_distance_mae_identity",
    "contact_f1", "contact_f1_identity",
    "clash_rate", "clash_rate_target",
    "phi_mae_deg", "psi_mae_deg",
    "train_loss", "val_loss",
    "aggregation", "manifest_hash", "phase1_sha256",
]


def rows_for(arm: str, seed: int, step: int, provenance: dict, summary: dict,
             train_loss: float, split: str) -> list[dict]:
    """One row per (lag, aggregation). Micro and macro are both reported."""
    out = []
    for row in summary.get("rows", []):
        for aggregation in ("micro", "domain_macro"):
            def value(name: str):
                return row.get(f"{name}_{aggregation}", float("nan"))

            out.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "lag_ns": row.get("lag_ns"),
                    "split": split,
                    "step": step,
                    "parameter_count": provenance["parameter_count"],
                    "trainable_parameter_count": provenance["trainable_parameter_count"],
                    "conditioner_parameters": provenance["parameter_breakdown"]["conditioner"],
                    "domain_count": row.get("domain_count"),
                    "pair_count": row.get("graph_count"),
                    "ca_rmsd": value("ca_rmsd"),
                    "ca_rmsd_identity": value("ca_rmsd_identity"),
                    "ca_rmsd_relative": _ratio(value("ca_rmsd"), value("ca_rmsd_identity")),
                    "translation_rmse": value("translation_rmse"),
                    "translation_rmse_identity": value("translation_rmse_identity"),
                    "rotation_geodesic_deg": value("rotation_geodesic_deg"),
                    "rotation_geodesic_deg_identity": value("rotation_geodesic_deg_identity"),
                    "rotation_relative": _ratio(
                        value("rotation_geodesic_deg"),
                        value("rotation_geodesic_deg_identity"),
                    ),
                    "pair_distance_mae": value("pair_distance_mae"),
                    "pair_distance_mae_identity": value("pair_distance_mae_identity"),
                    "contact_f1": value("contact_f1"),
                    "contact_f1_identity": value("contact_f1_identity"),
                    "clash_rate": value("clash_rate"),
                    "clash_rate_target": value("clash_rate_target"),
                    "phi_mae_deg": value("phi_mae_deg"),
                    "psi_mae_deg": value("psi_mae_deg"),
                    "train_loss": train_loss,
                    "val_loss": summary.get("loss_total", float("nan")),
                    "aggregation": aggregation,
                    "manifest_hash": provenance["manifest_hash"],
                    "phase1_sha256": provenance["phase1_sha256"],
                }
            )
    return out


def _ratio(value, baseline):
    if value != value or baseline != baseline or baseline == 0:
        return float("nan")
    return value / baseline


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(root / "configs" / "phase1_5_smoke.yaml"))
    parser.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    unknown = [a for a in args.arms if a not in CONDITIONER_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; available {sorted(CONDITIONER_ARMS)}")

    raw = yaml.safe_load(open(args.config))
    name = Path(args.config).stem
    seed = args.seed if args.seed is not None else raw["train"].get("seed", 0)
    out_dir = args.out_dir or str(root / "runs" / f"{name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    reference: dict | None = None
    started = time.time()

    for arm in args.arms:
        print(f"\n{'=' * 70}\narm: {arm}\n{'=' * 70}", flush=True)
        pairs, probe_config, train_config, data = build_configs(
            yaml.safe_load(open(args.config)), arm
        )
        overrides = {"seed": seed}
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps
        if args.device is not None:
            overrides["device"] = args.device
        train_config = type(train_config)(**{**train_config.__dict__, **overrides})
        pairs = type(pairs)(**{**pairs.__dict__, "seed": seed})

        arm_dir = os.path.join(out_dir, arm)
        os.makedirs(arm_dir, exist_ok=True)
        train_ds, val_ds, train_manifest, _ = build_datasets(pairs, data, arm_dir)
        train_loader = make_loader(
            train_ds, data.get("batch_size", 4), shuffle=True,
            workers=data.get("num_workers", 0), seed=seed,
        )
        val_loader = make_loader(
            val_ds, data.get("batch_size", 4), shuffle=False,
            workers=data.get("num_workers", 0), seed=seed,
        )

        extractor = FrozenPhase1Extractor.from_checkpoint(
            raw["phase1"]["checkpoint"], device=train_config.device,
            expect=raw["phase1"].get("expect_contract"),
        )
        probe = TransitionProbe(
            probe_config, latent_irreps=extractor.contract["physics_latent_irreps"]
        )
        trainer = TransitionTrainer(probe, extractor, train_config, manifest=train_manifest)
        provenance = trainer.provenance()

        # Fairness is asserted, not assumed.
        fingerprint = (provenance["manifest_hash"], provenance["phase1_sha256"],
                       provenance["seed"], provenance["train"]["max_steps"])
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise SystemExit(
                f"arm {arm} would run a different experiment than the first arm:\n"
                f"  this arm  {fingerprint}\n  first arm {reference}\n"
                "Refusing to produce a comparison table from mismatched runs."
            )

        print(
            f"parameters {provenance['parameter_count']:,} "
            f"(conditioner {provenance['parameter_breakdown']['conditioner']:,}) "
            f"| pairs {len(train_manifest)} | manifest {provenance['manifest_hash'][:12]}",
            flush=True,
        )
        Path(arm_dir, "provenance.json").write_text(
            json.dumps(provenance, indent=1, default=str)
        )

        if args.dry_run:
            train_ds.close()
            val_ds.close()
            continue

        checkpoint = os.path.join(arm_dir, "last.pt")
        history = trainer.fit(train_loader, val_loader, checkpoint_path=checkpoint)
        Path(arm_dir, "history.json").write_text(json.dumps(history, indent=1))

        summary, records = trainer.evaluate(val_loader, split="val")
        Path(arm_dir, "val_records.json").write_text(json.dumps(records, indent=1))
        train_summary, _ = trainer.evaluate(train_loader, split="train", max_batches=20)

        results.extend(
            rows_for(arm, seed, trainer.step, provenance, summary,
                     train_summary.get("loss_total", float("nan")), "val")
        )
        results.extend(
            rows_for(arm, seed, trainer.step, provenance, train_summary,
                     train_summary.get("loss_total", float("nan")), "train")
        )
        train_ds.close()
        val_ds.close()

    if results:
        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        Path(out_dir, "results.json").write_text(json.dumps(results, indent=1))

        print(f"\n{'=' * 70}\nvalidation, micro-averaged ({time.time() - started:.0f}s)\n{'=' * 70}")
        print(f"{'arm':22s} {'lag':>5s} {'CaRMSD':>8s} {'base':>8s} {'rel':>6s} "
              f"{'rot':>7s} {'base':>7s} {'rel':>6s}")
        for row in results:
            if row["split"] != "val" or row["aggregation"] != "micro":
                continue
            print(
                f"{row['arm']:22s} {row['lag_ns']:>4.0f}n {row['ca_rmsd']:8.4f} "
                f"{row['ca_rmsd_identity']:8.4f} {row['ca_rmsd_relative']:6.3f} "
                f"{row['rotation_geodesic_deg']:7.2f} "
                f"{row['rotation_geodesic_deg_identity']:7.2f} "
                f"{row['rotation_relative']:6.3f}"
            )
        print("\nrelative < 1.0 beats the identity baseline. A single seed decides nothing;")
        print("see the decision gates in docs/phase1_5_design.md.")
    print(f"\nresults -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
