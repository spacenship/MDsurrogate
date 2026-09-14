#!/usr/bin/env python
"""Run the Phase 1.6 pair-interaction ablation and write one tidy results table.

    python scripts/run_phase1_6_ablation.py --config configs/phase1_6_smoke.yaml
    python scripts/run_phase1_6_ablation.py --config configs/phase1_6_bounded.yaml

Arms are named by their **canonical** Phase 1.6 role (``P1_pair_physics_frozen``)
and resolved to the registered conditioner (``pair_physics``) through
:mod:`force_md.transition.arms`. Both names go into every row.

This extends the Phase 1.5 runner rather than replacing it: the row builder, the
dataset construction, the loaders and the fairness assertion are imported from
it, so a Phase 1.5 number and a Phase 1.6 number are produced by the same code.

What it adds:

* canonical arm names and an explicit ``oracle`` column;
* per-arm resource records -- peak GPU memory, wall time, parameter breakdown;
* every hash the plan asks for, in one ``reproducibility.json`` per run;
* ``--reuse-from``, which will only reuse an existing arm's result when the
  manifest, Phase 1 checkpoint, seed and step budget **all** match, and which
  prints why when they do not.

Nothing here declares a winner. A single seed is an observation; the statistics
live in ``scripts/analyze_phase1_6.py`` and the gates in ``docs/phase1_6_report.md``.
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
    CANONICAL_ARMS,
    SCREENING_ARMS,
    arm_spec,
)
from run_phase1_5_ablation import COLUMNS, rows_for  # noqa: E402
from train_transition import (  # noqa: E402
    build_configs,
    build_datasets,
    build_extractor,
    build_probe,
    make_loader,
)

#: Extra columns Phase 1.6 records on top of the Phase 1.5 set.
EXTRA_COLUMNS = [
    "canonical_arm", "oracle", "uses_pair_features", "future_physics",
    "conditioner_class", "peak_gpu_memory_mb", "train_wall_time_s",
    "git_commit", "git_dirty", "config_hash", "phase1_step",
]
PHASE16_COLUMNS = COLUMNS + EXTRA_COLUMNS


def config_hash(path: str) -> str:
    """SHA-256 of the config file as it was read, not of the parsed dict.

    The file is the artefact a person edits and quotes; hashing the parsed dict
    would make a comment change invisible and a key-order change significant,
    which is backwards.
    """
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decorate(rows: list[dict], provenance: dict, spec, cfg_hash: str) -> list[dict]:
    """Add the Phase 1.6 columns to every row the Phase 1.5 builder produced."""
    resources = provenance.get("resources", {})
    git = provenance.get("git", {})
    peak = resources.get("peak_gpu_memory_bytes")
    for row in rows:
        row.update(
            canonical_arm=spec.canonical,
            oracle=bool(spec.oracle),
            uses_pair_features=bool(provenance.get("uses_pair_features")),
            future_physics=bool(provenance.get("future_physics")),
            conditioner_class=provenance.get("conditioner_class", ""),
            peak_gpu_memory_mb=None if peak is None else round(peak / 2**20, 1),
            train_wall_time_s=resources.get("train_wall_time_s"),
            git_commit=git.get("commit"),
            git_dirty=git.get("dirty"),
            config_hash=cfg_hash,
            phase1_step=provenance.get("phase1_step"),
        )
    return rows


def load_reusable(path: str, arm: str, fingerprint: tuple) -> tuple[list[dict], str]:
    """Rows from an earlier run, but only if it was the same experiment.

    Returns ``(rows, reason)``. ``rows`` is empty whenever reuse is not legitimate
    and ``reason`` says which field disagreed -- a result table assembled from
    runs that merely resemble each other is the failure this whole runner exists
    to prevent.
    """
    provenance_path = os.path.join(path, arm, "provenance.json")
    results_path = os.path.join(path, "results.json")
    if not (os.path.exists(provenance_path) and os.path.exists(results_path)):
        return [], f"no saved {arm} run under {path}"
    provenance = json.loads(Path(provenance_path).read_text())
    theirs = (
        provenance.get("manifest_hash"),
        provenance.get("phase1_sha256"),
        provenance.get("seed"),
        provenance.get("train", {}).get("max_steps"),
    )
    if theirs != fingerprint:
        names = ("manifest_hash", "phase1_sha256", "seed", "max_steps")
        differences = [
            f"{n}: theirs {t!r} != ours {o!r}"
            for n, t, o in zip(names, theirs, fingerprint)
            if t != o
        ]
        return [], "; ".join(differences)
    rows = [r for r in json.loads(Path(results_path).read_text()) if r["arm"] == arm]
    return rows, "identical experiment" if rows else "no rows for this arm"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(root / "configs" / "phase1_6_smoke.yaml"))
    parser.add_argument("--arms", nargs="*", default=list(SCREENING_ARMS),
                        help="canonical names; defaults to the Stage B screening set")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-from", default=None,
                        help="run directory whose matching arms may be reused")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = [arm_spec(name) for name in args.arms]
    unknown = [n for n in args.arms if n not in CANONICAL_ARMS]
    if unknown:
        raise SystemExit(
            f"unknown canonical arm(s) {unknown}; available {sorted(CANONICAL_ARMS)}"
        )
    gated = [s.canonical for s in specs if s.stage == "gated"]
    if gated:
        raise SystemExit(
            f"{gated} is gated: its config exists but the plan forbids starting it "
            "automatically. Run it explicitly with scripts/train_transition.py once "
            "the screening result justifies it."
        )

    raw_text = yaml.safe_load(open(args.config))
    cfg_hash = config_hash(args.config)
    name = Path(args.config).stem
    seed = args.seed if args.seed is not None else raw_text["train"].get("seed", 0)
    out_dir = args.out_dir or str(root / "runs" / f"{name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    reference: tuple | None = None
    reuse_log: dict[str, str] = {}
    provenances: dict[str, dict] = {}
    started = time.time()

    for spec in specs:
        arm = spec.implementation
        print(f"\n{'=' * 74}\n{spec.canonical}  ->  {arm}"
              f"{'   [ORACLE, not a model]' if spec.oracle else ''}\n{'=' * 74}",
              flush=True)

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

        arm_dir = os.path.join(out_dir, spec.canonical)
        os.makedirs(arm_dir, exist_ok=True)
        train_ds, val_ds, train_manifest, val_manifest = build_datasets(
            pairs, data, arm_dir
        )
        train_loader = make_loader(
            train_ds, data.get("batch_size", 4), shuffle=True,
            workers=data.get("num_workers", 0), seed=seed,
        )
        val_loader = make_loader(
            val_ds, data.get("batch_size", 4), shuffle=False,
            workers=data.get("num_workers", 0), seed=seed,
        )

        extractor = build_extractor(raw_text, arm, train_config.device)
        probe = build_probe(probe_config, extractor)
        trainer = TransitionTrainer(probe, extractor, train_config, manifest=train_manifest)
        provenance = trainer.provenance()
        provenance["conditioner_class"] = type(probe.conditioner).__name__
        provenance["val_manifest_hash"] = val_manifest.content_hash()
        provenance["config_hash"] = cfg_hash
        provenance["config_path"] = os.path.abspath(args.config)

        # Fairness is asserted, not assumed -- the Phase 1.5 rule, unchanged.
        fingerprint = (provenance["manifest_hash"], provenance["phase1_sha256"],
                       provenance["seed"], provenance["train"]["max_steps"])
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise SystemExit(
                f"arm {spec.canonical} would run a different experiment than the "
                f"first arm:\n  this arm  {fingerprint}\n  first arm {reference}\n"
                "Refusing to produce a comparison table from mismatched runs."
            )

        print(
            f"parameters {provenance['parameter_count']:,} "
            f"(conditioner {provenance['parameter_breakdown']['conditioner']:,}"
            f"{', future head ' + format(provenance['parameter_breakdown'].get('future_physics_head', 0), ',') if spec.future_physics else ''}) "
            f"| pair features {provenance['uses_pair_features']} "
            f"| pairs {len(train_manifest)} "
            f"| manifest {provenance['manifest_hash'][:12]}",
            flush=True,
        )
        Path(arm_dir, "provenance.json").write_text(
            json.dumps(provenance, indent=1, default=str)
        )
        provenances[spec.canonical] = provenance

        if args.dry_run:
            train_ds.close()
            val_ds.close()
            continue

        if args.reuse_from:
            rows, reason = load_reusable(args.reuse_from, arm, fingerprint)
            reuse_log[spec.canonical] = reason
            if rows:
                print(f"reusing {len(rows)} rows from {args.reuse_from}: {reason}")
                results.extend(decorate(rows, provenance, spec, cfg_hash))
                train_ds.close()
                val_ds.close()
                continue
            print(f"not reusing {spec.canonical}: {reason}")

        checkpoint = os.path.join(arm_dir, "last.pt")
        history = trainer.fit(train_loader, val_loader, checkpoint_path=checkpoint)
        Path(arm_dir, "history.json").write_text(json.dumps(history, indent=1))

        summary, records = trainer.evaluate(val_loader, split="val")
        Path(arm_dir, "val_records.json").write_text(json.dumps(records, indent=1))
        train_summary, _ = trainer.evaluate(train_loader, split="train", max_batches=20)

        provenance = trainer.provenance()          # now carries wall time and peak memory
        provenance["conditioner_class"] = type(probe.conditioner).__name__
        provenance["val_manifest_hash"] = val_manifest.content_hash()
        provenance["config_hash"] = cfg_hash
        Path(arm_dir, "provenance.json").write_text(
            json.dumps(provenance, indent=1, default=str)
        )
        provenances[spec.canonical] = provenance

        for split, s in (("val", summary), ("train", train_summary)):
            results.extend(
                decorate(
                    rows_for(arm, seed, trainer.step, provenance, s,
                             train_summary.get("loss_total", float("nan")), split),
                    provenance, spec, cfg_hash,
                )
            )
        train_ds.close()
        val_ds.close()

    Path(out_dir, "reproducibility.json").write_text(json.dumps({
        "config": os.path.abspath(args.config),
        "config_hash": cfg_hash,
        "seed": seed,
        "arms": [s.canonical for s in specs],
        "reuse": reuse_log,
        "provenance": provenances,
    }, indent=1, default=str))

    if results:
        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PHASE16_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        Path(out_dir, "results.json").write_text(json.dumps(results, indent=1))
        with open(os.path.join(out_dir, "results.jsonl"), "w") as handle:
            for row in results:
                handle.write(json.dumps(row, default=str) + "\n")

        print(f"\n{'=' * 74}\nvalidation, micro-averaged ({time.time() - started:.0f}s)"
              f"\n{'=' * 74}")
        print(f"{'arm':32s} {'lag':>5s} {'CaRMSD':>8s} {'base':>8s} {'rel':>6s} "
              f"{'rot':>7s} {'rel':>6s} {'params':>9s}")
        for row in results:
            if row["split"] != "val" or row["aggregation"] != "micro":
                continue
            print(
                f"{row['canonical_arm']:32s} {row['lag_ns']:>4.0f}n "
                f"{row['ca_rmsd']:8.4f} {row['ca_rmsd_identity']:8.4f} "
                f"{row['ca_rmsd_relative']:6.3f} "
                f"{row['rotation_geodesic_deg']:7.2f} {row['rotation_relative']:6.3f} "
                f"{row['parameter_count']:9,d}"
            )
        print("\nrelative < 1.0 beats the identity baseline. One seed decides nothing;")
        print("run scripts/analyze_phase1_6.py for paired tests and bootstrap intervals.")
    print(f"\nresults -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
