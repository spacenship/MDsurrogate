#!/usr/bin/env python
"""Re-evaluate saved Phase 1.6 arms over the whole validation set. No retraining.

    python scripts/reevaluate_phase1_6.py --run runs/phase1_6_bounded_seed0 \
        --config configs/phase1_6_bounded.yaml

Phase 1.5 needed exactly this once before (report §5.0): the measurement was
wrong while the models were fine, and re-running training would have thrown away
the thing that was correct. The same applies here -- the Stage B records were
written before ``pair_id`` was recorded, and without a unique id a paired
analysis silently collapses every replica and frame of a trajectory onto one row.

What this does **not** do is touch a weight. It loads each arm's ``last.pt``,
rebuilds the identical manifest from the config, evaluates the full validation
loader and rewrites ``val_records.json`` and ``val_summary.json``. The
checkpoints, the provenance and the training history are left as they were.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import CANONICAL_ARMS  # noqa: E402
from train_transition import (  # noqa: E402
    build_configs,
    build_datasets,
    build_extractor,
    make_loader,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="a Phase 1.6 run directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    arms = args.arms or [
        d for d in sorted(os.listdir(args.run))
        if d in CANONICAL_ARMS and os.path.exists(os.path.join(args.run, d, "last.pt"))
    ]
    if not arms:
        raise SystemExit(f"no arm checkpoints found under {args.run}")

    for canonical in arms:
        arm_dir = os.path.join(args.run, canonical)
        provenance = json.loads(Path(arm_dir, "provenance.json").read_text())
        implementation = provenance["arm"]
        print(f"\n{'=' * 70}\n{canonical} -> {implementation}\n{'=' * 70}", flush=True)

        pairs, _, train_config, data = build_configs(
            yaml.safe_load(open(args.config)), implementation
        )
        seed = provenance["seed"]
        pairs = type(pairs)(**{**pairs.__dict__, "seed": seed})

        # Rebuild into the arm's own directory so the manifest files that are
        # written are the ones this evaluation used, then assert the manifest is
        # the one the checkpoint was trained against.
        train_ds, val_ds, train_manifest, val_manifest = build_datasets(
            pairs, data, arm_dir, log=lambda *a: None
        )
        if train_manifest.content_hash() != provenance["manifest_hash"]:
            raise SystemExit(
                f"{canonical}: rebuilt manifest {train_manifest.content_hash()[:12]} "
                f"!= the one this checkpoint trained on "
                f"{provenance['manifest_hash'][:12]}. Refusing to re-evaluate "
                "against different data."
            )
        val_loader = make_loader(
            val_ds, data.get("batch_size", 4), shuffle=False,
            workers=data.get("num_workers", 0), seed=seed,
        )

        extractor = build_extractor(raw, implementation, args.device)
        _, trainer = TransitionTrainer.load_checkpoint(
            os.path.join(arm_dir, "last.pt"), extractor, device=args.device
        )
        summary, records = trainer.evaluate(val_loader, split="val")
        missing = [r for r in records if "pair_id" not in r]
        if missing:
            raise SystemExit(
                f"{canonical}: {len(missing)} records still carry no pair_id"
            )
        Path(arm_dir, "val_records.json").write_text(json.dumps(records, indent=1))
        Path(arm_dir, "val_summary.json").write_text(
            json.dumps({"summary": summary}, indent=1, default=str)
        )
        unique = len({r["pair_id"] for r in records})
        print(
            f"step {trainer.step} | {len(records)} records, {unique} unique ids, "
            f"{len({r['domain'] for r in records})} domains", flush=True
        )
        for row in summary.get("rows", []):
            print(
                f"  lag {row['lag_ns']:g} ns | Ca RMSD {row['ca_rmsd_micro']:.4f} "
                f"(identity {row['ca_rmsd_identity_micro']:.4f}) | rotation "
                f"{row['rotation_geodesic_deg_micro']:.3f} "
                f"(identity {row['rotation_geodesic_deg_identity_micro']:.3f})",
                flush=True,
            )
        train_ds.close()
        val_ds.close()
    print(f"\nre-evaluated {len(arms)} arms in {args.run}; no weights were changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
