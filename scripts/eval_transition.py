#!/usr/bin/env python
"""Evaluate a trained transition probe. No training, no weight is written.

    python scripts/eval_transition.py --checkpoint runs/phase1_5_smoke_seed0/physics_latent/last.pt \
        --config configs/phase1_5_smoke.yaml

Reports the same metrics the ablation table carries, always beside the identity
("nothing moves") baseline, split by lag and reported both micro-averaged and
domain-macro-averaged. ``--manifest`` evaluates a manifest that was written by a
previous run instead of rebuilding one, which is what makes two evaluations
comparable rather than merely similar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.data.adapters.lag_pairs import LagPairDataset, LagPairManifest  # noqa: E402
from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import FrozenPhase1Extractor  # noqa: E402
from train_transition import build_configs, build_datasets, make_loader  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(root / "configs" / "phase1_5_smoke.yaml"))
    parser.add_argument("--manifest", default=None,
                        help="evaluate this manifest instead of rebuilding one")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    pairs, _, train_config, data = build_configs(raw, None)

    extractor = FrozenPhase1Extractor.from_checkpoint(
        raw["phase1"]["checkpoint"], device=args.device,
        expect=raw["phase1"].get("expect_contract"),
    )
    probe, trainer = TransitionTrainer.load_checkpoint(
        args.checkpoint, extractor, device=args.device
    )
    provenance = torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
        "provenance"
    ]
    print(
        f"arm {provenance['arm']} | step {trainer.step} "
        f"| parameters {provenance['parameter_count']:,} "
        f"| trained on manifest {str(provenance['manifest_hash'])[:12]}"
    )

    if args.manifest:
        manifest = LagPairManifest.load(args.manifest)
        dataset = LagPairDataset(pairs, manifest)
        print(f"evaluating manifest {manifest.content_hash()[:12]} ({len(manifest)} pairs)")
    else:
        train_ds, dataset, _, val_manifest = build_datasets(
            pairs, data, str(Path(args.checkpoint).parent)
        )
        train_ds.close()
        manifest = val_manifest

    loader = make_loader(
        dataset, data.get("batch_size", 4), shuffle=False,
        workers=data.get("num_workers", 0), seed=train_config.seed,
    )
    summary, records = trainer.evaluate(loader, split=args.split)

    print(f"\n{'lag':>6s} {'agg':>13s} {'CaRMSD':>8s} {'identity':>9s} {'rel':>6s} "
          f"{'rot deg':>8s} {'identity':>9s} {'contactF1':>10s} {'clash':>7s}")
    for row in summary.get("rows", []):
        for aggregation in ("micro", "domain_macro"):
            rmsd = row.get(f"ca_rmsd_{aggregation}", float("nan"))
            base = row.get(f"ca_rmsd_identity_{aggregation}", float("nan"))
            print(
                f"{row['lag_ns']:5.0f}n {aggregation:>13s} {rmsd:8.4f} {base:9.4f} "
                f"{(rmsd / base if base else float('nan')):6.3f} "
                f"{row.get(f'rotation_geodesic_deg_{aggregation}', float('nan')):8.2f} "
                f"{row.get(f'rotation_geodesic_deg_identity_{aggregation}', float('nan')):9.2f} "
                f"{row.get(f'contact_f1_{aggregation}', float('nan')):10.4f} "
                f"{row.get(f'clash_rate_{aggregation}', float('nan')):7.4f}"
            )
    print("\nrelative < 1.0 beats predicting that nothing moves.")

    out = args.out or str(Path(args.checkpoint).with_name("evaluation.json"))
    Path(out).write_text(
        json.dumps(
            {"provenance": provenance, "summary": summary, "records": records},
            indent=1, default=str,
        )
    )
    print(f"\nwritten -> {out}")
    dataset.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
