#!/usr/bin/env python
"""H0 -- score the frozen Phase 1.6 arms at heavy-atom resolution. No training.

    python scripts/evaluate_phase1_6_h0.py \
        --run runs/phase1_6_bounded_seed0 \
        --config configs/phase1_6_h0.yaml \
        --out runs/phase1_6_h0_seed0 \
        --device cuda:4

Every arm's ``last.pt`` is opened read-only, its sha256 taken before and after,
and the model run under ``torch.no_grad`` on the **same** validation manifest the
checkpoint trained against -- asserted against its own ``provenance.json``. This
is the pattern ``evaluate_phase1_6_extended.py`` established and it is reused
rather than reinvented.

What H0 adds is the placement step. The probe predicts a rigid update per residue
frame; H0 takes each residue's current heavy atoms, expressed in its current
frame, and puts them on the predicted frame. That is **transport, not
prediction** -- the side-chain conformation is a copy -- and every record says so
through ``construction_mode``.

Three scoring-only oracles are computed alongside, to decompose the error:
perfect frames with copied side chains (the internal-conformation floor),
predicted frames with true side chains (the frame-only component), and the
current structure unmoved (the baseline). Two of them read the future. None is
an arm, none reaches a conditioner, and each is labelled
``uses_future_for_scoring_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.heavy.backmapping import (  # noqa: E402
    CONSTRUCTION_MODES,
    heavy_atom_placements,
    target_heavy_atoms,
)
from force_md.heavy.chemistry import VDW_RADIUS_SOURCE  # noqa: E402
from force_md.heavy.domain_topology import load_domain_topology  # noqa: E402
from force_md.heavy.metrics import (  # noqa: E402
    AtomClassification,
    HeavyAtomMetricConfig,
    atom_contact_scores,
    bondi_overlap,
    heavy_atom_rmsd,
    sidechain_centroid_error,
)
from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import CANONICAL_ARMS  # noqa: E402
from force_md.transition.targets import (  # noqa: E402
    build_transition_target,
    identity_prediction,
)
from evaluate_phase1_6_extended import (  # noqa: E402
    environment,
    fix_numerics,
    git_state,
    sha256_file,
)
from train_transition import (  # noqa: E402
    build_configs,
    build_datasets,
    build_extractor,
    make_loader,
)

ROOT = Path(__file__).resolve().parents[1]
GIT = git_state()


def _graph_records(
    *,
    pair,
    placements,
    target_atoms,
    current_atoms,
    classification,
    excluded,
    is_1_4,
    resid_original,
    chain_index,
    config,
    context,
) -> list[dict]:
    """One record per (construction mode) for one graph."""
    records = []
    for mode, placement in placements.items():
        provenance = CONSTRUCTION_MODES[mode]
        row = {
            **context,
            "pair_id": pair.pair_id,
            "domain_id": pair.domain,
            "temperature": str(pair.temperature),
            "replica_id": str(pair.replica),
            "current_frame_index": int(pair.current_frame),
            "future_frame_index": int(pair.future_frame),
            "lag_ps": float(pair.lag_ps),
            "lag_ns": float(pair.lag_ps) / 1000.0,
            "construction_mode": mode,
            "prediction_source": mode,
            "is_model_predicted": provenance["is_model_predicted"],
            "uses_future_for_scoring_only": provenance["uses_future_for_scoring_only"],
            "sidechain_conformation_predicted": provenance[
                "sidechain_conformation_predicted"
            ],
            "is_construction_invariant": False,
            "atom_subset": "heavy",
            "pair_exclusion_rule": "psf_1_2_and_1_3; 1_4 kept in primary and tallied",
            "threshold_source": VDW_RADIUS_SOURCE,
        }
        row.update(heavy_atom_rmsd(placement.positions, target_atoms, classification))
        row.update(
            sidechain_centroid_error(placement.positions, target_atoms, classification)
        )
        row.update(
            bondi_overlap(
                placement.positions, classification, excluded, is_1_4, config
            )
        )
        row.update(
            atom_contact_scores(
                placement.positions, target_atoms, current_atoms, classification,
                resid_original, chain_index, config,
            )
        )
        records.append(row)
    return records


def evaluate_arm(
    *,
    canonical: str,
    arm_dir: Path,
    work_dir: Path,
    raw: dict,
    config_path: str,
    device: str,
    data_dir: str,
    metric_config: HeavyAtomMetricConfig,
    max_batches: int | None,
    log=print,
) -> tuple[list[dict], dict]:
    provenance = json.loads((arm_dir / "provenance.json").read_text())
    implementation = provenance["arm"]
    checkpoint = arm_dir / "last.pt"
    before = sha256_file(checkpoint)

    pairs, _, _, data = build_configs(yaml.safe_load(open(config_path)), implementation)
    seed = provenance["seed"]
    pairs = type(pairs)(**{**pairs.__dict__, "seed": seed})
    work_dir.mkdir(parents=True, exist_ok=True)
    _, val_ds, train_manifest, val_manifest = build_datasets(
        pairs, data, str(work_dir), log=lambda *a: None
    )
    if train_manifest.content_hash() != provenance["manifest_hash"]:
        raise SystemExit(
            f"{canonical}: rebuilt manifest {train_manifest.content_hash()[:12]} != "
            f"the checkpoint's {provenance['manifest_hash'][:12]}. Refusing."
        )
    loader = make_loader(
        val_ds, data.get("batch_size", 4), shuffle=False,
        workers=data.get("num_workers", 0), seed=seed,
    )
    extractor = build_extractor(raw, implementation, device)
    _, trainer = TransitionTrainer.load_checkpoint(
        str(checkpoint), extractor, device=device
    )
    model = trainer.module
    model.eval()

    spec = CANONICAL_ARMS.get(canonical)
    context = {
        "arm": implementation,
        "canonical_arm": canonical,
        "oracle": bool(spec.oracle) if spec else bool(provenance.get("oracle")),
        "seed": seed,
        "step": int(trainer.step),
        "split": "val",
        "manifest_hash": provenance["manifest_hash"],
        "phase1_checkpoint_hash": provenance.get("phase1_sha256", ""),
        "transition_checkpoint_hash": before,
        "config_hash": provenance.get("config_hash", ""),
        "git_commit": GIT["commit"],
        "git_dirty": GIT["dirty"],
        "source_diff_hash": GIT["source_diff_hash"],
    }

    records: list[dict] = []
    started = time.time()
    verified: set[str] = set()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = batch.to(device)
            target = build_transition_target(batch.current, batch.future)
            bundle = (
                trainer.extractor.oracle_bundle(batch.current)
                if model.conditioner.requires_oracle
                else trainer.extractor(batch.current)
            )
            prediction = model(
                batch.current, bundle, history=batch.history, lag_ps=batch.lag_ps
            )
            placements = heavy_atom_placements(
                prediction, target, batch.current, batch.future
            )
            future_atoms = target_heavy_atoms(target, batch.future, batch.current)

            atom_batch = batch.current.atoms.batch_index
            residue_batch = batch.current.residues.batch_index
            for graph, pair in enumerate(batch.pairs):
                atom_rows = (atom_batch == graph).nonzero(as_tuple=True)[0]
                residue_rows = (residue_batch == graph).nonzero(as_tuple=True)[0]
                topology = load_domain_topology(data_dir, pair.domain)
                if pair.domain not in verified:
                    topology.verify_against(
                        batch.current.atoms.atomic_number[atom_rows],
                        batch.current.atoms.atom_name_id[atom_rows],
                    )
                    verified.add(pair.domain)

                local_residue = torch.full_like(residue_batch, -1)
                local_residue[residue_rows] = torch.arange(
                    residue_rows.numel(), device=residue_rows.device
                )
                classification = AtomClassification(
                    is_backbone=topology.is_backbone.to(device),
                    is_sidechain=topology.is_sidechain.to(device),
                    residue_index=local_residue[
                        batch.current.atoms.atom_to_residue[atom_rows]
                    ],
                    vdw_radius=topology.vdw_radius.to(device),
                    valid=placements["transported_current_conformer"].valid[atom_rows],
                )
                excluded, is_1_4 = topology.exclusion_masks(device=device)
                records.extend(
                    _graph_records(
                        pair=pair,
                        placements={
                            m: type(p)(p.positions[atom_rows], m, p.valid[atom_rows])
                            for m, p in placements.items()
                        },
                        target_atoms=future_atoms[atom_rows],
                        current_atoms=batch.current.atoms.positions[atom_rows],
                        classification=classification,
                        excluded=excluded,
                        is_1_4=is_1_4,
                        resid_original=batch.current.residues.resid_original[residue_rows],
                        chain_index=batch.current.residues.chain_index[residue_rows],
                        config=metric_config,
                        context=context,
                    )
                )
    val_ds.close()

    after = sha256_file(checkpoint)
    if before != after:
        raise SystemExit(
            f"{canonical}: checkpoint changed during evaluation "
            f"({before[:12]} -> {after[:12]}). Stopping."
        )
    modes = sorted({r["construction_mode"] for r in records})
    log(
        f"  {len(records)} records over {len(modes)} construction modes, "
        f"{len({r['pair_id'] for r in records})} pairs, "
        f"{len({r['domain_id'] for r in records})} domains, "
        f"{time.time() - started:.0f}s | checkpoint {before[:12]} unchanged"
    )
    return records, {
        "canonical_arm": canonical,
        "implementation": implementation,
        "checkpoint_sha256": before,
        "step": int(trainer.step),
        "manifest_hash": provenance["manifest_hash"],
        "val_manifest_hash": val_manifest.content_hash(),
        "construction_modes": modes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="smoke only; the records are not a result",
    )
    args = parser.parse_args()
    numerics = fix_numerics()
    if args.max_batches is not None:
        print(
            f"SMOKE: only {args.max_batches} batches per arm. Not a result.",
            flush=True,
        )

    raw = yaml.safe_load(open(args.config))
    run = Path(args.run)
    arms = args.arms or [
        d for d in sorted(os.listdir(run))
        if d in CANONICAL_ARMS and (run / d / "last.pt").exists()
    ]
    if not arms:
        raise SystemExit(f"no arm checkpoints under {run}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metric_config = HeavyAtomMetricConfig(
        **(raw.get("h0", {}).get("metrics", {}))
    )
    data_dir = raw["data"]["data_dir"]

    records_path = out / "records.jsonl"
    temporary = records_path.with_suffix(".jsonl.partial")
    arm_provenance, counts = [], {}
    with open(temporary, "w") as handle:
        for canonical in arms:
            print(f"\n{'=' * 70}\n{canonical}\n{'=' * 70}", flush=True)
            records, provenance = evaluate_arm(
                canonical=canonical,
                arm_dir=run / canonical,
                work_dir=out / "manifests" / canonical,
                raw=raw,
                config_path=args.config,
                device=args.device,
                data_dir=data_dir,
                metric_config=metric_config,
                max_batches=args.max_batches,
            )
            for record in records:
                handle.write(json.dumps(record) + "\n")
            counts[canonical] = len(records)
            arm_provenance.append(provenance)
    sizes = set(counts.values())
    if len(sizes) > 1:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"arms produced different record counts: {counts}")
    temporary.replace(records_path)

    (out / "reproducibility_manifest.json").write_text(json.dumps({
        "stage": "H0 heavy-atom backmapping of frozen checkpoints",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": str(run),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "record_counts": counts,
        "arms": arm_provenance,
        "construction_modes": CONSTRUCTION_MODES,
        "metric_config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in metric_config.__dict__.items()
        },
        "vdw_source": VDW_RADIUS_SOURCE,
        "git": GIT,
        "environment": environment(args.device, numerics),
        "retrained": False,
        "checkpoints_verified_unchanged": True,
    }, indent=1, default=str))
    print(
        f"\nwrote {sum(counts.values())} records to {records_path}\n"
        "no weights were changed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
