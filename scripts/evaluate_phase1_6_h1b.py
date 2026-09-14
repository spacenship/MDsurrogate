#!/usr/bin/env python
"""H1b -- score the refined frames against the coarse arm and the identity baseline.

    python scripts/evaluate_phase1_6_h1b.py \
        --refiner runs/phase1_6_h1b_full_seed0/refiner_best.pt \
        --config configs/phase1_6_h1b_full.yaml \
        --run runs/phase1_6_bounded_seed0 \
        --out runs/phase1_6_h1b_eval_seed0 --device cuda:4

H1b exists because of one Stage M result: every trained arm was **worse than
"nothing moves"** on all ten physical-validity cells -- peptide C-N bond 4-5x,
backbone angle 3.5-5x, consecutive Ca 5-7x, Ca clash rate 361-1651x, backbone
torsion 1.1-1.2x. So the question this script answers is not "did the loss go
down" but "did those ten cells move, and what did it cost".

Three predictions are scored on the identical validation pairs:

===========  ==========================================================
``coarse``   the frozen arm alone -- must reproduce Stage M exactly
``refined``  the same arm with the H1b correction composed on
``identity`` nothing moves; the bar Stage M said none of the arms cleared
===========  ==========================================================

They are computed by ``extended_metric_records`` -- the **same function** that
produced the Stage M table, not a reimplementation -- so an H1b number and a
Stage M number can sit in one sentence. The ``coarse`` rows are the check on
that: they are the frozen arm on the frozen manifest, so they must reproduce the
Stage M values to printed precision, and if they do not, nothing else here is
trustworthy.

Neither the coarse checkpoint nor the refiner checkpoint is written. Both are
opened read-only and hashed before and after.

Scope: this reports the frame-level Stage M schema and the full H0 PSF-based
heavy-atom schema on the same held-out pairs. Future heavy coordinates are
scoring targets only; they are never fed to the refiner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.heavy.refiner import (  # noqa: E402
    BackboneFrameConstraintRefiner,
    RefinerConfig,
    compose,
    correction_magnitude,
    prediction_from_frames,
    refiner_feature_dim,
    refiner_features,
)
from force_md.heavy.backmapping import (  # noqa: E402
    CONSTRUCTION_MODES,
    HeavyAtomPlacement,
    heavy_atom_placements,
    identity_current_atoms,
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
from force_md.transition.extended_metrics import (  # noqa: E402
    ExtendedMetricConfig,
    RecordContext,
    extended_metric_records,
)
from force_md.transition.targets import (  # noqa: E402
    apply_prediction,
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

#: The ten cells Stage M reported the arms losing on, plus the primary metrics
#: they must not be bought with. Named here so the report cannot quietly drop
#: one that moved the wrong way.
VALIDITY = (
    "bond_length_rmse",
    "bond_angle_mae_deg",
    "ca_neighbor_distance_mae",
    "clash_rate",
    "backbone_torsion_mae_deg",
)
PRIMARY = (
    "ca_rmsd",
    "drmsd_long_range",
    "rotation_geodesic_mean_deg",
    "contact_f1",
)


def log(message: str) -> None:
    print(message, flush=True)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    """Fixed cap-report quantiles over valid residue corrections."""
    values = values.detach().to(torch.float64)
    if values.numel() == 0:
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _heavy_record(
    *,
    pair,
    stage: str,
    placement: HeavyAtomPlacement,
    target_atoms: torch.Tensor,
    current_atoms: torch.Tensor,
    topology,
    resid_original: torch.Tensor,
    chain_index: torch.Tensor,
    atom_rows: torch.Tensor,
    residue_index: torch.Tensor,
    excluded: torch.Tensor,
    is_1_4: torch.Tensor,
    metric_config: HeavyAtomMetricConfig,
    context: dict,
) -> dict:
    provenance = CONSTRUCTION_MODES[placement.construction_mode]
    classification = AtomClassification(
        is_backbone=topology.is_backbone.to(placement.positions.device),
        is_sidechain=topology.is_sidechain.to(placement.positions.device),
        residue_index=residue_index,
        vdw_radius=topology.vdw_radius.to(placement.positions.device),
        valid=placement.valid[atom_rows],
    )
    if bool((classification.residue_index < 0).any()):
        raise ValueError("heavy-atom mapping did not produce local residue indices")
    row = {
        **context,
        "h1b_stage": stage,
        "pair_id": pair.pair_id,
        "domain_id": pair.domain,
        "temperature": str(pair.temperature),
        "replica_id": str(pair.replica),
        "current_frame_index": int(pair.current_frame),
        "future_frame_index": int(pair.future_frame),
        "lag_ps": float(pair.lag_ps),
        "lag_ns": float(pair.lag_ps) / 1000.0,
        "construction_mode": placement.construction_mode,
        "prediction_source": stage,
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
    row.update(heavy_atom_rmsd(placement.positions[atom_rows], target_atoms, classification))
    row.update(sidechain_centroid_error(placement.positions[atom_rows], target_atoms, classification))
    row.update(bondi_overlap(
        placement.positions[atom_rows], classification, excluded, is_1_4, metric_config
    ))
    row.update(atom_contact_scores(
        placement.positions[atom_rows], target_atoms, current_atoms,
        classification, resid_original, chain_index, metric_config
    ))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument(
        "--h0-config", default=str(ROOT / "configs" / "phase1_6_h0.yaml"),
        help="frozen H0 PSF metric settings; kept separate from the H1b train config",
    )
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke only; the records are not a result")
    args = parser.parse_args()

    numerics = fix_numerics()
    raw = yaml.safe_load(open(args.config))
    h0_raw = yaml.safe_load(open(args.h0_config))
    refiner_state = torch.load(
        args.refiner, map_location=args.device, weights_only=False
    )
    canonical = refiner_state["canonical_arm"]

    run = Path(args.run)
    arm_dir = run / canonical
    checkpoint = arm_dir / "last.pt"
    coarse_before = sha256_file(checkpoint)
    refiner_before = sha256_file(args.refiner)
    if refiner_state["coarse_checkpoint_sha256"] != coarse_before:
        raise SystemExit(
            f"the refiner was trained against coarse checkpoint "
            f"{refiner_state['coarse_checkpoint_sha256'][:12]} but "
            f"{checkpoint} now hashes {coarse_before[:12]}. Refusing: the "
            "correction would be composed onto frames it never saw."
        )

    provenance = json.loads((arm_dir / "provenance.json").read_text())
    implementation = provenance["arm"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs_config, _, _, data = build_configs(raw, implementation)
    seed = provenance["seed"]
    pairs_config = type(pairs_config)(**{**pairs_config.__dict__, "seed": seed})
    _, val_ds, train_manifest, val_manifest = build_datasets(
        pairs_config, data, str(out / "manifests"), log=lambda *a: None
    )
    if train_manifest.content_hash() != provenance["manifest_hash"]:
        raise SystemExit(
            f"rebuilt manifest {train_manifest.content_hash()[:12]} != the "
            f"checkpoint's {provenance['manifest_hash'][:12]}. Refusing."
        )
    loader = make_loader(
        val_ds, data.get("batch_size", 4), shuffle=False,
        workers=data.get("num_workers", 0), seed=seed,
    )

    extractor = build_extractor(raw, implementation, args.device)
    _, trainer = TransitionTrainer.load_checkpoint(
        str(checkpoint), extractor, device=args.device
    )
    coarse = trainer.module
    coarse.eval()
    for parameter in coarse.parameters():
        parameter.requires_grad_(False)

    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(), RefinerConfig(**refiner_state["refiner_config"])
    ).to(args.device)
    refiner.load_state_dict(refiner_state["refiner"])
    refiner.eval()

    # The Stage M defaults, NOT the config's `train.metrics` block. The two
    # disagree on `contact_sequence_separation` -- 6 here, 3 there -- because the
    # train block configures the Stage B `MetricConfig`, a different contact
    # definition. Feeding it in changed every contact-derived metric while
    # leaving the geometry ones untouched, which is exactly what a silently
    # incomparable number looks like; the coarse-vs-Stage-M reproduction check
    # caught it. Whatever Stage M used, this must use.
    metric_config = ExtendedMetricConfig()
    heavy_metric_config = HeavyAtomMetricConfig(
        **h0_raw.get("h0", {}).get("metrics", {})
    )
    spec = CANONICAL_ARMS.get(canonical)

    def context_for(stage: str, step: int) -> RecordContext:
        return RecordContext(
            arm=implementation,
            canonical_arm=canonical,
            oracle=bool(spec.oracle) if spec else bool(provenance.get("oracle")),
            seed=seed,
            manifest_hash=provenance["manifest_hash"],
            phase1_checkpoint_hash=provenance.get("phase1_sha256", ""),
            transition_checkpoint_hash=coarse_before if stage != "identity" else "",
            config_hash=provenance.get("config_hash", ""),
            git_commit=GIT["commit"],
            git_dirty=GIT["dirty"],
            source_diff_hash=GIT["source_diff_hash"],
            extra={"step": step, "split": "val", "h1b_stage": stage},
        )

    contexts = {
        "coarse": context_for("coarse", int(trainer.step)),
        "refined": context_for("refined", int(refiner_state["step"])),
        "identity": context_for("identity", 0),
    }
    records: dict[str, list[dict]] = {key: [] for key in contexts}
    heavy_records: dict[str, list[dict]] = {key: [] for key in contexts}
    magnitudes: list[dict] = []
    cap_norm_path = out / "cap_norms.jsonl"
    cap_norm_handle = cap_norm_path.open("w")

    started = time.time()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            batch = batch.to(args.device)
            target = build_transition_target(batch.current, batch.future)
            bundle = (
                trainer.extractor.oracle_bundle(batch.current)
                if coarse.conditioner.requires_oracle
                else trainer.extractor(batch.current)
            )
            coarse_prediction = coarse(
                batch.current, bundle, history=batch.history, lag_ps=batch.lag_ps
            )
            correction = refiner(
                refiner_features(coarse_prediction, target, batch.lag_ps / 1000.0)
            )
            origin, rotation = compose(
                *apply_prediction(coarse_prediction, target), correction
            )
            refined_prediction = prediction_from_frames(origin, rotation, target)

            # Heavy-atom scoring uses exactly H0's PSF masks, atom mapping,
            # Bondi radii and the target's one proper-Kabsch alignment. Future
            # heavy coordinates enter only as the scoring target.
            coarse_placements = heavy_atom_placements(
                coarse_prediction, target, batch.current, batch.future
            )
            refined_placements = heavy_atom_placements(
                refined_prediction, target, batch.current, batch.future
            )
            identity_placement = identity_current_atoms(target, batch.current)
            future_atoms = target_heavy_atoms(target, batch.future, batch.current)
            atom_batch = batch.current.atoms.batch_index
            residue_batch = batch.current.residues.batch_index
            for graph, pair in enumerate(batch.pairs):
                atom_rows = (atom_batch == graph).nonzero(as_tuple=True)[0]
                residue_rows = (residue_batch == graph).nonzero(as_tuple=True)[0]
                topology = load_domain_topology(raw["data"]["data_dir"], pair.domain)
                topology.verify_against(
                    batch.current.atoms.atomic_number[atom_rows],
                    batch.current.atoms.atom_name_id[atom_rows],
                )
                local_residue = torch.full_like(residue_batch, -1)
                local_residue[residue_rows] = torch.arange(
                    residue_rows.numel(), device=residue_rows.device
                )
                residue_index = local_residue[
                    batch.current.atoms.atom_to_residue[atom_rows]
                ]
                excluded, is_1_4 = topology.exclusion_masks(device=args.device)
                context = {
                    "arm": implementation,
                    "canonical_arm": canonical,
                    "oracle": bool(spec.oracle) if spec else bool(provenance.get("oracle")),
                    "seed": seed,
                    "manifest_hash": provenance["manifest_hash"],
                    "phase1_checkpoint_hash": provenance.get("phase1_sha256", ""),
                    "transition_checkpoint_hash": coarse_before,
                    "config_hash": provenance.get("config_hash", ""),
                    "git_commit": GIT["commit"],
                    "git_dirty": GIT["dirty"],
                    "source_diff_hash": GIT["source_diff_hash"],
                    "step": int(refiner_state["step"]),
                    "split": "val",
                    "heavy_metric_config_source": str(Path(args.h0_config).resolve()),
                }
                for stage, placement in (
                    ("coarse", coarse_placements["transported_current_conformer"]),
                    ("refined", refined_placements["transported_current_conformer"]),
                    ("identity", identity_placement),
                ):
                    heavy_records[stage].append(_heavy_record(
                        pair=pair,
                        stage=stage,
                        placement=placement,
                        target_atoms=future_atoms[atom_rows],
                        current_atoms=batch.current.atoms.positions[atom_rows],
                        topology=topology,
                        resid_original=batch.current.residues.resid_original[residue_rows],
                        chain_index=batch.current.residues.chain_index[residue_rows],
                        atom_rows=atom_rows,
                        residue_index=residue_index,
                        excluded=excluded,
                        is_1_4=is_1_4,
                        metric_config=heavy_metric_config,
                        context=context,
                    ))

            for stage, prediction in (
                ("coarse", coarse_prediction),
                ("refined", refined_prediction),
                ("identity", identity_prediction(target)),
            ):
                records[stage].extend(
                    extended_metric_records(
                        prediction, target, pairs=batch.pairs,
                        context=contexts[stage], config=metric_config,
                        identity=(stage == "identity"),
                    )
                )

            # Per-graph correction size, so "the refiner did nothing" and "the
            # refiner reverted the transition" are distinguishable in the output
            # rather than inferred from the metrics.
            magnitude = correction_magnitude(correction)
            for graph, pair in enumerate(batch.pairs):
                rows = (target.residue_batch_index == graph) & target.valid
                if not bool(rows.any()):
                    continue
                translation_raw = _quantiles(magnitude["raw_translation_norm"][rows])
                translation_clipped = _quantiles(
                    magnitude["correction_translation_norm"][rows]
                )
                rotation_raw = _quantiles(magnitude["raw_rotation_deg"][rows])
                rotation_clipped = _quantiles(
                    magnitude["correction_rotation_deg"][rows]
                )
                translation_cap = refiner.config.max_translation
                rotation_cap = math.degrees(refiner.config.max_rotation_rad)
                translation_saturated = (
                    magnitude["raw_translation_norm"][rows]
                    >= translation_cap * (1.0 - 1e-6)
                )
                rotation_saturated = (
                    magnitude["raw_rotation_deg"][rows]
                    >= rotation_cap * (1.0 - 1e-6)
                )
                valid_indices = rows.nonzero(as_tuple=True)[0].tolist()
                raw_translation_values = magnitude["raw_translation_norm"][rows].tolist()
                clipped_translation_values = magnitude[
                    "correction_translation_norm"
                ][rows].tolist()
                raw_rotation_values = magnitude["raw_rotation_deg"][rows].tolist()
                clipped_rotation_values = magnitude["correction_rotation_deg"][rows].tolist()
                for residue_index, raw_t, clipped_t, raw_r, clipped_r in zip(
                    valid_indices,
                    raw_translation_values,
                    clipped_translation_values,
                    raw_rotation_values,
                    clipped_rotation_values,
                ):
                    cap_norm_handle.write(json.dumps({
                        "pair_id": pair.pair_id,
                        "domain_id": pair.domain,
                        "lag_ps": float(pair.lag_ps),
                        "lag_ns": float(pair.lag_ps) / 1000.0,
                        "residue_index": int(residue_index),
                        "raw_translation_norm_a": float(raw_t),
                        "clipped_translation_norm_a": float(clipped_t),
                        "raw_rotation_norm_deg": float(raw_r),
                        "clipped_rotation_norm_deg": float(clipped_r),
                        "translation_saturated": bool(raw_t >= translation_cap * (1.0 - 1e-6)),
                        "rotation_saturated": bool(raw_r >= rotation_cap * (1.0 - 1e-6)),
                    }) + "\n")
                magnitudes.append({
                    "pair_id": pair.pair_id,
                    "domain_id": pair.domain,
                    "lag_ps": float(pair.lag_ps),
                    "lag_ns": float(pair.lag_ps) / 1000.0,
                    "n_valid_residues": int(rows.sum()),
                    "raw_translation_norm_mean_a": translation_raw["mean"],
                    "raw_translation_norm_median_a": translation_raw["median"],
                    "raw_translation_norm_p90_a": translation_raw["p90"],
                    "raw_translation_norm_p95_a": translation_raw["p95"],
                    "raw_translation_norm_p99_a": translation_raw["p99"],
                    "raw_translation_norm_max_a": translation_raw["max"],
                    "clipped_translation_norm_mean_a": translation_clipped["mean"],
                    "clipped_translation_norm_max_a": translation_clipped["max"],
                    "raw_rotation_norm_mean_deg": rotation_raw["mean"],
                    "raw_rotation_norm_median_deg": rotation_raw["median"],
                    "raw_rotation_norm_p90_deg": rotation_raw["p90"],
                    "raw_rotation_norm_p95_deg": rotation_raw["p95"],
                    "raw_rotation_norm_p99_deg": rotation_raw["p99"],
                    "raw_rotation_norm_max_deg": rotation_raw["max"],
                    "clipped_rotation_norm_mean_deg": rotation_clipped["mean"],
                    "clipped_rotation_norm_max_deg": rotation_clipped["max"],
                    "correction_translation_mean_a": float(
                        magnitude["correction_translation_norm"][rows].mean()
                    ),
                    "correction_translation_max_a": float(
                        magnitude["correction_translation_norm"][rows].max()
                    ),
                    "correction_rotation_mean_deg": float(
                        magnitude["correction_rotation_deg"][rows].mean()
                    ),
                    "correction_rotation_max_deg": float(
                        magnitude["correction_rotation_deg"][rows].max()
                    ),
                    "translation_at_cap_fraction": float(
                        translation_saturated.to(torch.float64).mean()
                    ),
                    "rotation_at_cap_fraction": float(
                        rotation_saturated.to(torch.float64).mean()
                    ),
                })
    cap_norm_handle.close()
    val_ds.close()

    if sha256_file(checkpoint) != coarse_before:
        raise SystemExit("the coarse checkpoint changed during evaluation")
    if sha256_file(args.refiner) != refiner_before:
        raise SystemExit("the refiner checkpoint changed during evaluation")

    counts = {stage: len(rows) for stage, rows in records.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"stages produced different record counts: {counts}")

    with open(out / "records.jsonl", "w") as handle:
        for stage, rows in records.items():
            for row in rows:
                handle.write(json.dumps({**row, "h1b_stage": stage}) + "\n")
    with open(out / "heavy_records.jsonl", "w") as handle:
        for stage, rows in heavy_records.items():
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (out / "correction_magnitudes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in magnitudes) + "\n"
    )
    (out / "reproducibility_manifest.json").write_text(json.dumps({
        "stage": "H1b evaluation: coarse vs refined vs identity",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "is_smoke": args.max_batches is not None,
        "canonical_arm": canonical,
        "implementation": implementation,
        "coarse_checkpoint_sha256": coarse_before,
        "coarse_weights_unchanged": True,
        "refiner_checkpoint": str(Path(args.refiner).resolve()),
        "refiner_checkpoint_sha256": refiner_before,
        "refiner_step": int(refiner_state["step"]),
        "refiner_config": refiner_state["refiner_config"],
        "manifest_hash": provenance["manifest_hash"],
        "val_manifest_hash": val_manifest.content_hash(),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "record_counts": counts,
        "heavy_record_counts": {
            stage: len(rows) for stage, rows in heavy_records.items()
        },
        "cap_norm_records": str(cap_norm_path.resolve()),
        "h0_config": str(Path(args.h0_config).resolve()),
        "h0_config_sha256": sha256_file(args.h0_config),
        "heavy_metric_config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in heavy_metric_config.__dict__.items()
        },
        "cap_policy": {
            "translation_norm_max_a": float(refiner.config.max_translation),
            "rotation_geodesic_residual_max_deg": math.degrees(
                refiner.config.max_rotation_rad
            ),
            "applies_to": "vector magnitude; radial projection, not per-component clipping",
            "saturation_definition": "raw vector norm >= cap*(1-1e-6)",
        },
        "validity_metrics": list(VALIDITY),
        "primary_metrics": list(PRIMARY),
        "wall_seconds": round(time.time() - started, 1),
        "git": GIT,
        "numerics": numerics,
        "environment": environment(args.device, numerics),
    }, indent=2, default=str))

    log(
        f"{sum(counts.values())} records "
        f"({counts['coarse']} per stage, "
        f"{len({r['pair_id'] for r in records['coarse']})} pairs, "
        f"{len({r['domain_id'] for r in records['coarse']})} domains) in "
        f"{time.time() - started:.0f}s"
    )
    log("neither checkpoint was modified")
    log(f"wrote {out / 'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
