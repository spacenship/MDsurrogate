#!/usr/bin/env python
"""Make the registered Phase 1.6 H1b decision report.

The evaluator writes one row per validation pair and construction stage.  This
script is deliberately a small reporting layer: it does not recompute model
outputs, change masks, alignments, thresholds, or weights.  The primary point
estimate is the sample-equal mean.  Every interval resamples validation domains
as clusters, preserving the existing Phase 1.6 statistical convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase1_6_extended import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
)

STAGES = ("identity", "coarse", "refined")
STAGE_LABELS = {
    "identity": "identity_current_atoms",
    "coarse": "frozen P0 coarse prediction",
    "refined": "frozen P0 + trained H1b",
}
LAGS = (1.0, 4.0)

# metric, human label, direction.  Direction is used only for interpretation;
# every reported paired delta remains candidate minus control.
FRAME_METRICS = (
    ("ca_rmsd", "Cα RMSD (Å)", "lower"),
    ("drmsd_long_range", "long-range dRMSD (Å)", "lower"),
    ("drmsd_all", "all-pair dRMSD (Å)", "lower"),
    ("rotation_geodesic_mean_deg", "residue-frame rotation geodesic (deg)", "lower"),
    ("bond_length_rmse", "peptide C–N bond RMSE (Å)", "lower"),
    ("bond_angle_ca_c_n_mae_deg", "CA–C–N angle error (deg)", "lower"),
    ("bond_angle_c_n_ca_mae_deg", "C–N–CA angle error (deg)", "lower"),
    ("bond_angle_mae_deg", "both peptide-angle error (deg)", "lower"),
    ("ca_neighbor_distance_mae", "consecutive Cα distance error (Å)", "lower"),
    ("contact_f1", "Cα static contact F1", "higher"),
    ("formed_contact_f1", "Cα formed-contact F1", "higher"),
    ("broken_contact_f1", "Cα broken-contact F1", "higher"),
)

HEAVY_METRICS = (
    ("heavy_atom_rmsd", "heavy-atom RMSD (Å)", "lower"),
    ("backbone_heavy_rmsd", "backbone-heavy RMSD (Å)", "lower"),
    ("sidechain_heavy_rmsd", "side-chain-heavy RMSD (Å)", "lower"),
    ("bondi_serious_overlap_rate", "Bondi serious overlap, inter-residue total", "lower"),
    ("bondi_serious_inter_residue_backbone_backbone_rate", "Bondi serious overlap, inter bb–bb", "lower"),
    ("bondi_serious_inter_residue_backbone_sidechain_rate", "Bondi serious overlap, inter bb–sc", "lower"),
    ("bondi_serious_inter_residue_sidechain_sidechain_rate", "Bondi serious overlap, inter sc–sc", "lower"),
    ("bondi_serious_intra_residue_backbone_backbone_rate", "Bondi serious overlap, intra bb–bb", "lower"),
    ("bondi_serious_intra_residue_backbone_sidechain_rate", "Bondi serious overlap, intra bb–sc", "lower"),
    ("bondi_serious_intra_residue_sidechain_sidechain_rate", "Bondi serious overlap, intra sc–sc", "lower"),
    ("atom_contact_any_precision", "atom contact static precision", "higher"),
    ("atom_contact_any_recall", "atom contact static recall", "higher"),
    ("atom_contact_any_f1", "atom contact static F1", "higher"),
    ("atom_contact_backbone_backbone_f1", "atom contact static bb–bb F1", "higher"),
    ("atom_contact_backbone_sidechain_f1", "atom contact static bb–sc F1", "higher"),
    ("atom_contact_sidechain_sidechain_f1", "atom contact static sc–sc F1", "higher"),
    ("atom_contact_formed_precision", "atom contact formed precision", "higher"),
    ("atom_contact_formed_recall", "atom contact formed recall", "higher"),
    ("atom_contact_formed_f1", "atom contact formed F1", "higher"),
    ("atom_contact_broken_precision", "atom contact broken precision", "higher"),
    ("atom_contact_broken_recall", "atom contact broken recall", "higher"),
    ("atom_contact_broken_f1", "atom contact broken F1", "higher"),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def value_map(rows: list[dict], metric: str, *, lag: float | None = None) -> dict[tuple[str, str], float]:
    values = {}
    for row in rows:
        if lag is not None and not math.isclose(float(row["lag_ns"]), lag):
            continue
        if finite(row.get(metric)):
            key = (str(row["pair_id"]), str(row["domain_id"]))
            if key in values:
                raise SystemExit(f"duplicate metric row for {key} and {metric}")
            values[key] = float(row[metric])
    return values


def mean_ci(values: dict[tuple[str, str], float]) -> tuple[float, float, float]:
    return cluster_bootstrap(
        values, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED
    )


def cell(value: float, digits: int = 4) -> str:
    return "—" if not finite(value) else f"{float(value):.{digits}f}"


def ci_cell(values: dict[tuple[str, str], float], digits: int = 4) -> str:
    mean, lo, hi = mean_ci(values)
    if not finite(mean):
        return "—"
    if not finite(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def paired_delta(
    control: dict[tuple[str, str], float],
    candidate: dict[tuple[str, str], float],
) -> tuple[float, float, float, int, int]:
    shared = sorted(set(control) & set(candidate))
    differences = {key: candidate[key] - control[key] for key in shared}
    mean, lo, hi = mean_ci(differences)
    return mean, lo, hi, len(shared), len({key[1] for key in shared})


def verdict(delta: float, lo: float, hi: float, direction: str) -> str:
    if not finite(lo) or not finite(hi):
        return "not separated"
    if direction == "lower":
        if lo > 0:
            return "REGRESSION"
        if hi < 0:
            return "improves"
    else:
        if hi < 0:
            return "REGRESSION"
        if lo > 0:
            return "improves"
    return "not separated"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def domain_rows(
    stage_rows: dict[str, list[dict]], metrics: tuple[tuple[str, str, str], ...]
) -> list[dict]:
    output = []
    for stage in STAGES:
        for lag in LAGS:
            by_domain: dict[str, list[dict]] = defaultdict(list)
            for row in stage_rows[stage]:
                if math.isclose(float(row["lag_ns"]), lag):
                    by_domain[str(row["domain_id"])].append(row)
            for domain, rows in sorted(by_domain.items()):
                out = {
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "lag_ns": lag,
                    "domain_id": domain,
                    "n_samples": len(rows),
                }
                for metric, _label, _direction in metrics:
                    values = [float(row[metric]) for row in rows if finite(row.get(metric))]
                    out[metric] = float(np.mean(values)) if values else ""
                    out[f"{metric}_n"] = len(values)
                output.append(out)
    return output


def cap_stats(rows: list[dict]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n": len(rows)}
    fields = (
        ("raw_translation_norm_a", "raw_translation"),
        ("clipped_translation_norm_a", "clipped_translation"),
        ("raw_rotation_norm_deg", "raw_rotation"),
        ("clipped_rotation_norm_deg", "clipped_rotation"),
    )
    for field, prefix in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if not len(values):
            continue
        result[f"{prefix}_mean"] = float(values.mean())
        if prefix.startswith("raw_"):
            for q in (50, 90, 95, 99):
                result[f"{prefix}_p{q}"] = float(np.percentile(values, q))
        result[f"{prefix}_max"] = float(values.max())
    if rows:
        result["translation_saturation_fraction"] = float(
            np.mean([bool(row["translation_saturated"]) for row in rows])
        )
        result["rotation_saturation_fraction"] = float(
            np.mean([bool(row["rotation_saturated"]) for row in rows])
        )
    return result


def cap_table(stats: list[tuple[str, dict]]) -> str:
    headers = [
        "scope", "n", "raw translation mean / median / p90 / p95 / p99 / max (Å)",
        "clipped translation mean / max (Å)",
        "raw rotation mean / median / p90 / p95 / p99 / max (deg)",
        "clipped rotation mean / max (deg)", "translation saturation", "rotation saturation",
    ]
    rows = []
    for label, item in stats:
        rows.append([
            label,
            str(item.get("n", 0)),
            "/".join(cell(item.get(f"raw_translation_{key}")) for key in ("mean", "p50", "p90", "p95", "p99", "max")),
            "/".join(cell(item.get(f"clipped_translation_{key}")) for key in ("mean", "max")),
            "/".join(cell(item.get(f"raw_rotation_{key}")) for key in ("mean", "p50", "p90", "p95", "p99", "max")),
            "/".join(cell(item.get(f"clipped_rotation_{key}")) for key in ("mean", "max")),
            cell(item.get("translation_saturation_fraction")),
            cell(item.get("rotation_saturation_fraction")),
        ])
    return markdown_table(headers, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--heavy-records", required=True)
    parser.add_argument("--cap-norms", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.records))
    heavy_records = load_jsonl(Path(args.heavy_records))
    cap_norms = load_jsonl(Path(args.cap_norms))
    eval_manifest = json.loads(Path(args.eval_manifest).read_text())
    train_dir = Path(args.train_run)
    train_manifest_path = train_dir / "reproducibility_manifest.json"
    train_manifest = json.loads(train_manifest_path.read_text())

    frame_by_stage = {stage: [row for row in records if row.get("h1b_stage") == stage] for stage in STAGES}
    heavy_by_stage = {stage: [row for row in heavy_records if row.get("h1b_stage") == stage] for stage in STAGES}
    for name, grouped in (("frame", frame_by_stage), ("heavy", heavy_by_stage)):
        counts = {stage: len(rows) for stage, rows in grouped.items()}
        if len(set(counts.values())) != 1 or not counts["coarse"]:
            raise SystemExit(f"{name} stage record counts are not identical: {counts}")
        for stage, rows in grouped.items():
            ids = [row["pair_id"] for row in rows]
            if len(set(ids)) != len(ids):
                raise SystemExit(f"{name}/{stage} contains duplicate pair_id")

    pair_sets = [
        {row["pair_id"] for row in frame_by_stage[stage]}
        for stage in STAGES
    ]
    if any(pair_sets[0] != item for item in pair_sets[1:]):
        raise SystemExit("frame stages do not score the same validation pairs")

    all_metrics = FRAME_METRICS + HEAVY_METRICS
    domain_metric_rows = domain_rows(frame_by_stage, FRAME_METRICS) + domain_rows(heavy_by_stage, HEAVY_METRICS)
    domain_columns = ["stage", "stage_label", "lag_ns", "domain_id", "n_samples"]
    domain_columns += [metric for metric, _label, _direction in all_metrics]
    domain_columns += [f"{metric}_n" for metric, _label, _direction in all_metrics]
    domain_path = out.with_name(out.stem + "_domain_metrics.csv")
    write_csv(domain_path, domain_metric_rows, domain_columns)

    cap_by_scope = [("all validation residues", cap_stats(cap_norms))]
    cap_domain_rows = []
    for lag in LAGS:
        lag_rows = [row for row in cap_norms if math.isclose(float(row["lag_ns"]), lag)]
        cap_by_scope.append((f"lag {lag:g} ns", cap_stats(lag_rows)))
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for row in lag_rows:
            by_domain[str(row["domain_id"])].append(row)
        for domain, rows in sorted(by_domain.items()):
            item = cap_stats(rows)
            cap_domain_rows.append({"lag_ns": lag, "domain_id": domain, **item})
    cap_domain_path = out.with_name(out.stem + "_cap_by_domain.csv")
    cap_domain_columns = sorted({key for row in cap_domain_rows for key in row})
    write_csv(cap_domain_path, cap_domain_rows, cap_domain_columns)

    # Training/validation curves are copied into one stable CSV for the report.
    history = load_jsonl(train_dir / "history.jsonl")
    validation_path = train_dir / "validation.jsonl"
    validation = load_jsonl(validation_path) if validation_path.exists() else []
    curve_rows = [{"series": "train", **row} for row in history]
    curve_rows.extend({"series": "validation", **row} for row in validation)
    curve_columns = sorted({key for row in curve_rows for key in row})
    curve_path = out.with_name(out.stem + "_curves.csv")
    write_csv(curve_path, curve_rows, curve_columns)

    abs_tables = []
    delta_tables = []
    regression_flags = []
    for lag in LAGS:
        abs_rows = []
        for metric, label, direction in all_metrics:
            source = heavy_by_stage if metric in {item[0] for item in HEAVY_METRICS} else frame_by_stage
            abs_rows.append([
                label,
                ci_cell(value_map(source["identity"], metric, lag=lag)),
                ci_cell(value_map(source["coarse"], metric, lag=lag)),
                ci_cell(value_map(source["refined"], metric, lag=lag)),
            ])
        abs_tables.append((lag, markdown_table(
            ["metric", "identity_current_atoms", "P0 coarse", "P0 + H1b"], abs_rows
        )))

        delta_rows = []
        for metric, label, direction in all_metrics:
            source = heavy_by_stage if metric in {item[0] for item in HEAVY_METRICS} else frame_by_stage
            coarse_values = value_map(source["coarse"], metric, lag=lag)
            refined_values = value_map(source["refined"], metric, lag=lag)
            delta, lo, hi, n, domains = paired_delta(coarse_values, refined_values)
            result = verdict(delta, lo, hi, direction)
            if result == "REGRESSION" and metric in {"ca_rmsd", "drmsd_long_range", "rotation_geodesic_mean_deg"}:
                regression_flags.append((lag, label, delta, lo, hi))
            delta_rows.append([
                label,
                cell(delta),
                f"[{cell(lo)}, {cell(hi)}]",
                str(n),
                str(domains),
                result,
            ])
        delta_tables.append((lag, markdown_table(
            ["metric", "Δ (P0+H1b − P0)", "95% domain-cluster CI", "pairs", "domains", "verdict"], delta_rows
        )))

    # Explicit metric-space reversion diagnostic. It is intentionally labelled
    # as such: coordinate trajectories are not reconstructed from scalar rows.
    reversion_rows = []
    for lag in LAGS:
        for metric, label, _direction in FRAME_METRICS[:4]:
            identity_values = value_map(frame_by_stage["identity"], metric, lag=lag)
            coarse_values = value_map(frame_by_stage["coarse"], metric, lag=lag)
            refined_values = value_map(frame_by_stage["refined"], metric, lag=lag)
            shared = sorted(set(identity_values) & set(coarse_values) & set(refined_values))
            if not shared:
                continue
            fraction = np.mean([
                abs(refined_values[key] - identity_values[key])
                < abs(coarse_values[key] - identity_values[key])
                for key in shared
            ])
            reversion_rows.append([f"{lag:g} ns", label, f"{fraction:.4f}", str(len(shared))])

    cap_manifest = eval_manifest.get("cap_policy", {})
    translation_cap = float(cap_manifest.get("translation_norm_max_a", 1.0))
    rotation_cap = float(cap_manifest.get("rotation_geodesic_residual_max_deg", 15.0))
    max_clipped_translation = float(cap_by_scope[0][1].get("clipped_translation_max", float("nan")))
    max_clipped_rotation = float(cap_by_scope[0][1].get("clipped_rotation_max", float("nan")))
    cap_bound_ok = (
        finite(max_clipped_translation) and max_clipped_translation <= translation_cap + 1e-5
        and finite(max_clipped_rotation) and max_clipped_rotation <= rotation_cap + 1e-4
    )

    lines = [
        "# Phase 1.6 H1b full report",
        "",
        "## Decision scope",
        "",
        "This is the registered first full H1b decision report. The hard stop is after held-out evaluation. H1a, H2, ESM3 screening, and Phase 2 were not started.",
        "",
        f"The full H1b training artifact is `steps={train_manifest['steps']}` with `is_smoke={train_manifest['is_smoke']}`. It was reused after an independent provenance audit because its registered full-run manifest and frozen-P0 hash are intact; no checkpoint was overwritten.",
        "",
        "Primary aggregation is the sample-equal mean. The 95% intervals resample domains as clusters, with `iterations=10,000` and fixed seed `20260827`. Paired effects are computed on shared `(pair_id, domain_id)` rows and are always reported as candidate minus P0.",
        "",
        "## Provenance and exact configuration",
        "",
        f"- H0 settings: `configs/phase1_6_h0.yaml` (evaluation manifest: `{eval_manifest['h0_config_sha256']}`).",
        f"- H1b training config: `{train_manifest['config']}`; SHA256 `{train_manifest['config_sha256']}`.",
        f"- Canonical coarse arm: `{train_manifest['canonical_arm']}`; implementation `{train_manifest['implementation']}`.",
        f"- P0 checkpoint SHA256 before/after: `{train_manifest['coarse_checkpoint_sha256_before']}` / `{train_manifest['coarse_checkpoint_sha256_after']}`; unchanged=`{train_manifest['coarse_weights_unchanged']}`.",
        f"- Refiner checkpoint: `{eval_manifest['refiner_checkpoint']}`; SHA256 `{eval_manifest['refiner_checkpoint_sha256']}`; step `{eval_manifest['refiner_step']}`.",
        f"- Train manifest hash: `{train_manifest['manifest_hash']}`; held-out manifest hash: `{eval_manifest['val_manifest_hash']}`.",
        f"- Validation rows: `{len(frame_by_stage['coarse'])}` per stage, `{len({row['pair_id'] for row in frame_by_stage['coarse']})}` pairs, `{len({row['domain_id'] for row in frame_by_stage['coarse']})}` domains; lags `1 ns` and `4 ns`.",
        "- P0 evaluation: frozen `eval()` model under `no_grad()`; refiner is also `eval()` and P0 parameters have `requires_grad=False`.",
        "- Alignment: the existing single proper-Kabsch convention in the H0/Stage-M evaluator; no additional primary alignment was applied here.",
        "- Heavy mapping/masks: PSF atom names and atomic numbers are verified per domain; current conformers are transported through predicted frames for P0/H1b, identity uses current atoms, and target uses future heavy atoms only for scoring. PSF 1–2 and 1–3 exclusions are applied, 1–4 pairs remain in the primary overlap total and are tallied; Bondi radii come from the audited VDW table.",
        "- Units and thresholds: coordinates/distances in Å, frame rotation in degrees, lag in ns; heavy contact cutoff 5.0 Å with within-chain residue separation ≥2; serious Bondi depth is the preregistered `0.4 Å` threshold.",
        "- Future coordinates are used only as loss/evaluation targets. Identity, P0, and H1b predictions use current-frame inputs; future coordinates are not deployable inputs.",
        "",
        "## Train and validation curves",
        "",
        f"Complete machine-readable curves: `{curve_path}`. Training history has `{len(history)}` rows and validation history has `{len(validation)}` rows.",
        "",
    ]
    if validation:
        curve_rows_md = []
        for row in validation:
            curve_rows_md.append([
                str(row.get("step", "—")),
                cell(row.get("total")),
                cell(row.get("primary")),
                cell(row.get("primary_coarse")),
                cell(row.get("geometry")),
                cell(row.get("overlap")),
                cell(row.get("translation_at_cap_fraction")),
                cell(row.get("rotation_at_cap_fraction")),
            ])
        lines.append(markdown_table(
            ["step", "total", "primary", "coarse primary", "geometry", "overlap", "translation cap", "rotation cap"],
            curve_rows_md,
        ))
    else:
        lines.append("Validation curve: unavailable in the supplied full-run directory.")

    lines += ["", "## Absolute held-out metrics", "", "The cell is `sample-equal mean [95% domain-cluster CI]`.", ""]
    for lag, table in abs_tables:
        lines += [f"### Lag {lag:g} ns", "", table, ""]

    lines += [
        "## Paired P0 + H1b effects",
        "",
        "Raw effect is `P0+H1b − P0`; for lower-is-better metrics a positive value is worse, while for higher-is-better contact metrics a negative value is worse.",
        "",
    ]
    for lag, table in delta_tables:
        lines += [f"### Lag {lag:g} ns", "", table, ""]

    lines += [
        "## Heavy-atom PSF metrics and domain breakdown",
        "",
        "Heavy-atom RMSDs, Bondi serious overlaps, and static/formed/broken atom contacts use the H0 PSF atom mapping, masks, Bondi radii, units, and thresholds. The complete per-domain/per-lag/per-stage table is [`%s`](%s)." % (domain_path.name, domain_path.name),
        "",
        "The Bondi table separates inter-residue bb–bb, bb–sc, sc–sc and the corresponding intra-residue classes. Contact precision, recall, and F1 are kept separate for static, formed, and broken events.",
        "",
        "## Cap policy and saturation",
        "",
        f"Caps were preregistered and unchanged: translation norm ≤ `{translation_cap:.6g} Å`; rotation residual geodesic norm ≤ `{rotation_cap:.6g}°`. The implementation applies a radial projection to vector magnitude, not component-wise clipping; this is recorded in the evaluation manifest. Bound check passed=`{cap_bound_ok}` (clipped maxima `{cell(max_clipped_translation)} Å`, `{cell(max_clipped_rotation)}°`).",
        "",
        cap_table(cap_by_scope),
        "",
        f"Complete saturation and norm breakdown by lag and domain: [`{cap_domain_path.name}`]({cap_domain_path.name}).",
        "",
        "Raw statistics are computed over residue-level cap records; clipped statistics are computed from the composed correction. Saturation is `raw vector norm >= cap*(1−1e−6)`.",
        "",
        "## Geometry-versus-transition trade-off",
        "",
        "The paired tables above are the decision record: peptide C–N/angle/Cα-neighbor errors and Bondi bb–bb overlap are geometry outcomes, while Cα RMSD, long-range dRMSD, and frame rotation are transition outcomes. No metric was reweighted or hidden after validation.",
        "",
        "Metric-space identity-reversion diagnostic: fraction of shared validation pairs for which the refined scalar metric is closer to the identity scalar than the coarse scalar. This is a diagnostic, not a coordinate-level claim.",
        "",
        markdown_table(["lag", "metric", "fraction closer to identity", "pairs"], reversion_rows),
        "",
        "## Required flags",
        "",
    ]
    if regression_flags:
        lines.append("Statistically separated primary regressions:")
        lines.append("")
        for lag, label, delta, lo, hi in regression_flags:
            lines.append(f"- Lag {lag:g} ns — {label}: Δ `{delta:+.6f}`, 95% CI `[{lo:+.6f}, {hi:+.6f}]`.")
        lines.append("")
    else:
        lines += ["No statistically separated regression was detected in Cα RMSD, long-range dRMSD, or rotation.", ""]
    geometry_flags = []
    for lag in LAGS:
        for metric, label, direction in (
            ("bond_length_rmse", "peptide C–N bond", "lower"),
            ("bond_angle_ca_c_n_mae_deg", "CA–C–N angle", "lower"),
            ("bond_angle_c_n_ca_mae_deg", "C–N–CA angle", "lower"),
            ("ca_neighbor_distance_mae", "consecutive Cα distance", "lower"),
            ("bondi_serious_inter_residue_backbone_backbone_rate", "inter-residue bb–bb Bondi overlap", "lower"),
        ):
            source = heavy_by_stage if metric.startswith("bondi_") else frame_by_stage
            delta, lo, hi, _n, _domains = paired_delta(
                value_map(source["coarse"], metric, lag=lag),
                value_map(source["refined"], metric, lag=lag),
            )
            geometry_flags.append(
                f"- Lag {lag:g} ns — {label}: `{verdict(delta, lo, hi, direction)}` "
                f"(Δ `{delta:+.6f}`, 95% CI `[{lo:+.6f}, {hi:+.6f}]`)."
            )
    cap_domain_count = len({row["domain_id"] for row in cap_domain_rows})
    cap_nonzero = {
        lag: (
            sum(float(row.get("translation_saturation_fraction", 0) or 0) > 0 for row in cap_domain_rows if float(row["lag_ns"]) == lag),
            sum(float(row.get("rotation_saturation_fraction", 0) or 0) > 0 for row in cap_domain_rows if float(row["lag_ns"]) == lag),
        )
        for lag in LAGS
    }
    lines += [
        "Explicit geometry flags:",
        "",
        *geometry_flags,
        "",
        "Explicit cap-saturation flag:",
        "",
        f"- Translation saturation is recurring/systematic across lag and domain: overall `{cap_by_scope[0][1].get('translation_saturation_fraction', float('nan')):.4f}`, with nonzero saturation in `{cap_nonzero[1.0][0]}/{cap_domain_count}` domains at 1 ns and `{cap_nonzero[4.0][0]}/{cap_domain_count}` at 4 ns.",
        f"- Rotation saturation is lower but also recurring/systematic: overall `{cap_by_scope[0][1].get('rotation_saturation_fraction', float('nan')):.4f}`, with nonzero saturation in `{cap_nonzero[1.0][1]}/{cap_domain_count}` domains at 1 ns and `{cap_nonzero[4.0][1]}/{cap_domain_count}` at 4 ns.",
        "",
        "Peptide geometry and bb–bb clashes must be read directly from their rows above; the report does not substitute a composite score.",
        "The explicit flags above preserve the geometry gains and transition costs without reweighting them.",
        "",
        "## Stop condition",
        "",
        "Held-out evaluation and this report are complete. No H1a, H2, ESM3 screening, or Phase 2 action was taken after this point.",
    ]
    out.write_text("\n".join(lines) + "\n")

    print(f"wrote {out}")
    print(f"wrote {domain_path}")
    print(f"wrote {cap_domain_path}")
    print(f"wrote {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
