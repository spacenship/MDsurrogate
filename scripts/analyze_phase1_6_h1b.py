#!/usr/bin/env python
"""H1b -- did the refiner fix what Stage M found broken, and what did it cost?

    python scripts/analyze_phase1_6_h1b.py \
        --records runs/phase1_6_h1b_eval_seed0/records.jsonl \
        --stage-m runs/phase1_6_extended_metrics_seed0/records.jsonl \
        --out docs/phase1_6_h1b_results.md

Stage M's headline was that **every trained arm was worse than "nothing moves"**
on all ten physical-validity cells. H1b was built for exactly those cells, so
this report is organised around them and around one question per cell: did the
refined prediction close the gap to the identity baseline, and did the primary
transition metrics pay for it.

Three properties.

*The coarse stage is a control, not a result.* It is the frozen arm on the frozen
manifest, computed by the same ``extended_metric_records`` Stage M used, so it
must reproduce the Stage M numbers. If ``--stage-m`` is given, that reproduction
is checked pair by pair and reported first; a mismatch there invalidates
everything below it, so it is not buried.

*The cost is never omitted.* A refiner can buy geometry by giving back transition
accuracy. Both are printed, in the same table, with the same sign convention, and
the summary states the trade in one line whichever way it went.

*It does not declare victory on a ratio alone.* "5x better than the arm" means
nothing if the arm was 1651x worse than doing nothing. Every validity row carries
its ratio to the **identity baseline**, which is the bar Stage M set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_phase1_6 import format_ci  # noqa: E402
from analyze_phase1_6_extended import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    LAGS,
    cell,
    cluster_bootstrap,
    paired_delta,
    write_csv,
)
from force_md.transition.extended_metrics import HIGHER_IS_BETTER  # noqa: E402

STAGES = ("identity", "coarse", "refined")

#: The Stage M validity cells, in the order that report used.
VALIDITY = (
    ("bond_length_rmse", "peptide C-N bond RMSE (A)"),
    ("bond_angle_mae_deg", "backbone angle MAE (deg)"),
    ("ca_neighbor_distance_mae", "consecutive Ca distance MAE (A)"),
    ("clash_rate", "Ca clash rate"),
    ("backbone_torsion_mae_deg", "backbone torsion MAE (deg)"),
)

#: What the geometry must not be bought with.
PRIMARY = (
    ("ca_rmsd", "Ca RMSD (A)"),
    ("drmsd_long_range", "dRMSD, |i-j| >= 6 (A)"),
    ("rotation_geodesic_mean_deg", "frame rotation error (deg)"),
    ("contact_f1", "contact F1"),
    ("formed_contact_f1", "formed-contact F1"),
)

COMPARISONS = (
    ("coarse", "refined", "H1b's own effect: refined vs the frozen arm"),
    ("identity", "refined", "against the bar Stage M set: refined vs nothing moves"),
    ("identity", "coarse", "the Stage M gap this stage set out to close"),
)


def load(path: str, key: str = "h1b_stage") -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record[key]].append(record)
    for stage, records in rows.items():
        ids = [r["pair_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{stage}: duplicate pair_id(s)")
    return dict(rows)


def values_of(records: list[dict], metric: str, lag: float) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for record in records:
        if record["lag_ns"] != lag:
            continue
        value = record.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        out[(record["pair_id"], record["domain_id"])] = float(value)
    return out


def mean_of(records: list[dict], metric: str, lag: float) -> float:
    values = values_of(records, metric, lag)
    return float(np.mean(list(values.values()))) if values else float("nan")


# --------------------------------------------------------------------------


def reproduction_check(by_stage: dict, stage_m_path: str | None) -> list[str]:
    """Does the ``coarse`` stage reproduce Stage M pair for pair?

    The coarse stage runs the same frozen weights over the same frozen manifest
    through the same metric function, so the answer must be yes to the last bit.
    It is the only thing standing between "H1b improved the geometry" and "the
    evaluation harness differs from the one the baseline came from".
    """
    lines = ["## 1. Does the coarse control reproduce Stage M?", ""]
    if stage_m_path is None:
        lines.append(
            "Not checked: `--stage-m` was not given. Every comparison below is "
            "internally consistent, but the link to the published Stage M table "
            "is unverified."
        )
        return lines

    stage_m = load(stage_m_path, key="canonical_arm")
    arm = {r["canonical_arm"] for r in by_stage["coarse"]}
    if len(arm) != 1:
        lines.append(f"Cannot check: the coarse rows name {len(arm)} arms.")
        return lines
    name = arm.pop()
    if name not in stage_m:
        lines.append(f"Cannot check: `{name}` is absent from the Stage M records.")
        return lines

    reference = {r["pair_id"]: r for r in stage_m[name]}
    metrics = [m for m, _ in VALIDITY] + [m for m, _ in PRIMARY]
    worst, worst_metric, compared = 0.0, None, 0
    for record in by_stage["coarse"]:
        other = reference.get(record["pair_id"])
        if other is None:
            continue
        compared += 1
        for metric in metrics:
            a, b = record.get(metric), other.get(metric)
            if a is None or b is None:
                continue
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                continue
            difference = abs(float(a) - float(b))
            if difference > worst:
                worst, worst_metric = difference, metric

    lines.append(
        f"- `{name}`, {compared} pairs matched against "
        f"`{Path(stage_m_path).name}`, over {len(metrics)} metrics."
    )
    if worst == 0.0:
        lines.append(
            "- **Reproduces exactly** (max absolute difference 0). The H1b "
            "harness and the Stage M harness agree bit for bit, so a refined "
            "number below can be quoted beside a Stage M number."
        )
    elif worst < 1e-9:
        lines.append(
            f"- Reproduces to {worst:.3e} on `{worst_metric}` — floating-point "
            "reassociation, not a difference in what was computed."
        )
    else:
        lines.append(
            f"- **FAILS: max absolute difference {worst:.6g} on "
            f"`{worst_metric}`.** The coarse control does not reproduce Stage M, "
            "so the comparisons below measure the harness as well as the "
            "refiner. Nothing here should be reported until this is resolved."
        )
    return lines


def validity_table(by_stage: dict, lag: float) -> list[str]:
    """The ten Stage M cells: all three stages, with the ratio to identity."""
    lines = [
        f"**Physical validity, lag {lag:g} ns** — lower is better throughout. "
        "The `x identity` columns are the Stage M framing: how many times worse "
        "than doing nothing.",
        "",
        "| metric | identity | coarse arm | refined | coarse ×id | refined ×id |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, label in VALIDITY:
        base = mean_of(by_stage["identity"], metric, lag)
        coarse = mean_of(by_stage["coarse"], metric, lag)
        refined = mean_of(by_stage["refined"], metric, lag)
        ratio = (lambda v: "—" if not (math.isfinite(v) and base) else f"{v / base:.1f}×")
        lines.append(
            f"| {label} | {cell(base, 5)} | {cell(coarse, 5)} | "
            f"**{cell(refined, 5)}** | {ratio(coarse)} | {ratio(refined)} |"
        )
    return lines


def primary_table(by_stage: dict, lag: float) -> list[str]:
    lines = [
        f"**Primary transition metrics, lag {lag:g} ns** — what the geometry must "
        "not be bought with.",
        "",
        "| metric | identity | coarse arm | refined | refined − coarse |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in PRIMARY:
        base = mean_of(by_stage["identity"], metric, lag)
        coarse = mean_of(by_stage["coarse"], metric, lag)
        refined = mean_of(by_stage["refined"], metric, lag)
        higher = metric in HIGHER_IS_BETTER
        delta = (refined - coarse) if higher else (coarse - refined)
        mark = "" if not math.isfinite(delta) else (" ✓" if delta > 0 else " ✗")
        lines.append(
            f"| {label} | {cell(base, 5)} | {cell(coarse, 5)} | "
            f"**{cell(refined, 5)}** | {cell(delta, 5)}{mark} |"
        )
    lines.append("")
    lines.append(
        "`refined − coarse` is sign-normalised so **positive means refined is "
        "better**, for lower- and higher-is-better metrics alike."
    )
    return lines


def delta_table(by_stage: dict, metrics, lag: float) -> list[str]:
    lines = [
        f"Paired Δ over shared samples, lag {lag:g} ns, domain-cluster bootstrap. "
        "**Positive = candidate better.**",
        "",
        "| comparison | metric | Δ [95% CI] | relative | significant |",
        "|---|---|---|---|---|",
    ]
    for control, candidate, question in COMPARISONS:
        if control not in by_stage or candidate not in by_stage:
            continue
        for metric, label in metrics:
            result = paired_delta(
                values_of(by_stage[control], metric, lag),
                values_of(by_stage[candidate], metric, lag),
                metric,
            )
            if not result["n"]:
                continue
            lines.append(
                f"| {candidate} − {control} | {label} | "
                f"{format_ci(result['delta'], result['lo'], result['hi'], 5)} | "
                f"{result['relative'] * 100:+.1f}% | "
                f"{'yes' if result['significant'] else 'no'} |"
            )
        lines.append(f"| *{question}* | | | | |")
    return lines


def correction_section(path: Path) -> list[str]:
    """How far the refiner actually moved things, and whether the caps bind."""
    lines = ["## 5. How much did the refiner move, and do the caps bind?", ""]
    if not path.exists():
        lines.append(f"Not available: `{path.name}` was not written.")
        return lines
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        lines.append("Not available: no rows.")
        return lines

    def stat(key: str) -> tuple[float, float, float]:
        values = {(r["pair_id"], r["domain_id"]): r[key] for r in rows}
        return cluster_bootstrap(values)

    lines.extend([
        "| quantity | mean [95% CI] |",
        "|---|---|",
    ])
    for key, label in (
        ("correction_translation_mean_a", "translation, mean over residues (Å)"),
        ("correction_translation_max_a", "translation, per-structure max (Å)"),
        ("correction_rotation_mean_deg", "rotation, mean over residues (°)"),
        ("correction_rotation_max_deg", "rotation, per-structure max (°)"),
        ("translation_at_cap_fraction", "residues at the translation cap"),
        ("rotation_at_cap_fraction", "residues at the rotation cap"),
    ):
        lines.append(f"| {label} | {format_ci(*stat(key), 4)} |")
    at_cap = max(
        float(np.mean([r["translation_at_cap_fraction"] for r in rows])),
        float(np.mean([r["rotation_at_cap_fraction"] for r in rows])),
    )
    lines.append("")
    if at_cap > 0.25:
        lines.append(
            f"**The caps bind: {at_cap:.1%} of residues sit at a bound.** The "
            "correction is being clipped for a large fraction of the structure, "
            "so the reported result is as much a property of the cap as of the "
            "refiner, and the cap should be revisited before any further tuning."
        )
    else:
        lines.append(
            f"At most {at_cap:.1%} of residues sit at a cap, so the bounds are "
            "protective rather than binding: the refiner is choosing these "
            "magnitudes, not being clipped to them."
        )
    return lines


def summary(by_stage: dict) -> list[str]:
    """One paragraph that states the trade whichever way it went."""
    lines = ["## 6. What H1b did", ""]
    for lag in LAGS:
        gains, losses = [], []
        for metric, label in VALIDITY:
            result = paired_delta(
                values_of(by_stage["coarse"], metric, lag),
                values_of(by_stage["refined"], metric, lag),
                metric,
            )
            if not result["n"] or not math.isfinite(result["delta"]):
                continue
            target = gains if result["delta"] > 0 else losses
            target.append((label, result))
        costs = []
        for metric, label in PRIMARY:
            result = paired_delta(
                values_of(by_stage["coarse"], metric, lag),
                values_of(by_stage["refined"], metric, lag),
                metric,
            )
            if result["n"] and math.isfinite(result["delta"]) and result["delta"] < 0:
                costs.append((label, result))

        lines.append(f"**Lag {lag:g} ns.**")
        lines.append("")
        if gains:
            lines.append(
                f"- Validity improved on {len(gains)}/{len(VALIDITY)} cells: "
                + ", ".join(
                    f"{label} ({r['relative'] * 100:+.1f}%"
                    + (", significant)" if r["significant"] else ")")
                    for label, r in gains
                )
            )
        if losses:
            lines.append(
                f"- Validity **worsened** on {len(losses)}/{len(VALIDITY)}: "
                + ", ".join(
                    f"{label} ({r['relative'] * 100:+.1f}%)" for label, r in losses
                )
            )
        if not gains and not losses:
            lines.append("- No validity cell was measurable at this lag.")
        if costs:
            lines.append(
                "- **Paid for in:** "
                + ", ".join(
                    f"{label} ({r['relative'] * 100:+.1f}%)" for label, r in costs
                )
            )
        else:
            lines.append(
                "- No primary transition metric regressed: the geometry was not "
                "bought with transition accuracy."
            )

        # The bar Stage M set.
        cleared = []
        for metric, label in VALIDITY:
            base = mean_of(by_stage["identity"], metric, lag)
            refined = mean_of(by_stage["refined"], metric, lag)
            if math.isfinite(base) and math.isfinite(refined) and refined <= base:
                cleared.append(label)
        lines.append(
            f"- Cells now at or better than the identity baseline: "
            f"**{len(cleared)}/{len(VALIDITY)}**"
            + (f" ({', '.join(cleared)})" if cleared else
               " — the bar Stage M set is still not cleared on any cell")
        )
        lines.append("")
    return lines


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--stage-m", default=None,
                        help="Stage M records, to verify the coarse control")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-dir", default=None)
    args = parser.parse_args()

    by_stage = load(args.records)
    missing = [s for s in STAGES if s not in by_stage]
    if missing:
        raise SystemExit(f"records are missing stage(s): {missing}")

    manifest_path = Path(args.records).with_name("reproducibility_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    lines = [
        "# Phase 1.6 H1b — backbone frame constraint refiner, results", "",
        f"Records: `{args.records}`  ",
        f"Refiner: `{manifest.get('refiner_checkpoint', '?')}` at step "
        f"{manifest.get('refiner_step', '?')}, on `"
        f"{manifest.get('canonical_arm', '?')}`  ",
        f"Bootstrap: {BOOTSTRAP_ITERATIONS:,} resamples over domains, seed "
        f"{BOOTSTRAP_SEED}  ",
        "Neither the coarse checkpoint nor the refiner checkpoint was modified.",
        "",
        f"- {sum(len(v) for v in by_stage.values())} records, "
        f"{len(by_stage['coarse'])} per stage, "
        f"{len({r['pair_id'] for r in by_stage['coarse']})} pairs, "
        f"{len({r['domain_id'] for r in by_stage['coarse']})} domains.",
        "",
    ]
    lines.extend(reproduction_check(by_stage, args.stage_m))
    lines.extend(["", "## 2. Physical validity — the ten Stage M cells", ""])
    for lag in LAGS:
        lines.extend(validity_table(by_stage, lag))
        lines.append("")
    lines.extend(["## 3. The cost: primary transition metrics", ""])
    for lag in LAGS:
        lines.extend(primary_table(by_stage, lag))
        lines.append("")
    lines.extend(["## 4. Paired deltas", ""])
    for lag in LAGS:
        lines.extend(delta_table(by_stage, VALIDITY + PRIMARY, lag))
        lines.append("")
    lines.extend(
        correction_section(Path(args.records).with_name("correction_magnitudes.jsonl"))
    )
    lines.append("")
    lines.extend(summary(by_stage))

    csv_dir = Path(args.csv_dir or Path(args.records).parent)
    rows = []
    for stage in STAGES:
        for lag in LAGS:
            per_domain: dict[str, list[dict]] = defaultdict(list)
            for record in by_stage[stage]:
                if record["lag_ns"] == lag:
                    per_domain[record["domain_id"]].append(record)
            for domain, group in sorted(per_domain.items()):
                row = {"h1b_stage": stage, "lag_ns": lag, "domain_id": domain,
                       "n_samples": len(group)}
                for metric, _ in VALIDITY + PRIMARY:
                    finite = [
                        float(r[metric]) for r in group
                        if r.get(metric) is not None and math.isfinite(float(r[metric]))
                    ]
                    row[metric] = float(np.mean(finite)) if finite else float("nan")
                rows.append(row)
    write_csv(
        csv_dir / "h1b_domain_summary.csv", rows,
        ["h1b_stage", "lag_ns", "domain_id", "n_samples"]
        + [m for m, _ in VALIDITY + PRIMARY],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print(f"wrote {csv_dir / 'h1b_domain_summary.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
