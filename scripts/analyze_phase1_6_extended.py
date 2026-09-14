#!/usr/bin/env python
"""Stage M2 -- aggregate, compare and report the Phase 1.6 extended metrics.

    python scripts/analyze_phase1_6_extended.py \
        --records runs/phase1_6_extended_metrics_seed0/records.jsonl \
        --out docs/phase1_6_results_extended.md \
        --plots docs/figures

This is a **layer on** ``scripts/analyze_phase1_6.py``, not a second analysis
framework: the cluster bootstrap, the paired delta and the SVG canvas are
imported from it, so a Stage B number and a Stage M number are produced by the
same statistics. What is added here is the aggregation rules the extended schema
needs and that the Stage B analyser has no notion of -- pooled contact counts,
residue-pooled torsions, and a sign convention that survives higher-is-better
metrics.

Three things it refuses to do.

*It does not average macro and micro into one number.* Contact metrics are
reported twice: a sample-macro mean over samples that had an event, and a micro
metric recomputed from pooled TP/FP/FN. They answer different questions and a
blend answers neither.

*It does not build a composite score.* There is no weighted sum of RMSD, dRMSD
and contact F1 anywhere in this file. A composite hides exactly the trade-off
the Pareto plots exist to show.

*It does not impute.* An arm with no records is ``pending``; a metric that was
not measured is ``n/a`` with the reason from the manifest, never 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_phase1_6 import Svg, _COLOURS, _limits, format_ci  # noqa: E402
from force_md.transition.arms import CANONICAL_ARMS  # noqa: E402
from force_md.transition.extended_metrics import HIGHER_IS_BETTER  # noqa: E402

IDENTITY = "identity_baseline"

#: Brief section 14. Two co-primaries, no composite of them.
CO_PRIMARY = ("ca_rmsd", "drmsd_long_range")
KEY_SECONDARY = (
    "contact_f1", "formed_contact_f1", "broken_contact_f1",
    "backbone_torsion_mae_deg", "rotation_geodesic_mean_deg",
)
VALIDITY = (
    "bond_length_rmse", "bond_angle_mae_deg", "ca_neighbor_distance_mae",
    "clash_rate",
)
PAIR_SUBSETS = (
    "drmsd_all", "drmsd_local", "drmsd_medium", "drmsd_long_range",
    "drmsd_current_spatial_edges",
)
TORSIONS = ("phi_mae_deg", "psi_mae_deg", "omega_mae_deg", "backbone_torsion_mae_deg")

#: Every paired comparison the brief asks for, control first.
COMPARISONS = (
    ("S0_structure_history", "P0_pair_geometry_control", "P0 − S0 (pair architecture)"),
    ("P0_pair_geometry_control", "P1_pair_physics_frozen", "P1 − P0 (pair physics)"),
    ("P1_pair_physics_frozen", "P2_pair_physics_moments", "P2 − P1 (force moments)"),
    ("S0_structure_history", "S2_node_physics_152d", "S2 − S0 (node physics)"),
    ("S0_structure_history", "O_current_gt_force_oracle", "O − S0 (GT-force oracle)"),
    ("S2_node_physics_152d", "P1_pair_physics_frozen",
     "P1 − S2 (reference only; architectures differ)"),
)

#: Bootstrap settings, fixed here and recorded in the report.
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260827
LAGS = (1.0, 4.0)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_records(path: str) -> dict[str, list[dict]]:
    """``{canonical_arm: [record, ...]}``, with duplicate ids refused.

    Brief section 16.19. A duplicate ``pair_id`` does not raise anywhere in the
    Stage B analyser -- it silently overwrites, keeping one row per group. That
    failure is invisible in the output, so it is checked here rather than hoped
    against.
    """
    by_arm: dict[str, list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_arm[record["canonical_arm"]].append(record)

    for arm, records in by_arm.items():
        ids = [r["pair_id"] for r in records]
        if len(set(ids)) != len(ids):
            seen, duplicates = set(), []
            for identifier in ids:
                if identifier in seen:
                    duplicates.append(identifier)
                seen.add(identifier)
            raise SystemExit(
                f"{arm}: {len(duplicates)} duplicate pair_id(s), e.g. "
                f"{duplicates[:3]}. Every downstream comparison keys on this id."
            )
    return dict(by_arm)


def check_same_samples(by_arm: dict[str, list[dict]]) -> list[str]:
    """Brief section 16.20: every arm must have scored the identical id set."""
    problems, reference, reference_arm = [], None, None
    for arm, records in sorted(by_arm.items()):
        ids = frozenset(r["pair_id"] for r in records)
        if reference is None:
            reference, reference_arm = ids, arm
            continue
        if ids != reference:
            problems.append(
                f"{arm} scored {len(ids)} samples against {reference_arm}'s "
                f"{len(reference)} ({len(reference - ids)} missing, "
                f"{len(ids - reference)} extra)"
            )
    return problems


def values_of(
    records: list[dict], metric: str, *, lag: float | None = None,
    temperature: str | None = None, length_bin: str | None = None,
) -> dict[tuple, float]:
    """``{(pair_id, domain): value}`` for the finite rows of one slice."""
    out: dict[tuple, float] = {}
    for record in records:
        if lag is not None and record["lag_ns"] != lag:
            continue
        if temperature is not None and record["temperature"] != temperature:
            continue
        if length_bin is not None and record["length_bin"] != length_bin:
            continue
        value = record.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        out[(record["pair_id"], record["domain_id"])] = float(value)
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def cluster_bootstrap(
    values: dict[tuple, float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Sample-equal mean and a 95% interval, resampling **domains**.

    The domain is the cluster because the split was made by domain: two pairs
    from the same protein are not two independent looks at the question, and
    treating them as such produces intervals several times too narrow.

    Algebraically identical to ``analyze_phase1_6.cluster_bootstrap`` -- the mean
    of a concatenation is the ratio of summed sums to summed counts -- but it
    resamples the two small arrays instead of rebuilding a 1,760-element vector
    ten thousand times, which is what makes 10,000 iterations affordable across
    every metric. ``test_fast_cluster_bootstrap_matches_the_stage_b_implementation``
    pins the two together.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for (_pair_id, domain), value in values.items():
        groups[domain].append(value)
    order = sorted(groups)
    sums = np.array([sum(groups[d]) for d in order], dtype=float)
    counts = np.array([len(groups[d]) for d in order], dtype=float)

    observed = float(sums.sum() / counts.sum())
    if len(order) < 2:
        return observed, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(order), size=(iterations, len(order)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return observed, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(
    control: dict[tuple, float],
    candidate: dict[tuple, float],
    metric: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired ``delta`` over shared samples, **normalised so positive = better**.

    Brief section 13.2. Lower-is-better metrics give ``control - candidate``;
    higher-is-better metrics give ``candidate - control``. One convention for
    every table, so a reader never has to check which way a column points.
    """
    shared = sorted(set(control) & set(candidate))
    if not shared:
        return {"n": 0, "domains": 0, "delta": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "relative": float("nan"), "significant": False}
    higher_better = metric in HIGHER_IS_BETTER
    differences = {
        k: (candidate[k] - control[k]) if higher_better else (control[k] - candidate[k])
        for k in shared
    }
    delta, lo, hi = cluster_bootstrap(differences, iterations=iterations, seed=seed)
    base = float(np.mean([control[k] for k in shared]))
    return {
        "n": len(shared),
        "domains": len({k[1] for k in shared}),
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "relative": delta / abs(base) if base else float("nan"),
        "significant": bool(lo > 0 or hi < 0) if math.isfinite(lo) else False,
    }


def domain_macro(values: dict[tuple, float]) -> float:
    """Mean over domain means -- every protein counts once."""
    groups: dict[str, list[float]] = defaultdict(list)
    for (_pair_id, domain), value in values.items():
        groups[domain].append(value)
    if not groups:
        return float("nan")
    return float(np.mean([np.mean(v) for v in groups.values()]))


def residue_weighted(records: list[dict], metric: str, lag: float) -> float:
    """The Stage B ``*_micro`` weighting, for the reproduction check only.

    Reported beside the sample-equal mean because the two disagree by ~10% and
    the Stage B report never said which it was quoting. It is **not** the primary
    aggregation: weighting by residue count lets one 250-residue domain outweigh
    five 50-residue ones (audit section 3).
    """
    total, weight = 0.0, 0.0
    for record in records:
        if record["lag_ns"] != lag:
            continue
        value, count = record.get(metric), record.get("n_valid_residues")
        if value is None or count in (None, 0) or not math.isfinite(float(value)):
            continue
        total += float(value) * count
        weight += count
    return total / weight if weight else float("nan")


def pooled_contact(records: list[dict], prefix: str, lag: float) -> dict:
    """Micro contact metrics from pooled TP/FP/FN, plus the sample bookkeeping.

    Brief section 12.2 requires both this and the sample-macro mean, and requires
    that they are never blended. They are returned separately and printed in
    separate columns.
    """
    tp = fp = fn = 0
    with_event = 0
    without_event = 0
    total = 0
    for record in records:
        if record["lag_ns"] != lag:
            continue
        total += 1
        a, b, c = (record.get(f"{prefix}_tp"), record.get(f"{prefix}_fp"),
                   record.get(f"{prefix}_fn"))
        if a is None:
            continue
        tp, fp, fn = tp + a, fp + b, fn + c
        events = record.get(f"{prefix}_events_target")
        if events is None:
            events = record.get("contact_target_positives", 0)
        if events:
            with_event += 1
        else:
            without_event += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard,
        "samples": total, "with_event": with_event, "without_event": without_event,
    }


def pooled_torsion(records: list[dict], lag: float) -> dict:
    """Residue-pooled, sample-macro and domain-macro torsion error, side by side."""
    total, weight = 0.0, 0
    per_sample: dict[tuple, float] = {}
    for record in records:
        if record["lag_ns"] != lag:
            continue
        value, count = record.get("backbone_torsion_mae_deg"), record.get("n_valid_torsions")
        if value is None or not count or not math.isfinite(float(value)):
            continue
        total += float(value) * count
        weight += count
        per_sample[(record["pair_id"], record["domain_id"])] = float(value)
    return {
        "residue_pooled": total / weight if weight else float("nan"),
        "sample_macro": float(np.mean(list(per_sample.values()))) if per_sample
        else float("nan"),
        "domain_macro": domain_macro(per_sample),
        "residues": weight,
    }


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def arm_order(by_arm: dict) -> list[str]:
    return [a for a in CANONICAL_ARMS if a in by_arm]


def cell(value: float, digits: int = 4) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def per_arm_table(by_arm: dict, metric: str, lag: float, *, digits: int = 4) -> list[str]:
    """One metric, every arm, with the identity baseline as its own row."""
    higher = metric in HIGHER_IS_BETTER
    direction = "higher is better" if higher else "lower is better"
    lines = [
        f"| arm | {metric} @ {lag:g} ns ({direction}, 95% CI) | domain-macro | "
        "vs identity | samples | domains |",
        "|---|---|---|---|---|---|",
    ]
    identity_values = values_of(by_arm.get(IDENTITY, []), metric, lag=lag)
    identity_mean = (
        float(np.mean(list(identity_values.values()))) if identity_values
        else float("nan")
    )
    for arm in arm_order(by_arm):
        values = values_of(by_arm[arm], metric, lag=lag)
        mean, lo, hi = cluster_bootstrap(values)
        if math.isfinite(identity_mean) and identity_mean != 0 and math.isfinite(mean):
            relative = (mean - identity_mean) / abs(identity_mean) * 100
            improvement = -relative if not higher else relative
            versus = f"{improvement:+.2f}%"
        else:
            versus = "—"
        flag = " **[oracle]**" if CANONICAL_ARMS[arm].oracle else ""
        lines.append(
            f"| `{arm}`{flag} | {format_ci(mean, lo, hi, digits)} | "
            f"{cell(domain_macro(values), digits)} | {versus} | "
            f"{len(values)} | {len({k[1] for k in values})} |"
        )
    if identity_values:
        mean, lo, hi = cluster_bootstrap(identity_values)
        lines.append(
            f"| `identity_baseline` (nothing moves) | {format_ci(mean, lo, hi, digits)} | "
            f"{cell(domain_macro(identity_values), digits)} | — | "
            f"{len(identity_values)} | {len({k[1] for k in identity_values})} |"
        )
    for arm, spec in CANONICAL_ARMS.items():
        if arm not in by_arm and spec.stage != "gated":
            lines.append(f"| `{arm}` | **pending — not run** | — | — | — | — |")
    return lines


def comparison_table(by_arm: dict, metric: str, lag: float, *, digits: int = 4) -> list[str]:
    lines = [
        f"| comparison | Δ @ {lag:g} ns (positive = candidate better) | 95% CI | "
        "relative | shared | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for control, candidate, label in COMPARISONS:
        if control not in by_arm or candidate not in by_arm:
            lines.append(f"| {label} | pending | — | — | — | not run |")
            continue
        result = paired_delta(
            values_of(by_arm[control], metric, lag=lag),
            values_of(by_arm[candidate], metric, lag=lag),
            metric,
        )
        if not result["n"]:
            lines.append(f"| {label} | — | — | — | 0 | no shared samples |")
            continue
        verdict = (
            "improves" if result["significant"] and result["delta"] > 0
            else "worse" if result["significant"] and result["delta"] < 0
            else "not separated"
        )
        lines.append(
            f"| {label} | {result['delta']:+.{digits}f} | "
            f"[{result['lo']:+.{digits}f}, {result['hi']:+.{digits}f}] | "
            f"{result['relative'] * 100:+.2f}% | {result['n']} | {verdict} |"
        )
    return lines


def stratified_table(
    by_arm: dict, metric: str, lag: float, key: str, *, digits: int = 4
) -> list[str]:
    """Per-arm means split by ``temperature`` or ``length_bin``."""
    levels = sorted(
        {r[key] for records in by_arm.values() for r in records if key in r},
        key=lambda v: (float(v) if key == "temperature" else _bin_sort(v)),
    )
    if not levels:
        return [f"`{key}` was not recorded — no stratification possible."]
    header = "| arm | " + " | ".join(
        f"{v} K" if key == "temperature" else str(v) for v in levels
    ) + " |"
    lines = [header, "|---" * (len(levels) + 1) + "|"]
    for arm in arm_order(by_arm) + ([IDENTITY] if IDENTITY in by_arm else []):
        cells = []
        for level in levels:
            values = values_of(by_arm[arm], metric, lag=lag, **{key: level})
            cells.append(
                cell(float(np.mean(list(values.values()))), digits) if values else "—"
            )
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return lines


def _bin_sort(label: str) -> float:
    digits = "".join(c if c.isdigit() else " " for c in label).split()
    return float(digits[0]) if digits else 0.0


def contact_table(by_arm: dict, prefix: str, lag: float) -> list[str]:
    """Macro and micro contact metrics, in separate columns, never blended."""
    lines = [
        f"| arm | macro F1 (samples with an event) | micro F1 (pooled TP/FP/FN) | "
        "micro precision | micro recall | TP | FP | FN | samples w/ event | "
        "samples w/o event |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arm_order(by_arm) + ([IDENTITY] if IDENTITY in by_arm else []):
        records = by_arm[arm]
        macro = values_of(records, f"{prefix}_f1", lag=lag)
        pooled = pooled_contact(records, prefix, lag)
        lines.append(
            f"| `{arm}` | "
            f"{cell(float(np.mean(list(macro.values()))) if macro else float('nan'))} "
            f"({len(macro)}) | {cell(pooled['f1'])} | {cell(pooled['precision'])} | "
            f"{cell(pooled['recall'])} | {pooled['tp']} | {pooled['fp']} | "
            f"{pooled['fn']} | {pooled['with_event']} | {pooled['without_event']} |"
        )
    return lines


# --------------------------------------------------------------------------
# machine-readable output
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append("" if not math.isfinite(value) else repr(value))
            else:
                text = str(value)
                cells.append(f'"{text}"' if "," in text else text)
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def domain_summary_rows(by_arm: dict, metrics: list[str]) -> list[dict]:
    """One row per (arm, lag, domain), so the bootstrap's clusters are inspectable."""
    rows = []
    for arm in arm_order(by_arm) + ([IDENTITY] if IDENTITY in by_arm else []):
        for lag in LAGS:
            per_domain: dict[str, list[dict]] = defaultdict(list)
            for record in by_arm[arm]:
                if record["lag_ns"] == lag:
                    per_domain[record["domain_id"]].append(record)
            for domain, records in sorted(per_domain.items()):
                row = {
                    "arm": arm, "lag_ns": lag, "domain_id": domain,
                    "n_samples": len(records),
                    "n_valid_residues_mean": float(
                        np.mean([r["n_valid_residues"] for r in records])
                    ),
                }
                for metric in metrics:
                    finite = [
                        float(r[metric]) for r in records
                        if r.get(metric) is not None and math.isfinite(float(r[metric]))
                    ]
                    row[metric] = float(np.mean(finite)) if finite else float("nan")
                    row[f"{metric}_n"] = len(finite)
                rows.append(row)
    return rows


def paired_delta_rows(by_arm: dict, metrics: list[str]) -> list[dict]:
    rows = []
    for lag in LAGS:
        for control, candidate, label in COMPARISONS:
            if control not in by_arm or candidate not in by_arm:
                continue
            for metric in metrics:
                result = paired_delta(
                    values_of(by_arm[control], metric, lag=lag),
                    values_of(by_arm[candidate], metric, lag=lag),
                    metric,
                )
                rows.append({
                    "comparison": label, "control": control, "candidate": candidate,
                    "metric": metric, "lag_ns": lag,
                    "higher_is_better": metric in HIGHER_IS_BETTER,
                    "delta_positive_means_candidate_better": result["delta"],
                    "ci_lo": result["lo"], "ci_hi": result["hi"],
                    "relative": result["relative"], "n_shared": result["n"],
                    "n_domains": result["domains"],
                    "separated_at_95": result["significant"],
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                })
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _error_bar_plot(by_arm, metric, lag, directory, *, label=None) -> str | None:
    entries, identity = [], None
    for arm in arm_order(by_arm):
        values = values_of(by_arm[arm], metric, lag=lag)
        if not values:
            continue
        mean, lo, hi = cluster_bootstrap(values, iterations=2000)
        entries.append((arm, mean, lo, hi))
    baseline = values_of(by_arm.get(IDENTITY, []), metric, lag=lag)
    if baseline:
        identity = float(np.mean(list(baseline.values())))
    if not entries:
        return None

    spread = [v for _, m, lo, hi in entries for v in (m, lo, hi) if math.isfinite(v)]
    if identity is not None and math.isfinite(identity):
        spread.append(identity)
    svg = Svg().setup((-0.6, len(entries) - 0.4), _limits(spread))
    svg.axes(
        f"{label or metric} at {lag:g} ns — 95% domain cluster bootstrap",
        label or metric,
        xticks=[(i, a.split("_")[0]) for i, (a, *_) in enumerate(entries)],
    )
    if identity is not None and math.isfinite(identity):
        y = svg.py(identity)
        svg.line(svg.x0, y, svg.x1, y, colour="#cc0000", dash="5,4")
        svg.text(svg.x1 - 4, y - 5, "identity baseline", size=10, anchor="end",
                 colour="#cc0000")
    for i, (arm, mean, lo, hi) in enumerate(entries):
        colour = _COLOURS[7] if CANONICAL_ARMS[arm].oracle else _COLOURS[0]
        x = svg.px(i)
        if math.isfinite(lo):
            svg.line(x, svg.py(lo), x, svg.py(hi), colour=colour, width=1.6)
            svg.line(x - 4, svg.py(lo), x + 4, svg.py(lo), colour=colour)
            svg.line(x - 4, svg.py(hi), x + 4, svg.py(hi), colour=colour)
        svg.circle(x, svg.py(mean), 4, colour)
    svg.legend([("arm", _COLOURS[0]), ("oracle", _COLOURS[7])])
    return svg.save(os.path.join(directory, f"p16ext_{metric}_lag{lag:g}ns.svg"))


def _pareto_plot(by_arm, x_metric, y_metric, lag, directory) -> str | None:
    points = []
    for arm in arm_order(by_arm):
        x_values = values_of(by_arm[arm], x_metric, lag=lag)
        y_values = values_of(by_arm[arm], y_metric, lag=lag)
        if x_values and y_values:
            points.append((
                arm,
                float(np.mean(list(x_values.values()))),
                float(np.mean(list(y_values.values()))),
            ))
    if len(points) < 2:
        return None
    svg = Svg(width=700, height=420, margin=(50, 30, 60, 80)).setup(
        _limits([p[1] for p in points]), _limits([p[2] for p in points])
    )
    svg.axes(f"{x_metric} vs {y_metric} at {lag:g} ns (both lower is better)", y_metric)
    # `Svg.axes` only labels the y axis numerically; a scatter needs the x axis
    # labelled too, so its ticks are drawn here rather than by adding a mode to
    # the shared canvas that every other plot would have to ignore.
    lo, hi = svg.xlim
    for i in range(6):
        value = lo + (hi - lo) * i / 5
        svg.text(svg.px(value), svg.y1 + 16, f"{value:.3g}", size=10)
    svg.text((svg.x0 + svg.x1) / 2, svg.height - 8, x_metric, size=11)
    for i, (arm, x, y) in enumerate(points):
        colour = _COLOURS[7] if CANONICAL_ARMS[arm].oracle else _COLOURS[i % 7]
        svg.circle(svg.px(x), svg.py(y), 5, colour)
        svg.text(svg.px(x), svg.py(y) - 9, arm.split("_")[0], size=10, colour=colour)
    return svg.save(
        os.path.join(directory, f"p16ext_pareto_{x_metric}_{y_metric}_lag{lag:g}ns.svg")
    )


def _paired_domain_delta_plot(by_arm, metric, lag, control, candidate, directory):
    if control not in by_arm or candidate not in by_arm:
        return None
    a = values_of(by_arm[control], metric, lag=lag)
    b = values_of(by_arm[candidate], metric, lag=lag)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    higher = metric in HIGHER_IS_BETTER
    per_domain: dict[str, list[float]] = defaultdict(list)
    for key in shared:
        delta = (b[key] - a[key]) if higher else (a[key] - b[key])
        per_domain[key[1]].append(delta)
    means = sorted((float(np.mean(v)), d) for d, v in per_domain.items())

    svg = Svg(width=760, height=380).setup(
        (-0.6, len(means) - 0.4), _limits([m for m, _ in means] + [0.0])
    )
    svg.axes(
        f"{candidate.split('_')[0]} − {control.split('_')[0]}, {metric} @ {lag:g} ns "
        "— per domain (positive = better)",
        f"Δ {metric}",
        xticks=[(i, d) for i, (_, d) in enumerate(means)],
    )
    zero = svg.py(0.0)
    svg.line(svg.x0, zero, svg.x1, zero, colour="#999", dash="4,4")
    for i, (value, _) in enumerate(means):
        colour = _COLOURS[2] if value > 0 else _COLOURS[1]
        svg.line(svg.px(i), zero, svg.px(i), svg.py(value), colour=colour, width=3)
    svg.legend([("better", _COLOURS[2]), ("worse", _COLOURS[1])])
    return svg.save(os.path.join(
        directory,
        f"p16ext_domain_delta_{candidate.split('_')[0]}_vs_"
        f"{control.split('_')[0]}_{metric}_lag{lag:g}ns.svg",
    ))


def write_plots(by_arm: dict, directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    written = []
    plotted = [
        *CO_PRIMARY, *KEY_SECONDARY,
        "bond_length_rmse", "ca_neighbor_distance_mae", "clash_rate",
    ]
    for metric in plotted:
        for lag in LAGS:
            path = _error_bar_plot(by_arm, metric, lag, directory)
            if path:
                written.append(path)
    for lag in LAGS:
        for x_metric, y_metric in (
            ("ca_rmsd", "rotation_geodesic_mean_deg"),
            ("ca_rmsd", "drmsd_long_range"),
        ):
            path = _pareto_plot(by_arm, x_metric, y_metric, lag, directory)
            if path:
                written.append(path)
        for metric in CO_PRIMARY:
            path = _paired_domain_delta_plot(
                by_arm, metric, lag,
                "P0_pair_geometry_control", "P1_pair_physics_frozen", directory,
            )
            if path:
                written.append(path)
    for lag in LAGS:
        path = _temperature_plot(by_arm, "ca_rmsd", lag, directory)
        if path:
            written.append(path)
    return written


def _temperature_plot(by_arm, metric, lag, directory):
    temperatures = sorted(
        {r["temperature"] for records in by_arm.values() for r in records},
        key=float,
    )
    series = {}
    for arm in arm_order(by_arm) + ([IDENTITY] if IDENTITY in by_arm else []):
        points = []
        for temperature in temperatures:
            values = values_of(by_arm[arm], metric, lag=lag, temperature=temperature)
            if values:
                points.append(float(np.mean(list(values.values()))))
        if len(points) == len(temperatures):
            series[arm] = points
    if not series:
        return None
    spread = [v for points in series.values() for v in points]
    svg = Svg().setup((-0.3, len(temperatures) - 0.7), _limits(spread))
    svg.axes(
        f"{metric} by temperature at {lag:g} ns",
        metric,
        xticks=[(i, f"{t} K") for i, t in enumerate(temperatures)],
    )
    for k, (arm, points) in enumerate(sorted(series.items())):
        colour = _COLOURS[k % len(_COLOURS)]
        svg.polyline([(svg.px(i), svg.py(v)) for i, v in enumerate(points)], colour)
        for i, v in enumerate(points):
            svg.circle(svg.px(i), svg.py(v), 3, colour)
    svg.legend([(a.split("_")[0], _COLOURS[k % len(_COLOURS)])
                for k, a in enumerate(sorted(series))])
    return svg.save(os.path.join(directory, f"p16ext_{metric}_by_temperature_lag{lag:g}ns.svg"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--manifest", default=None,
                        help="reproducibility_manifest.json; defaults to the sibling")
    parser.add_argument("--out", default=None)
    parser.add_argument("--plots", default=None)
    parser.add_argument("--csv-dir", default=None,
                        help="where domain_summary.csv / paired_deltas.csv go; "
                             "defaults to the records' directory")
    args = parser.parse_args()

    by_arm = load_records(args.records)
    manifest_path = args.manifest or str(
        Path(args.records).with_name("reproducibility_manifest.json")
    )
    manifest = (
        json.loads(Path(manifest_path).read_text())
        if os.path.exists(manifest_path) else {}
    )
    scored = {a: r for a, r in by_arm.items() if a != IDENTITY}
    problems = check_same_samples(scored)

    reported = [*CO_PRIMARY, *KEY_SECONDARY, *PAIR_SUBSETS, *TORSIONS, *VALIDITY,
                "ca_rmsd_superposed", "rotation_geodesic_superposed_mean_deg",
                "rotation_geodesic_median_deg", "translation_rmse",
                "contact_precision", "contact_recall", "contact_jaccard",
                "formed_contact_precision", "formed_contact_recall",
                "broken_contact_precision", "broken_contact_recall",
                "contact_f1_legacy_sep3", "contact_f1_cut6", "contact_f1_cut10",
                "bond_length_mae"]
    reported = list(dict.fromkeys(reported))

    csv_dir = Path(args.csv_dir or Path(args.records).parent)
    csv_dir.mkdir(parents=True, exist_ok=True)
    domain_rows = domain_summary_rows(by_arm, reported)
    write_csv(
        csv_dir / "domain_summary.csv", domain_rows,
        ["arm", "lag_ns", "domain_id", "n_samples", "n_valid_residues_mean"]
        + [c for m in reported for c in (m, f"{m}_n")],
    )
    delta_rows = paired_delta_rows(scored, reported)
    write_csv(
        csv_dir / "paired_deltas.csv", delta_rows,
        ["comparison", "control", "candidate", "metric", "lag_ns",
         "higher_is_better", "delta_positive_means_candidate_better",
         "ci_lo", "ci_hi", "relative", "n_shared", "n_domains",
         "separated_at_95", "bootstrap_iterations", "bootstrap_seed"],
    )

    lines = _report(by_arm, scored, problems, manifest, args.plots)
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"report -> {args.out}")
    else:
        print(text)
    print(f"csv -> {csv_dir / 'domain_summary.csv'}, {csv_dir / 'paired_deltas.csv'}")
    return 0


def _report(by_arm, scored, problems, manifest, plots_dir) -> list[str]:
    """Assemble the Markdown, in the order the brief fixes (section 18)."""
    from report_phase1_6_extended import build_report  # noqa: PLC0415

    return build_report(
        by_arm=by_arm, scored=scored, problems=problems, manifest=manifest,
        plots_dir=plots_dir,
        helpers={
            "per_arm_table": per_arm_table,
            "comparison_table": comparison_table,
            "stratified_table": stratified_table,
            "contact_table": contact_table,
            "values_of": values_of,
            "cluster_bootstrap": cluster_bootstrap,
            "paired_delta": paired_delta,
            "residue_weighted": residue_weighted,
            "pooled_torsion": pooled_torsion,
            "pooled_contact": pooled_contact,
            "domain_macro": domain_macro,
            "write_plots": write_plots,
            "cell": cell,
            "arm_order": arm_order,
        },
    )


if __name__ == "__main__":
    sys.exit(main())
