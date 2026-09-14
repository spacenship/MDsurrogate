#!/usr/bin/env python
"""Statistics and report for the Phase 1.6 pair-interaction ablation.

    python scripts/analyze_phase1_6.py --runs runs/phase1_6_bounded_seed0 \
        --out docs/phase1_6_results_bounded.md

    python scripts/analyze_phase1_6.py --runs runs/phase1_6_full_seed{0,1,2} \
        --out docs/phase1_6_report.md --plots docs/figures

**Frames are not independent samples.** Two pairs from the same domain share a
fold, a length and a flexibility; treating 72,000 of them as 72,000 independent
observations produces confidence intervals several times too narrow, and a
frame-level t-test will call a 0.05% difference significant. So:

* comparisons between arms are **paired on the evaluation pair id** -- the same
  domain, temperature, replica, frame and lag, evaluated by both arms;
* intervals come from a **cluster bootstrap over domains**, resampling whole
  domains with replacement, which is the unit the split was made on;
* across seeds the report quotes mean +- sd of three, never a pooled t-test.

Nothing is computed that was not measured. An arm with no run directory is
reported as ``pending``, not imputed, and every derived quantity whose
denominator is degenerate is reported as ``not identifiable`` rather than as a
large number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_md.transition.arms import CANONICAL_ARMS, canonical_name  # noqa: E402

#: Lower is better for all of these.
PRIMARY = ("ca_rmsd", "rotation_geodesic_deg")
SECONDARY = (
    "translation_rmse", "pair_distance_mae", "clash_rate", "phi_mae_deg", "psi_mae_deg",
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_run(run_dir: str) -> dict:
    """One run directory -> ``{canonical_arm: {"records": [...], "provenance": {}}}``."""
    out: dict[str, dict] = {}
    for entry in sorted(os.listdir(run_dir)):
        arm_dir = os.path.join(run_dir, entry)
        records_path = os.path.join(arm_dir, "val_records.json")
        provenance_path = os.path.join(arm_dir, "provenance.json")
        if not (os.path.isdir(arm_dir) and os.path.exists(records_path)):
            continue
        provenance = (
            json.loads(Path(provenance_path).read_text())
            if os.path.exists(provenance_path) else {}
        )
        canonical = entry if entry in CANONICAL_ARMS else canonical_name(
            provenance.get("arm", entry)
        )
        out[canonical] = {
            "records": json.loads(Path(records_path).read_text()),
            "provenance": provenance,
            "dir": arm_dir,
        }
    return out


def key_of(record: dict) -> tuple:
    """Identity of one evaluation sample, for pairing arms row-for-row.

    ``pair_id`` is ``domain/temperature/replica/frame@lag`` and is **unique**;
    ``domain`` is carried alongside it because the bootstrap clusters on it.

    Requiring the id rather than falling back to ``(domain, temperature, lag)``
    is not pedantry. That triple is not unique -- one trajectory contributes many
    frames and one domain many replicas -- so a dictionary keyed on it silently
    keeps the last row of each group. Measured on the Stage B records: 1,760
    pairs per lag collapsed to 150, and the identity baseline moved from 3.058 to
    3.619 A. Every downstream number was then computed on a sixteenth of the data
    with no indication that anything had happened.
    """
    if "pair_id" not in record:
        raise KeyError(
            "records carry no 'pair_id'. They were written before the id was "
            "recorded, and pairing on (domain, temperature, lag) is not unique: "
            "replicas and frames of one trajectory would silently collapse onto "
            "one row. Re-evaluate the checkpoints with the current code."
        )
    return (record["pair_id"], record["domain"])


def check_same_samples(runs: dict[str, dict]) -> list[str]:
    """Every arm must have evaluated the identical sample ids. Returns problems."""
    problems = []
    reference_arm, reference = None, None
    for arm, payload in runs.items():
        keys = sorted(key_of(r) for r in payload["records"])
        if reference is None:
            reference_arm, reference = arm, keys
            continue
        if keys != reference:
            missing = len(set(reference) - set(keys))
            extra = len(set(keys) - set(reference))
            problems.append(
                f"{arm} evaluated {len(keys)} samples against {reference_arm}'s "
                f"{len(reference)} ({missing} missing, {extra} extra)"
            )
    return problems


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def by_lag(records: list[dict], metric: str) -> dict[float, dict[tuple, float]]:
    """``{lag_ns: {sample_key: value}}``, dropping non-finite rows."""
    out: dict[float, dict[tuple, float]] = defaultdict(dict)
    for record in records:
        value = record.get(metric)
        if value is None or not math.isfinite(value):
            continue
        out[record["lag_ns"]][key_of(record)] = float(value)
    return out


def cluster_bootstrap(
    values: dict[tuple, float], iterations: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and a 95% interval, resampling **domains** with replacement.

    The domain is the cluster because the split was made by domain: two pairs
    from the same protein are not two independent looks at the question.

    Returns ``(mean, lo, hi)``; ``(nan, nan, nan)`` for an empty input.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for (_pair_id, domain), value in values.items():
        groups[domain].append(value)
    domains = list(groups)
    arrays = [np.asarray(groups[d], dtype=float) for d in domains]
    rng = np.random.default_rng(seed)

    observed = float(np.mean(np.concatenate(arrays)))
    if len(domains) < 2:
        return observed, float("nan"), float("nan")
    means = np.empty(iterations)
    for i in range(iterations):
        picks = rng.integers(0, len(domains), len(domains))
        means[i] = float(np.mean(np.concatenate([arrays[p] for p in picks])))
    return observed, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(
    a: dict[tuple, float], b: dict[tuple, float], iterations: int = 2000, seed: int = 0
) -> dict:
    """``mean(a - b)`` over shared samples, with a domain cluster bootstrap.

    Positive means ``b`` is better, since every metric here is lower-is-better.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "relative": float("nan")}
    differences = {k: a[k] - b[k] for k in shared}
    delta, lo, hi = cluster_bootstrap(differences, iterations=iterations, seed=seed)
    base = float(np.mean([a[k] for k in shared]))
    return {
        "n": len(shared),
        "domains": len({k[1] for k in shared}),
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "relative": delta / base if base else float("nan"),
        "significant": bool(lo > 0 or hi < 0) if math.isfinite(lo) else False,
    }


def recoverability(baseline: float, best: float, oracle: float) -> str:
    """How much of the oracle's improvement the best predicted arm recovers.

    ``(M(S0) - M(best)) / (M(S0) - M(O))``. The denominator is the whole point:
    when the oracle barely beats the baseline it is a ratio of two noise terms,
    and any number it produces is an artefact. Phase 1.5 measured exactly that
    situation, so the degenerate case is the expected one and is named rather
    than printed.
    """
    denominator = baseline - oracle
    if not math.isfinite(denominator) or denominator <= 1e-9:
        return "not identifiable"
    return f"{(baseline - best) / denominator:.3f}"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def format_ci(mean: float, lo: float, hi: float, digits: int = 4) -> str:
    if not math.isfinite(mean):
        return "—"
    if not math.isfinite(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def arm_table(runs: dict[str, dict], metric: str, lag: float, seed_label: str) -> list[str]:
    lines = [
        f"| arm | {metric} @ {lag:g} ns (95% CI, domain cluster bootstrap) | "
        f"identity | vs identity | pairs | domains |",
        "|---|---|---|---|---|---|",
    ]
    for arm in CANONICAL_ARMS:
        if arm not in runs:
            continue
        records = runs[arm]["records"]
        values = by_lag(records, metric).get(lag, {})
        identity = by_lag(records, f"{metric}_identity").get(lag, {})
        mean, lo, hi = cluster_bootstrap(values)
        base = float(np.mean(list(identity.values()))) if identity else float("nan")
        relative = (mean / base - 1.0) * 100 if base and math.isfinite(base) else float("nan")
        flag = " **[oracle]**" if CANONICAL_ARMS[arm].oracle else ""
        lines.append(
            f"| `{arm}`{flag} | {format_ci(mean, lo, hi)} | {base:.4f} | "
            f"{relative:+.2f}% | {len(values)} | {len({k[1] for k in values})} |"
        )
    for arm in CANONICAL_ARMS:
        if arm not in runs and CANONICAL_ARMS[arm].stage != "gated":
            lines.append(f"| `{arm}` | **pending — not run** | — | — | — | — |")
    return lines


def comparison_table(runs: dict[str, dict], metric: str, lag: float) -> list[str]:
    comparisons = [
        ("P0_pair_geometry_control", "P1_pair_physics_frozen", "Δ_pair-vs-geom (P1)"),
        ("P0_pair_geometry_control", "P2_pair_physics_moments", "Δ_pair-vs-geom (P2)"),
        ("S2_node_physics_152d", "P1_pair_physics_frozen", "Δ_pair-vs-node (P1)"),
        ("S2_node_physics_152d", "P2_pair_physics_moments", "Δ_pair-vs-node (P2)"),
        ("P1_pair_physics_frozen", "P2_pair_physics_moments", "Δ moments (P2 − P1)"),
        ("S0_structure_history", "O_current_gt_force_oracle", "Δ_oracle"),
        ("S0_structure_history", "S2_node_physics_152d", "Δ node physics"),
        ("P2_pair_physics_moments", "P3_pair_physics_uncertainty", "Δ uncertainty gate"),
        ("P2_pair_physics_moments", "P4_future_physics_consistency", "Δ future consistency"),
    ]
    lines = [
        f"| comparison | mean Δ @ {lag:g} ns | 95% CI | relative | shared pairs | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for left, right, label in comparisons:
        if left not in runs or right not in runs:
            lines.append(f"| {label} | pending | — | — | — | not run |")
            continue
        a = by_lag(runs[left]["records"], metric).get(lag, {})
        b = by_lag(runs[right]["records"], metric).get(lag, {})
        result = paired_delta(a, b)
        if not result["n"]:
            lines.append(f"| {label} | — | — | — | 0 | no shared samples |")
            continue
        verdict = (
            "improves" if result["significant"] and result["delta"] > 0
            else "worse" if result["significant"] and result["delta"] < 0
            else "not separated"
        )
        lines.append(
            f"| {label} | {result['delta']:+.4f} | "
            f"[{result['lo']:+.4f}, {result['hi']:+.4f}] | "
            f"{result['relative'] * 100:+.2f}% | {result['n']} | {verdict} |"
        )
    return lines


def temperature_table(runs: dict[str, dict], metric: str, lag: float) -> list[str]:
    temperatures = sorted({
        r.get("temperature") for payload in runs.values() for r in payload["records"]
        if r.get("temperature") is not None
    }, key=lambda t: float(t) if t is not None else 0.0)
    if not temperatures:
        return ["Temperature was not recorded in these runs — no stratification possible."]
    lines = ["| arm | " + " | ".join(f"{t} K" for t in temperatures) + " |",
             "|---" * (len(temperatures) + 1) + "|"]
    for arm in CANONICAL_ARMS:
        if arm not in runs:
            continue
        cells = []
        for temperature in temperatures:
            values = {
                key_of(r): float(r[metric])
                for r in runs[arm]["records"]
                if r.get("temperature") == temperature and r["lag_ns"] == lag
                and r.get(metric) is not None and math.isfinite(r[metric])
            }
            cells.append(f"{np.mean(list(values.values())):.4f}" if values else "—")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return lines


def resource_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| arm | total params | trainable | conditioner | peak GPU (MiB) | wall (s) | "
        "pair features | oracle |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, payload in runs.items():
        p = payload["provenance"]
        if not p:
            continue
        resources = p.get("resources", {})
        peak = resources.get("peak_gpu_memory_bytes")
        lines.append(
            f"| `{arm}` | {p.get('parameter_count', 0):,} | "
            f"{p.get('trainable_parameter_count', 0):,} | "
            f"{p.get('parameter_breakdown', {}).get('conditioner', 0):,} | "
            f"{'—' if peak is None else round(peak / 2**20)} | "
            f"{resources.get('train_wall_time_s', '—')} | "
            f"{p.get('uses_pair_features')} | {p.get('oracle')} |"
        )
    return lines


def hash_table(runs: dict[str, dict]) -> list[str]:
    lines = ["| arm | manifest | Phase 1 ckpt | config | git | dirty |",
             "|---|---|---|---|---|---|"]
    for arm, payload in runs.items():
        p = payload["provenance"]
        if not p:
            continue
        git = p.get("git", {})
        lines.append(
            f"| `{arm}` | `{str(p.get('manifest_hash'))[:12]}` | "
            f"`{str(p.get('phase1_sha256'))[:12]}` | "
            f"`{str(p.get('config_hash'))[:12]}` | "
            f"`{str(git.get('commit'))[:12]}` | {git.get('dirty')} |"
        )
    return lines


def seed_summary(all_runs: list[dict], metric: str, lag: float) -> list[str]:
    """Mean +- sd over seeds. With one seed it says so instead of quoting sd 0."""
    if len(all_runs) < 2:
        return [
            f"Only {len(all_runs)} seed available: a standard deviation over seeds "
            "is not defined and none is quoted. Stage C needs three."
        ]
    lines = [f"| arm | {metric} @ {lag:g} ns, mean ± sd over {len(all_runs)} seeds |",
             "|---|---|"]
    for arm in CANONICAL_ARMS:
        values = []
        for runs in all_runs:
            if arm not in runs:
                continue
            per_lag = by_lag(runs[arm]["records"], metric).get(lag, {})
            if per_lag:
                values.append(float(np.mean(list(per_lag.values()))))
        if len(values) < 2:
            continue
        lines.append(
            f"| `{arm}` | {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} |"
        )
    return lines


#: Plot palette. Colour-blind-safe and readable in greyscale, because a figure in
#: a report is read by whoever opens it, on whatever they open it with.
_COLOURS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00",
            "#7f7f7f", "#111111")


class Svg:
    """A minimal SVG canvas: axes, points, error bars, lines, labels.

    Written out rather than reached for, because this repository keeps its
    dependency set deliberately small -- ``matplotlib`` is not in
    ``requirements.txt`` and is not installed in the ``md`` env, and a report that
    silently ships without its figures because an import failed is worse than one
    whose figures are plain. SVG needs nothing, renders in any browser and in
    GitHub markdown, and diffs as text.
    """

    def __init__(self, width: int = 760, height: int = 380, *, margin=(60, 20, 90, 70)):
        self.width, self.height = width, height
        self.top, self.right, self.bottom, self.left = margin
        self.parts: list[str] = []
        self.x0, self.x1 = self.left, width - self.right
        self.y0, self.y1 = self.top, height - self.bottom

    # -- coordinates ------------------------------------------------------
    def setup(self, xlim, ylim):
        self.xlim, self.ylim = xlim, ylim
        return self

    def px(self, x: float) -> float:
        lo, hi = self.xlim
        span = (hi - lo) or 1.0
        return self.x0 + (x - lo) / span * (self.x1 - self.x0)

    def py(self, y: float) -> float:
        lo, hi = self.ylim
        span = (hi - lo) or 1.0
        return self.y1 - (y - lo) / span * (self.y1 - self.y0)

    # -- primitives -------------------------------------------------------
    def text(self, x, y, s, *, size=11, anchor="middle", colour="#222", rotate=None):
        transform = f' transform="rotate({rotate},{x},{y})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif" '
            f'text-anchor="{anchor}"{transform}>{s}</text>'
        )

    def line(self, x1, y1, x2, y2, *, colour="#444", width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}" stroke-width="{width}"{d}/>'
        )

    def circle(self, x, y, r=4, colour="#0072b2"):
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}"/>'
        )

    def polyline(self, points, colour="#0072b2", width=1.8):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}"/>'
        )

    def axes(self, title, ylabel, xticks=None, yticks=5):
        self.parts.append(
            f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>'
        )
        self.line(self.x0, self.y1, self.x1, self.y1, colour="#888")
        self.line(self.x0, self.y0, self.x0, self.y1, colour="#888")
        lo, hi = self.ylim
        for i in range(yticks + 1):
            value = lo + (hi - lo) * i / yticks
            y = self.py(value)
            self.line(self.x0, y, self.x1, y, colour="#eeeeee")
            self.text(self.x0 - 8, y + 4, f"{value:.3g}", size=10, anchor="end")
        for x, label in (xticks or []):
            self.text(self.px(x), self.y1 + 14, label, size=10, anchor="end",
                      rotate=-35)
        self.text((self.x0 + self.x1) / 2, self.top - 6, title, size=13)
        self.text(16, (self.y0 + self.y1) / 2, ylabel, size=11, rotate=-90)

    def legend(self, entries):
        for i, (label, colour) in enumerate(entries):
            y = self.y0 + 14 * i + 4
            self.line(self.x1 - 120, y, self.x1 - 100, y, colour=colour, width=3)
            self.text(self.x1 - 96, y + 4, label, size=10, anchor="start")

    def save(self, path: str) -> str:
        body = "\n".join(self.parts)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
            f"\n{body}\n</svg>\n"
        )
        return path


def _limits(values, pad=0.12):
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - span * pad, hi + span * pad


def write_plots(all_runs: list[dict], directory: str) -> list[str]:
    """Error-bar, oracle-gap and learning-curve figures, as dependency-free SVG."""
    os.makedirs(directory, exist_ok=True)
    runs = all_runs[0]
    written: list[str] = []

    # 1. per-metric, per-lag error bars against the identity baseline
    for metric in PRIMARY:
        for lag in (1.0, 4.0):
            entries, identity = [], []
            for arm in CANONICAL_ARMS:
                if arm not in runs:
                    continue
                values = by_lag(runs[arm]["records"], metric).get(lag, {})
                if not values:
                    continue
                mean, lo, hi = cluster_bootstrap(values, iterations=500)
                entries.append((arm, mean, lo, hi))
                base = by_lag(runs[arm]["records"], f"{metric}_identity").get(lag, {})
                if base:
                    identity.append(float(np.mean(list(base.values()))))
            if not entries:
                continue
            spread = [v for _, m, lo, hi in entries for v in (m, lo, hi)
                      if math.isfinite(v)] + identity
            svg = Svg().setup((-0.6, len(entries) - 0.4), _limits(spread))
            svg.axes(
                f"{metric} at {lag:g} ns — 95% domain cluster bootstrap",
                metric,
                xticks=[(i, a.split("_")[0]) for i, (a, *_ ) in enumerate(entries)],
            )
            if identity:
                y = svg.py(float(np.mean(identity)))
                svg.line(svg.x0, y, svg.x1, y, colour="#cc0000", dash="5,4")
                svg.text(svg.x1 - 4, y - 5, "identity baseline", size=10,
                         anchor="end", colour="#cc0000")
            for i, (arm, mean, lo, hi) in enumerate(entries):
                colour = _COLOURS[7] if CANONICAL_ARMS[arm].oracle else _COLOURS[0]
                x = svg.px(i)
                if math.isfinite(lo):
                    svg.line(x, svg.py(lo), x, svg.py(hi), colour=colour, width=1.6)
                    svg.line(x - 4, svg.py(lo), x + 4, svg.py(lo), colour=colour)
                    svg.line(x - 4, svg.py(hi), x + 4, svg.py(hi), colour=colour)
                svg.circle(x, svg.py(mean), 4, colour)
            svg.legend([("arm", _COLOURS[0]), ("oracle", _COLOURS[7])])
            written.append(svg.save(
                os.path.join(directory, f"phase1_6_{metric}_lag{lag:g}ns.svg")
            ))

    # 2. oracle gap: improvement over S0, per arm, both lags
    if "S0_structure_history" in runs:
        series = {}
        for lag in (1.0, 4.0):
            base = by_lag(runs["S0_structure_history"]["records"], "ca_rmsd").get(lag, {})
            if not base:
                continue
            points = []
            for arm in CANONICAL_ARMS:
                if arm not in runs or arm == "S0_structure_history":
                    continue
                values = by_lag(runs[arm]["records"], "ca_rmsd").get(lag, {})
                result = paired_delta(base, values, iterations=200)
                if result["n"]:
                    points.append((arm, result["delta"]))
            if points:
                series[lag] = points
        if series:
            names = [a for a, _ in next(iter(series.values()))]
            spread = [d for points in series.values() for _, d in points] + [0.0]
            svg = Svg().setup((-0.4, len(names) - 0.6), _limits(spread))
            svg.axes(
                "Oracle gap — Cα RMSD improvement over S0 (positive is better)",
                "Δ Cα RMSD (Å)",
                xticks=[(i, n.split("_")[0]) for i, n in enumerate(names)],
            )
            zero = svg.py(0.0)
            svg.line(svg.x0, zero, svg.x1, zero, colour="#999", dash="4,4")
            for k, (lag, points) in enumerate(series.items()):
                colour = _COLOURS[k]
                coords = [(svg.px(i), svg.py(d)) for i, (_, d) in enumerate(points)]
                svg.polyline(coords, colour)
                for x, y in coords:
                    svg.circle(x, y, 4, colour)
            svg.legend([(f"{lag:g} ns", _COLOURS[k])
                        for k, lag in enumerate(series)])
            written.append(svg.save(
                os.path.join(directory, "phase1_6_oracle_gap.svg")
            ))

    # 3. learning curves
    curves = {}
    for arm, payload in runs.items():
        history_path = os.path.join(payload["dir"], "history.json")
        if not os.path.exists(history_path):
            continue
        history = json.loads(Path(history_path).read_text())
        points = [(h["step"], h["loss_total"]) for h in history
                  if h.get("loss_total") is not None and math.isfinite(h["loss_total"])]
        if points:
            curves[arm] = points
    if curves:
        xs = [s for points in curves.values() for s, _ in points]
        ys = [v for points in curves.values() for _, v in points]
        svg = Svg().setup((min(xs), max(xs)), _limits(ys))
        svg.axes("Validation loss", "loss",
                 xticks=[(min(xs), str(min(xs))), (max(xs), str(max(xs)))])
        for k, (arm, points) in enumerate(sorted(curves.items())):
            colour = _COLOURS[k % len(_COLOURS)]
            svg.polyline([(svg.px(s), svg.py(v)) for s, v in points], colour)
        svg.legend([(a.split("_")[0], _COLOURS[k % len(_COLOURS)])
                    for k, a in enumerate(sorted(curves))])
        written.append(svg.save(
            os.path.join(directory, "phase1_6_learning_curves.svg")
        ))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--plots", default=None)
    parser.add_argument("--stage", default="Stage B (bounded screening)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    all_runs = [load_run(directory) for directory in args.runs]
    present = [r for r in all_runs if r]
    if not present:
        raise SystemExit(f"no arm results found under {args.runs}")

    lines: list[str] = [
        f"# Phase 1.6 results — {args.stage}",
        "",
        f"Generated by `scripts/analyze_phase1_6.py` from "
        + ", ".join(f"`{d}`" for d in args.runs) + ".",
        "",
        "Lower is better for every metric here. Intervals are 95% cluster "
        "bootstraps over **domains**, not over frames.",
        "",
    ]

    problems = check_same_samples(present[0])
    lines += ["## Sample-identity check", ""]
    lines += (
        ["Every arm evaluated the identical set of sample ids "
         f"({len(next(iter(present[0].values()))['records'])} pairs)."]
        if not problems else
        ["**Arms did not evaluate the same samples. The comparisons below are "
         "invalid:**", ""] + [f"* {p}" for p in problems]
    )
    lines.append("")

    for lag in (1.0, 4.0):
        lines += [f"## {lag:g} ns", "", "### Per-arm Cα RMSD", ""]
        lines += arm_table(present[0], "ca_rmsd", lag, "seed 0")
        lines += ["", "### Per-arm rotation geodesic error (deg)", ""]
        lines += arm_table(present[0], "rotation_geodesic_deg", lag, "seed 0")
        lines += ["", "### Paired comparisons — Cα RMSD", ""]
        lines += comparison_table(present[0], "ca_rmsd", lag)
        lines += ["", "### Paired comparisons — rotation", ""]
        lines += comparison_table(present[0], "rotation_geodesic_deg", lag)
        lines += ["", "### By temperature (Cα RMSD)", ""]
        lines += temperature_table(present[0], "ca_rmsd", lag)
        lines.append("")

        # recoverability
        needed = ("S0_structure_history", "O_current_gt_force_oracle")
        if all(a in present[0] for a in needed):
            base_values = by_lag(present[0]["S0_structure_history"]["records"],
                                 "ca_rmsd").get(lag, {})
            oracle_values = by_lag(present[0]["O_current_gt_force_oracle"]["records"],
                                   "ca_rmsd").get(lag, {})
            # A legacy Phase 1.5 arm can appear in a reused run directory and has
            # no canonical entry; skip it rather than raising on the lookup.
            candidates = {
                arm: float(np.mean(list(by_lag(present[0][arm]["records"], "ca_rmsd")
                                        .get(lag, {}).values())))
                for arm in present[0]
                if arm in CANONICAL_ARMS and not CANONICAL_ARMS[arm].oracle
                and by_lag(present[0][arm]["records"], "ca_rmsd").get(lag)
            }
            best_arm = min(candidates, key=candidates.get)
            lines += [
                "### Recoverability", "",
                f"Best predicted arm at this lag: `{best_arm}`.", "",
                "```",
                f"M(S0) = {np.mean(list(base_values.values())):.4f}",
                f"M(O)  = {np.mean(list(oracle_values.values())):.4f}",
                f"M(best) = {candidates[best_arm]:.4f}",
                "recoverability = " + recoverability(
                    float(np.mean(list(base_values.values()))),
                    candidates[best_arm],
                    float(np.mean(list(oracle_values.values()))),
                ),
                "```",
                "",
            ]

    lines += ["## Across seeds", ""]
    lines += seed_summary(present, "ca_rmsd", 1.0)
    lines += ["", "## Resources and capacity", ""]
    lines += resource_table(present[0])
    lines += ["", "## Reproducibility hashes", ""]
    lines += hash_table(present[0])
    lines.append("")

    if args.plots:
        lines += ["## Figures", ""]
        lines += [f"* `{p}`" for p in write_plots(present, args.plots)]
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"report -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
