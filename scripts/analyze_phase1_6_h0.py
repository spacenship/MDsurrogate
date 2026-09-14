#!/usr/bin/env python
"""H0 -- aggregate and report the heavy-atom backmapping of the frozen arms.

    python scripts/analyze_phase1_6_h0.py \
        --records runs/phase1_6_h0_seed0/records.jsonl \
        --out docs/phase1_6_h0_results.md

Stage M measured the Phase 1.6 arms at the resolution they were trained on: one
rigid frame per residue. H0 asks the question that resolution cannot answer --
**when an arm moves a frame, what happens to the atoms hanging off it?** -- by
placing real heavy atoms on the frozen predictions and scoring them. Nothing is
trained here and no checkpoint is touched; the arms are exactly the ones Stage M
scored, on exactly the same 3,520 validation pairs.

The whole design rests on four *construction modes*, which are the columns of
every table below. Two of them are the model's own output and two are oracles
that replace one half of the construction with ground truth:

===============================  ===================  ===================
mode                             frames from          local atoms from
===============================  ===================  ===================
``identity_current_atoms``       (nothing moves)      current
``transported_current_conformer``  **model**          current
``future_frame_current_local``   true future          current
``predicted_frame_future_local``   **model**          true future
===============================  ===================  ===================

Reading down that table gives the decomposition H0 exists for. The gap between
``transported_current_conformer`` and ``future_frame_current_local`` is the part
of the heavy-atom error caused by **getting the frame wrong**; the gap to
``predicted_frame_future_local`` is the part caused by **carrying the current
side-chain conformer forward unchanged**. H1a can only attack the second; H1b
only the first. Which one dominates decides which is worth building.

Three properties of this script are load-bearing.

*Model dependence is measured, not trusted.* Two of the four modes ignore the
model entirely, so every arm must produce bit-identical records for them. That
is checked here across all arms rather than assumed, because a wiring error that
let a prediction leak into the identity baseline would otherwise be invisible --
it would just make the baseline look better. The check also lets the two
model-independent modes be reported once instead of seven times.

*The oracles are labelled as oracles everywhere.* ``future_frame_current_local``
and ``predicted_frame_future_local`` read the future structure. They are
diagnostics that bound what a decoder could achieve; they are not results and
they are never compared with each other as if they were rival methods.

*It does not impute.* An arm with no records is ``pending``; a metric absent from
the schema is ``n/a``, never 0.

Note on the ``is_model_predicted`` provenance flag carried in the records: it
marks a **deployable** prediction, so it is ``False`` for
``predicted_frame_future_local`` even though that mode does consume the model's
frames. This script therefore derives model dependence from the data instead of
from the flag, and reports both.
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
    domain_macro,
    pooled_contact,
    write_csv,
)
from force_md.heavy.backmapping import CONSTRUCTION_MODES  # noqa: E402
from force_md.transition.arms import CANONICAL_ARMS  # noqa: E402

# --------------------------------------------------------------------------
# what H0 reports
# --------------------------------------------------------------------------

#: Placement order for the tables: baseline, the model's own output, then the
#: two oracles. Not alphabetical -- the reading order *is* the argument.
MODE_ORDER = (
    "identity_current_atoms",
    "transported_current_conformer",
    "future_frame_current_local",
    "predicted_frame_future_local",
)

#: The single mode that is a deployable prediction. The other three are a
#: baseline and two future-reading diagnostics.
DEPLOYABLE = "transported_current_conformer"

STRUCTURE = (
    "heavy_atom_rmsd",
    "backbone_heavy_rmsd",
    "sidechain_heavy_rmsd",
    "sidechain_centroid_error",
)

CLASH = (
    "bondi_overlap_rate_serious",
    "bondi_clashes_per_1000_heavy_atoms",
    "bondi_overlap_depth_mean",
    "bondi_overlap_depth_max",
    "bondi_serious_1_4_rate",
)

#: Serious overlap resolved by which atom classes touch, inter- and
#: intra-residue kept apart. An intra-residue overlap in a *transported* mode is
#: inherited from the current conformer, not created by the model; separating
#: them is what makes the inter-residue column attributable.
CLASH_CLASSES = tuple(
    f"bondi_serious_{scope}_residue_{pair}_rate"
    for scope in ("inter", "intra")
    for pair in ("backbone_backbone", "backbone_sidechain", "sidechain_sidechain")
)

CONTACT_PREFIXES = (
    "atom_contact_any",
    "atom_contact_backbone_backbone",
    "atom_contact_backbone_sidechain",
    "atom_contact_sidechain_sidechain",
    "atom_contact_formed",
    "atom_contact_broken",
)
CONTACT_METRICS = tuple(
    f"{prefix}_{suffix}"
    for prefix in CONTACT_PREFIXES
    for suffix in ("f1", "precision", "recall")
)

#: Explicit rather than suffix-inferred, so a metric added later cannot pick up
#: a sign convention by accident of its name.
HIGHER_IS_BETTER = frozenset(CONTACT_METRICS)

ALL_METRICS = STRUCTURE + CLASH + CLASH_CLASSES + CONTACT_METRICS

#: ``(control, candidate, question)``. Every one is *within* an arm, across
#: construction modes -- that is the axis H0 adds.
DECOMPOSITION = (
    (
        "identity_current_atoms",
        "transported_current_conformer",
        "does the predicted frame motion beat 'nothing moves' at atom resolution?",
    ),
    (
        "transported_current_conformer",
        "future_frame_current_local",
        "error attributable to the predicted frames (oracle frames, same conformer)",
    ),
    (
        "transported_current_conformer",
        "predicted_frame_future_local",
        "error attributable to the frozen conformer (same frames, oracle conformer)",
    ),
    (
        "identity_current_atoms",
        "future_frame_current_local",
        "ceiling of a frame-only correction (H1b's target)",
    ),
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_records(path: str) -> dict[tuple[str, str], list[dict]]:
    """``{(canonical_arm, construction_mode): [record, ...]}``.

    A duplicate ``(pair_id, mode)`` inside one arm is refused: the paired
    comparisons key on the pair id, and a silent overwrite there would drop
    samples from one side of a delta without changing any printed count.
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                cells[(record["canonical_arm"], record["construction_mode"])].append(
                    record
                )

    for key, records in cells.items():
        ids = [r["pair_id"] for r in records]
        if len(set(ids)) != len(ids):
            seen, duplicates = set(), []
            for identifier in ids:
                if identifier in seen:
                    duplicates.append(identifier)
                seen.add(identifier)
            raise SystemExit(
                f"{key[0]}/{key[1]}: {len(duplicates)} duplicate pair_id(s), e.g. "
                f"{duplicates[:3]}."
            )
    return dict(cells)


def arms_present(cells: dict) -> list[str]:
    found = {arm for arm, _mode in cells}
    return [arm for arm in CANONICAL_ARMS if arm in found]


def modes_present(cells: dict) -> list[str]:
    found = {mode for _arm, mode in cells}
    extra = sorted(found - set(MODE_ORDER))
    return [m for m in MODE_ORDER if m in found] + extra


def check_same_samples(cells: dict) -> list[str]:
    """Every (arm, mode) cell must have scored the identical pair-id set."""
    problems, reference, reference_key = [], None, None
    for key in sorted(cells):
        ids = frozenset(r["pair_id"] for r in cells[key])
        if reference is None:
            reference, reference_key = ids, key
            continue
        if ids != reference:
            problems.append(
                f"{key[0]}/{key[1]} scored {len(ids)} samples against "
                f"{reference_key[0]}/{reference_key[1]}'s {len(reference)} "
                f"({len(reference - ids)} missing, {len(ids - reference)} extra)"
            )
    return problems


def model_dependence(cells: dict, metrics: tuple[str, ...]) -> dict[str, dict]:
    """Per mode: does any metric change when the arm changes?

    This is the wiring check. ``identity_current_atoms`` and
    ``future_frame_current_local`` are built without ever reading the model, so
    across arms they must agree **exactly** -- not to a tolerance, bitwise, since
    it is literally the same arithmetic on the same inputs. Any drift means a
    prediction reached a construction that is supposed to be model-free, which
    would flatter the baseline and corrupt every delta measured against it.
    """
    report: dict[str, dict] = {}
    for mode in modes_present(cells):
        arms = [a for a in arms_present(cells) if (a, mode) in cells]
        entry = {
            "arms_compared": len(arms),
            "identical": None,
            "max_abs_difference": float("nan"),
            "worst_metric": None,
            "compared_samples": 0,
        }
        if len(arms) >= 2:
            tables = [
                {
                    r["pair_id"]: r
                    for r in cells[(arm, mode)]
                }
                for arm in arms
            ]
            shared = set(tables[0])
            for table in tables[1:]:
                shared &= set(table)
            entry["compared_samples"] = len(shared)
            worst, worst_metric = 0.0, None
            for metric in metrics:
                for table in tables[1:]:
                    for pair_id in shared:
                        a = tables[0][pair_id].get(metric)
                        b = table[pair_id].get(metric)
                        if a is None or b is None:
                            continue
                        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                            continue
                        difference = abs(float(a) - float(b))
                        if difference > worst:
                            worst, worst_metric = difference, metric
            entry["identical"] = worst == 0.0
            entry["max_abs_difference"] = worst
            entry["worst_metric"] = worst_metric
        report[mode] = entry
    return report


def values_of(records: list[dict], metric: str, lag: float) -> dict[tuple, float]:
    """``{(pair_id, domain): value}`` for the finite rows at one lag."""
    out: dict[tuple, float] = {}
    for record in records:
        if record["lag_ns"] != lag:
            continue
        value = record.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        out[(record["pair_id"], record["domain_id"])] = float(value)
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def paired_delta(
    control: dict[tuple, float],
    candidate: dict[tuple, float],
    metric: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired delta over shared samples, **normalised so positive = better**.

    Same convention and the same domain-cluster bootstrap as Stage M, so an H0
    delta and a Stage M delta mean the same thing and can sit in one sentence.
    """
    shared = sorted(set(control) & set(candidate))
    if not shared:
        return {
            "n": 0, "domains": 0, "delta": float("nan"), "lo": float("nan"),
            "hi": float("nan"), "relative": float("nan"), "significant": False,
        }
    higher_better = metric in HIGHER_IS_BETTER
    differences = {
        key: (candidate[key] - control[key]) if higher_better
        else (control[key] - candidate[key])
        for key in shared
    }
    delta, lo, hi = cluster_bootstrap(differences, iterations=iterations, seed=seed)
    base = float(np.mean([control[key] for key in shared]))
    return {
        "n": len(shared),
        "domains": len({key[1] for key in shared}),
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "relative": delta / abs(base) if base else float("nan"),
        "significant": bool(lo > 0 or hi < 0) if math.isfinite(lo) else False,
    }


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _mode_label(mode: str, dependence: dict) -> str:
    if mode == DEPLOYABLE:
        return f"`{mode}`"
    if CONSTRUCTION_MODES.get(mode, {}).get("uses_future_for_scoring_only"):
        return f"`{mode}` *(oracle)*"
    return f"`{mode}` *(baseline)*"


def mode_semantics_table(cells: dict, dependence: dict) -> list[str]:
    """What each construction mode is, and whether the data says it saw a model."""
    lines = [
        "| construction mode | frames | local atoms | reads the future | "
        "varies by arm (measured) |",
        "|---|---|---|---|---|",
    ]
    frames = {
        "identity_current_atoms": "none (unmoved)",
        "transported_current_conformer": "**model**",
        "future_frame_current_local": "true future",
        "predicted_frame_future_local": "**model**",
    }
    locals_ = {
        "identity_current_atoms": "current",
        "transported_current_conformer": "current",
        "future_frame_current_local": "current",
        "predicted_frame_future_local": "true future",
    }
    for mode in modes_present(cells):
        info = CONSTRUCTION_MODES.get(mode, {})
        entry = dependence.get(mode, {})
        if entry.get("identical") is None:
            varies = "not testable (one arm)"
        elif entry["identical"]:
            varies = f"no (bitwise identical over {entry['compared_samples']} pairs)"
        else:
            varies = f"yes (max Δ {entry['max_abs_difference']:.3e})"
        lines.append(
            f"| {_mode_label(mode, dependence)} | {frames.get(mode, '?')} | "
            f"{locals_.get(mode, '?')} | "
            f"{'yes' if info.get('uses_future_for_scoring_only') else 'no'} | "
            f"{varies} |"
        )
    return lines


def per_cell_table(cells: dict, metric: str, lag: float, *, digits: int = 4) -> list[str]:
    """One metric, every (arm, mode) cell, with model-free modes collapsed.

    The two model-free modes are reported once as a shared reference row rather
    than repeated per arm -- but only when the check above proved they really are
    identical. If they are not, they are printed per arm and the discrepancy
    stands in the table where it cannot be missed.
    """
    higher = metric in HIGHER_IS_BETTER
    arrow = "higher better" if higher else "lower better"
    lines = [
        f"**`{metric}`** ({arrow}, lag {lag:g} ns) — mean [95% CI], "
        "domain-cluster bootstrap",
        "",
        "| construction mode | arm | value | n | domains |",
        "|---|---|---|---|---|",
    ]
    dependence = model_dependence(cells, (metric,))
    for mode in modes_present(cells):
        collapse = dependence.get(mode, {}).get("identical") is True
        arms = [a for a in arms_present(cells) if (a, mode) in cells]
        shown = arms[:1] if collapse else arms
        for arm in shown:
            values = values_of(cells[(arm, mode)], metric, lag)
            mean, lo, hi = cluster_bootstrap(values)
            label = "all arms (identical)" if collapse else arm
            lines.append(
                f"| {_mode_label(mode, dependence)} | {label} | "
                f"{format_ci(mean, lo, hi, digits)} | {len(values)} | "
                f"{len({k[1] for k in values})} |"
            )
    return lines


def decomposition_table(cells: dict, metric: str, lag: float, *, digits: int = 4) -> list[str]:
    """Per arm, the four modes side by side for one metric.

    This is the table the stage exists for: read a row and the frame error and
    the conformer error are next to each other, in the same units, for the same
    arm and the same samples.
    """
    higher = metric in HIGHER_IS_BETTER
    modes = [m for m in MODE_ORDER if any((a, m) in cells for a in arms_present(cells))]
    header = " | ".join(m.replace("_", " ") for m in modes)
    lines = [
        f"**`{metric}`** ({'higher' if higher else 'lower'} better, "
        f"lag {lag:g} ns) — mean over the shared samples",
        "",
        f"| arm | {header} |",
        "|---|" + "---|" * len(modes),
    ]
    for arm in arms_present(cells):
        row = [arm]
        for mode in modes:
            records = cells.get((arm, mode))
            if not records:
                row.append("pending")
                continue
            values = values_of(records, metric, lag)
            row.append(cell(float(np.mean(list(values.values()))) if values
                            else float("nan"), digits))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def delta_table(cells: dict, metric: str, lag: float, *, digits: int = 4) -> list[str]:
    """The four decomposition comparisons, per arm. Positive = candidate better."""
    lines = [
        f"**`{metric}`** (lag {lag:g} ns) — paired Δ, **positive = candidate better**",
        "",
        "| comparison | arm | Δ [95% CI] | relative | n | significant |",
        "|---|---|---|---|---|---|",
    ]
    for control_mode, candidate_mode, question in DECOMPOSITION:
        for arm in arms_present(cells):
            control = cells.get((arm, control_mode))
            candidate = cells.get((arm, candidate_mode))
            if not control or not candidate:
                continue
            result = paired_delta(
                values_of(control, metric, lag),
                values_of(candidate, metric, lag),
                metric,
            )
            if not result["n"]:
                continue
            lines.append(
                f"| {candidate_mode} − {control_mode} | {arm} | "
                f"{format_ci(result['delta'], result['lo'], result['hi'], digits)} | "
                f"{result['relative'] * 100:+.1f}% | {result['n']} | "
                f"{'yes' if result['significant'] else 'no'} |"
            )
        lines.append(f"| *{question}* | | | | | |")
    return lines


def contact_table(cells: dict, prefix: str, lag: float) -> list[str]:
    """Sample-macro and pooled-micro contact metrics, side by side, never blended."""
    lines = [
        f"**`{prefix}`** (lag {lag:g} ns)",
        "",
        "| construction mode | arm | macro F1 | micro F1 | micro P | micro R | "
        "TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    dependence = model_dependence(cells, (f"{prefix}_f1",))
    for mode in modes_present(cells):
        collapse = dependence.get(mode, {}).get("identical") is True
        arms = [a for a in arms_present(cells) if (a, mode) in cells]
        for arm in (arms[:1] if collapse else arms):
            records = cells[(arm, mode)]
            macro = values_of(records, f"{prefix}_f1", lag)
            micro = pooled_contact(records, prefix, lag)
            label = "all arms (identical)" if collapse else arm
            lines.append(
                f"| {_mode_label(mode, dependence)} | {label} | "
                f"{cell(domain_macro(macro))} | {cell(micro['f1'])} | "
                f"{cell(micro['precision'])} | {cell(micro['recall'])} | "
                f"{micro['tp']} | {micro['fp']} | {micro['fn']} |"
            )
    return lines


# --------------------------------------------------------------------------
# machine-readable output
# --------------------------------------------------------------------------


def domain_summary_rows(cells: dict, metrics: tuple[str, ...]) -> list[dict]:
    """One row per (arm, mode, lag, domain) -- the bootstrap's clusters, inspectable."""
    rows = []
    for arm in arms_present(cells):
        for mode in modes_present(cells):
            records_all = cells.get((arm, mode))
            if not records_all:
                continue
            for lag in LAGS:
                per_domain: dict[str, list[dict]] = defaultdict(list)
                for record in records_all:
                    if record["lag_ns"] == lag:
                        per_domain[record["domain_id"]].append(record)
                for domain, records in sorted(per_domain.items()):
                    row = {
                        "canonical_arm": arm,
                        "construction_mode": mode,
                        "lag_ns": lag,
                        "domain_id": domain,
                        "n_samples": len(records),
                        "n_valid_atoms_mean": float(
                            np.mean([r["n_valid_atoms"] for r in records])
                        ),
                    }
                    for metric in metrics:
                        finite = [
                            float(r[metric])
                            for r in records
                            if r.get(metric) is not None
                            and math.isfinite(float(r[metric]))
                        ]
                        row[metric] = float(np.mean(finite)) if finite else float("nan")
                        row[f"{metric}_n"] = len(finite)
                    rows.append(row)
    return rows


def decomposition_rows(cells: dict, metrics: tuple[str, ...]) -> list[dict]:
    rows = []
    for lag in LAGS:
        for control_mode, candidate_mode, question in DECOMPOSITION:
            for arm in arms_present(cells):
                control = cells.get((arm, control_mode))
                candidate = cells.get((arm, candidate_mode))
                if not control or not candidate:
                    continue
                for metric in metrics:
                    result = paired_delta(
                        values_of(control, metric, lag),
                        values_of(candidate, metric, lag),
                        metric,
                    )
                    rows.append({
                        "canonical_arm": arm,
                        "control_mode": control_mode,
                        "candidate_mode": candidate_mode,
                        "question": question,
                        "metric": metric,
                        "lag_ns": lag,
                        "higher_is_better": metric in HIGHER_IS_BETTER,
                        **result,
                    })
    return rows


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _integrity_section(cells: dict, problems: list[str], dependence: dict) -> list[str]:
    lines = ["## 1. Integrity checks", ""]
    counts = {f"{arm}/{mode}": len(records) for (arm, mode), records in sorted(cells.items())}
    sizes = sorted(set(counts.values()))
    lines.append(
        f"- {len(cells)} (arm, mode) cells, "
        f"{sum(counts.values())} records, "
        + (f"**{sizes[0]} per cell**." if len(sizes) == 1
           else f"**cell sizes differ: {sizes}** — comparisons below are on shared samples only.")
    )
    if problems:
        lines.append("- **Sample sets differ across cells:**")
        lines.extend(f"    - {problem}" for problem in problems)
    else:
        lines.append("- Every cell scored the identical pair-id set.")

    lines.extend([
        "",
        "The two construction modes that never read the model must be identical "
        "across arms. This is checked over every reported metric, bitwise:",
        "",
        "| mode | arms compared | identical | max abs difference | worst metric |",
        "|---|---|---|---|---|",
    ])
    for mode in modes_present(cells):
        entry = dependence[mode]
        if entry["identical"] is None:
            verdict, difference, worst = "not testable", "—", "—"
        elif entry["identical"]:
            verdict, difference, worst = "**yes**", "0", "—"
        else:
            verdict = "no"
            difference = f"{entry['max_abs_difference']:.3e}"
            worst = f"`{entry['worst_metric']}`"
        lines.append(
            f"| `{mode}` | {entry['arms_compared']} | {verdict} | {difference} | {worst} |"
        )
    expected_free = [
        m for m in ("identity_current_atoms", "future_frame_current_local")
        if m in dependence
    ]
    violated = [
        m for m in expected_free if dependence[m]["identical"] is False
    ]
    lines.append("")
    if violated:
        lines.append(
            "**FAILED.** " + ", ".join(f"`{m}`" for m in violated) + " is built "
            "without reading the model, so it cannot vary by arm. It does. A "
            "prediction is reaching a construction that must be model-free, and "
            "every delta measured against it is suspect."
        )
    elif expected_free:
        lines.append(
            "Passed: " + ", ".join(f"`{m}`" for m in expected_free) + " agree to "
            "the last bit across every arm, which is what a model-free "
            "construction must do. They are reported once below rather than "
            "repeated per arm."
        )
    return lines


def _scope_section() -> list[str]:
    return [
        "## 2. What H0 is, and what it is not", "",
        "H0 trains nothing. It takes the frozen Phase 1.6 checkpoints exactly as "
        "Stage M scored them and asks what their per-residue frame predictions "
        "imply for heavy atoms, by rigidly transporting each residue's **current** "
        "heavy atoms onto its predicted frame.",
        "",
        "- `transported_current_conformer` is the only **deployable** row: it uses "
        "nothing but the model's own output.",
        "- `identity_current_atoms` is the baseline Stage M already used, carried "
        "to atom resolution: nothing moves.",
        "- `future_frame_current_local` and `predicted_frame_future_local` **read "
        "the future structure**. They are oracles. They bound what a decoder could "
        "reach if one half of the construction were solved perfectly, and they are "
        "not achievable methods.",
        "",
        "Side chains are carried forward rigidly in every mode except the conformer "
        "oracle, so H0 cannot change a torsion. Any error it reports as "
        "'side-chain' is therefore an error a rigid transport **cannot** fix — "
        "which is precisely the quantity H1a needs in order to be worth building.",
    ]


def build_report(cells: dict, records_path: str, manifest: dict | None) -> list[str]:
    dependence = model_dependence(cells, ALL_METRICS)
    problems = check_same_samples(cells)
    lines = [
        "# Phase 1.6 H0 — heavy-atom backmapping of the frozen arms", "",
        f"Records: `{records_path}`  ",
        f"Bootstrap: {BOOTSTRAP_ITERATIONS:,} resamples over domains, seed "
        f"{BOOTSTRAP_SEED}  ",
        "Generated by `scripts/analyze_phase1_6_h0.py`. No checkpoint was read or "
        "written by this script.",
        "",
    ]
    if manifest:
        lines.extend([
            f"Evaluation manifest: `{manifest.get('stage', '?')}`, generated "
            f"{manifest.get('generated', '?')}, config "
            f"`{manifest.get('config', '?')}`.",
            "",
        ])
    lines.extend(_integrity_section(cells, problems, dependence))
    lines.extend(["", *_scope_section(), ""])

    lines.extend(["## 3. Construction modes", ""])
    lines.extend(mode_semantics_table(cells, dependence))

    lines.extend(["", "## 4. Structural accuracy", ""])
    for lag in LAGS:
        if not any(values_of(r, "heavy_atom_rmsd", lag) for r in cells.values()):
            continue
        for metric in STRUCTURE:
            lines.extend(per_cell_table(cells, metric, lag))
            lines.append("")

    lines.extend(["## 5. Frame error vs conformer error", "",
                  "The decomposition H0 exists for. Each row is one arm; the "
                  "columns replace one half of the construction with ground "
                  "truth.", ""])
    for lag in LAGS:
        for metric in STRUCTURE:
            table = decomposition_table(cells, metric, lag)
            if len(table) > 4:
                lines.extend(table)
                lines.append("")
        for metric in STRUCTURE:
            table = delta_table(cells, metric, lag)
            if len(table) > 4:
                lines.extend(table)
                lines.append("")

    lines.extend(["## 6. Physical validity (Bondi overlap)", "",
                  "Radii: Bondi (1964). Overlap `o_ij = r_i + r_j - d_ij`; "
                  "*serious* is `o_ij >= 0.4 Å` (MolProbity's threshold). PSF 1-2 "
                  "and 1-3 pairs are excluded; 1-4 pairs are kept in the primary "
                  "rate and tallied separately.", ""])
    for lag in LAGS:
        for metric in CLASH:
            lines.extend(per_cell_table(cells, metric, lag, digits=6))
            lines.append("")
        lines.extend(["**Serious overlap by atom class** — an intra-residue "
                      "overlap under a rigid transport is inherited from the "
                      "current conformer, not created by the model.", ""])
        for metric in CLASH_CLASSES:
            lines.extend(per_cell_table(cells, metric, lag, digits=6))
            lines.append("")

    lines.extend(["## 7. Atom-level contacts", "",
                  "Macro is the domain-macro mean of per-sample F1; micro is "
                  "recomputed from pooled TP/FP/FN. They answer different "
                  "questions and are never averaged together.", ""])
    for lag in LAGS:
        for prefix in CONTACT_PREFIXES:
            lines.extend(contact_table(cells, prefix, lag))
            lines.append("")

    return lines


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--csv-dir", default=None,
        help="where domain_summary.csv and decomposition_deltas.csv go "
             "(default: the records' directory)",
    )
    args = parser.parse_args()

    cells = load_records(args.records)
    if not cells:
        raise SystemExit(f"no records in {args.records}")

    manifest_path = Path(args.records).with_name("reproducibility_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    csv_dir = Path(args.csv_dir or Path(args.records).parent)
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary = domain_summary_rows(cells, ALL_METRICS)
    write_csv(
        csv_dir / "h0_domain_summary.csv",
        summary,
        ["canonical_arm", "construction_mode", "lag_ns", "domain_id", "n_samples",
         "n_valid_atoms_mean"]
        + [c for metric in ALL_METRICS for c in (metric, f"{metric}_n")],
    )
    deltas = decomposition_rows(cells, STRUCTURE + CLASH + CONTACT_METRICS)
    write_csv(
        csv_dir / "h0_decomposition_deltas.csv",
        deltas,
        ["canonical_arm", "control_mode", "candidate_mode", "metric", "lag_ns",
         "higher_is_better", "n", "domains", "delta", "lo", "hi", "relative",
         "significant", "question"],
    )

    report = build_report(cells, args.records, manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n")

    print(f"wrote {out}")
    print(f"wrote {csv_dir / 'h0_domain_summary.csv'} ({len(summary)} rows)")
    print(f"wrote {csv_dir / 'h0_decomposition_deltas.csv'} ({len(deltas)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
