"""Aggregation and statistics for the Phase 1.6 extended evaluation (Stage M2).

The metric tests check what a number *is*. These check what happens to it on the
way into a table -- which is where the Stage B analyser had its two silent
failure modes:

* a duplicate ``pair_id`` overwrote rather than raised, so a paired comparison
  could run on a sixteenth of the data with nothing to show for it;
* the runner and the analyser used different weightings under the same name.

Both are now errors or are labelled, and both are pinned here. The third group
checks the delta-sign convention, which must give "positive = the candidate
improved" for lower-is-better and higher-is-better metrics alike, or a reader has
to check each column's direction by hand.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import analyze_phase1_6 as stage_b  # noqa: E402
import analyze_phase1_6_extended as stage_m  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def record(
    *, arm, pair_id, domain, lag=1.0, temperature="320", ca_rmsd=1.0,
    contact_f1=0.5, n_valid_residues=100, **extra,
):
    row = {
        "arm": arm, "canonical_arm": arm, "oracle": False, "seed": 0,
        "pair_id": pair_id, "domain_id": domain, "temperature": temperature,
        "replica_id": "0", "current_frame_index": 2, "future_frame_index": 3,
        "lag_ns": lag, "lag_ps": lag * 1000,
        "n_valid_residues": n_valid_residues, "n_valid_torsions": 3 * n_valid_residues,
        "length_bin": "100-149",
        "ca_rmsd": ca_rmsd, "contact_f1": contact_f1,
        "backbone_torsion_mae_deg": 20.0,
        "contact_tp": 10, "contact_fp": 2, "contact_fn": 3,
        "contact_target_positives": 13,
        "formed_contact_tp": 1, "formed_contact_fp": 1, "formed_contact_fn": 4,
        "formed_contact_events_target": 5, "formed_contact_events_predicted": 2,
        "invalid_reason": None,
    }
    row.update(extra)
    return row


def two_arm_fixture(tmp_path, *, control_rmsd=2.0, candidate_rmsd=1.8, domains=6):
    """Two arms over the same ids, with the candidate uniformly better."""
    rows = []
    for d in range(domains):
        for f in range(4):
            pair_id = f"dom{d}/320/0/t{f}/f{f + 1}/lag1000ps"
            rows.append(record(
                arm="S0_structure_history", pair_id=pair_id, domain=f"dom{d}",
                ca_rmsd=control_rmsd + 0.01 * f, contact_f1=0.50,
            ))
            rows.append(record(
                arm="P0_pair_geometry_control", pair_id=pair_id, domain=f"dom{d}",
                ca_rmsd=candidate_rmsd + 0.01 * f, contact_f1=0.62,
            ))
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# --------------------------------------------------------------------------
# 1. loading
# --------------------------------------------------------------------------


def test_a_duplicate_pair_id_is_a_hard_error(tmp_path):
    """Test 19. The Stage B loader overwrote silently; this one must not."""
    rows = [
        record(arm="S0_structure_history", pair_id="same/id", domain="d0"),
        record(arm="S0_structure_history", pair_id="same/id", domain="d0", ca_rmsd=9.0),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(SystemExit, match="duplicate pair_id"):
        stage_m.load_records(str(path))


def test_arms_scoring_different_samples_are_reported_not_silently_intersected(tmp_path):
    """Test 20."""
    rows = [
        record(arm="S0_structure_history", pair_id="a", domain="d0"),
        record(arm="S0_structure_history", pair_id="b", domain="d0"),
        record(arm="P0_pair_geometry_control", pair_id="a", domain="d0"),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    by_arm = stage_m.load_records(str(path))
    problems = stage_m.check_same_samples(by_arm)
    assert problems and "missing" in problems[0]


def test_matching_arms_produce_no_problems(tmp_path):
    by_arm = stage_m.load_records(str(two_arm_fixture(tmp_path)))
    assert stage_m.check_same_samples(by_arm) == []


def test_values_of_drops_null_and_non_finite_rows_rather_than_zero_filling(tmp_path):
    """Test 18 again, at the aggregation layer: a null must not become a 0."""
    rows = [
        record(arm="S0_structure_history", pair_id="a", domain="d0", ca_rmsd=1.0),
        record(arm="S0_structure_history", pair_id="b", domain="d0", ca_rmsd=None),
        record(arm="S0_structure_history", pair_id="c", domain="d0",
               ca_rmsd=float("nan")),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    by_arm = stage_m.load_records(str(path))
    values = stage_m.values_of(by_arm["S0_structure_history"], "ca_rmsd", lag=1.0)
    assert len(values) == 1
    assert float(np.mean(list(values.values()))) == 1.0


# --------------------------------------------------------------------------
# 2. bootstrap
# --------------------------------------------------------------------------


def test_fast_cluster_bootstrap_matches_the_stage_b_implementation():
    """The Stage M bootstrap resamples sums; the Stage B one resamples vectors.

    They must agree on the observed mean exactly and on the interval to within
    Monte-Carlo noise, or the extended report's intervals are not comparable with
    the Stage B report's.
    """
    rng = np.random.default_rng(0)
    values = {}
    for d in range(25):
        for f in range(rng.integers(20, 90)):
            values[(f"dom{d}/f{f}", f"dom{d}")] = float(rng.normal(3.0, 0.8))

    mean_m, lo_m, hi_m = stage_m.cluster_bootstrap(values, iterations=4000, seed=7)
    mean_b, lo_b, hi_b = stage_b.cluster_bootstrap(values, iterations=4000, seed=7)

    assert mean_m == pytest.approx(mean_b, rel=1e-12)
    assert lo_m == pytest.approx(lo_b, abs=0.02)
    assert hi_m == pytest.approx(hi_b, abs=0.02)


def test_the_bootstrap_clusters_on_domains_not_frames():
    """Frames of one protein are not independent looks at the question.

    Both datasets hold 200 values with the same between-protein spread. In one
    they come from 20 proteins, in the other from 4. Resampling *frames* would
    give both the same interval, because both have 200 numbers. Resampling
    **domains** must make the 4-protein interval much wider, because it has only
    four independent observations of the thing that actually varies.
    """
    rng = np.random.default_rng(1)
    values_wide, values_narrow = {}, {}
    wide_offsets = rng.normal(0.0, 1.0, 20)
    narrow_offsets = rng.normal(0.0, 1.0, 4)
    for i in range(200):
        jitter = float(rng.normal(0.0, 0.05))
        values_wide[(f"p{i}", f"dom{i % 20}")] = 2.0 + wide_offsets[i % 20] + jitter
        values_narrow[(f"p{i}", f"dom{i % 4}")] = 2.0 + narrow_offsets[i % 4] + jitter

    _, lo_w, hi_w = stage_m.cluster_bootstrap(values_wide, iterations=4000)
    _, lo_n, hi_n = stage_m.cluster_bootstrap(values_narrow, iterations=4000)
    assert (hi_n - lo_n) > 2 * (hi_w - lo_w)


def test_a_frame_level_interval_would_be_far_narrower_than_the_cluster_one():
    """Quantifies why the clustering matters, rather than only asserting it."""
    rng = np.random.default_rng(2)
    offsets = rng.normal(0.0, 0.9, 12)
    values = {
        (f"p{i}", f"dom{i % 12}"): 3.0 + offsets[i % 12] + float(rng.normal(0, 0.05))
        for i in range(600)
    }
    _, lo, hi = stage_m.cluster_bootstrap(values, iterations=4000)
    naive = 1.96 * float(np.std(list(values.values()), ddof=1)) / math.sqrt(600)
    assert (hi - lo) > 4 * (2 * naive)


def test_the_bootstrap_is_deterministic_for_a_fixed_seed():
    values = {(f"p{i}", f"dom{i % 5}"): float(i % 7) for i in range(80)}
    first = stage_m.cluster_bootstrap(values, iterations=500, seed=42)
    second = stage_m.cluster_bootstrap(values, iterations=500, seed=42)
    assert first == second


def test_a_single_domain_gives_no_interval_rather_than_a_fake_one():
    values = {(f"p{i}", "dom0"): float(i) for i in range(10)}
    mean, lo, hi = stage_m.cluster_bootstrap(values)
    assert mean == pytest.approx(4.5)
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------
# 3. delta sign convention
# --------------------------------------------------------------------------


def test_positive_delta_means_candidate_better_for_a_lower_is_better_metric(tmp_path):
    by_arm = stage_m.load_records(str(two_arm_fixture(tmp_path)))
    result = stage_m.paired_delta(
        stage_m.values_of(by_arm["S0_structure_history"], "ca_rmsd", lag=1.0),
        stage_m.values_of(by_arm["P0_pair_geometry_control"], "ca_rmsd", lag=1.0),
        "ca_rmsd",
    )
    assert result["delta"] == pytest.approx(0.2)   # control 2.0 -> candidate 1.8
    assert result["significant"]


def test_positive_delta_means_candidate_better_for_a_higher_is_better_metric(tmp_path):
    """Brief §13.2. Without the flip this column would point the other way."""
    from force_md.transition.extended_metrics import HIGHER_IS_BETTER

    assert "contact_f1" in HIGHER_IS_BETTER
    by_arm = stage_m.load_records(str(two_arm_fixture(tmp_path)))
    result = stage_m.paired_delta(
        stage_m.values_of(by_arm["S0_structure_history"], "contact_f1", lag=1.0),
        stage_m.values_of(by_arm["P0_pair_geometry_control"], "contact_f1", lag=1.0),
        "contact_f1",
    )
    assert result["delta"] == pytest.approx(0.12)  # 0.62 - 0.50, not 0.50 - 0.62


def test_a_worse_candidate_gives_a_negative_delta_in_both_directions(tmp_path):
    path = two_arm_fixture(tmp_path, control_rmsd=1.5, candidate_rmsd=2.1)
    by_arm = stage_m.load_records(str(path))
    result = stage_m.paired_delta(
        stage_m.values_of(by_arm["S0_structure_history"], "ca_rmsd", lag=1.0),
        stage_m.values_of(by_arm["P0_pair_geometry_control"], "ca_rmsd", lag=1.0),
        "ca_rmsd",
    )
    assert result["delta"] < 0

    rows = json.loads("[" + ",".join(
        line for line in path.read_text().splitlines() if line
    ) + "]")
    for row in rows:
        if row["arm"] == "P0_pair_geometry_control":
            row["contact_f1"] = 0.30       # worse than the control's 0.50
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    by_arm = stage_m.load_records(str(path))
    result = stage_m.paired_delta(
        stage_m.values_of(by_arm["S0_structure_history"], "contact_f1", lag=1.0),
        stage_m.values_of(by_arm["P0_pair_geometry_control"], "contact_f1", lag=1.0),
        "contact_f1",
    )
    assert result["delta"] == pytest.approx(-0.20)


def test_paired_delta_uses_only_shared_samples():
    control = {("a", "d0"): 1.0, ("b", "d0"): 5.0}
    candidate = {("a", "d0"): 0.5}
    result = stage_m.paired_delta(control, candidate, "ca_rmsd")
    assert result["n"] == 1
    assert result["delta"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 4. aggregation rules
# --------------------------------------------------------------------------


def test_micro_and_macro_contact_metrics_are_computed_separately():
    """Brief §12.2. Pooled counts and a mean of per-sample F1 are not the same."""
    records = [
        record(arm="a", pair_id="p0", domain="d0",
               contact_tp=90, contact_fp=10, contact_fn=0),
        record(arm="a", pair_id="p1", domain="d0",
               contact_tp=0, contact_fp=1, contact_fn=1),
    ]
    pooled = stage_m.pooled_contact(records, "contact", 1.0)
    assert pooled["tp"] == 90 and pooled["fp"] == 11 and pooled["fn"] == 1
    assert pooled["f1"] == pytest.approx(2 * 90 / (2 * 90 + 11 + 1))
    # the second sample's own F1 is 0.0, so a macro mean would be ~0.47 -- a very
    # different number from the micro 0.94. Neither is wrong; blending is.
    assert pooled["f1"] > 0.9


def test_samples_with_and_without_an_event_are_counted_separately():
    records = [
        record(arm="a", pair_id="p0", domain="d0", formed_contact_events_target=4),
        record(arm="a", pair_id="p1", domain="d0", formed_contact_events_target=0),
        record(arm="a", pair_id="p2", domain="d0", formed_contact_events_target=0),
    ]
    pooled = stage_m.pooled_contact(records, "formed_contact", 1.0)
    assert pooled["with_event"] == 1
    assert pooled["without_event"] == 2
    assert pooled["samples"] == 3


def test_torsion_aggregations_are_kept_apart():
    """Brief §12.3: residue-pooled, sample-macro and domain-macro are three rows."""
    records = [
        record(arm="a", pair_id="p0", domain="big", n_valid_residues=200,
               backbone_torsion_mae_deg=10.0),
        record(arm="a", pair_id="p1", domain="small", n_valid_residues=20,
               backbone_torsion_mae_deg=40.0),
        record(arm="a", pair_id="p2", domain="small", n_valid_residues=20,
               backbone_torsion_mae_deg=40.0),
    ]
    summary = stage_m.pooled_torsion(records, 1.0)
    # residue-pooled: the 200-residue protein dominates
    assert summary["residue_pooled"] == pytest.approx(
        (10.0 * 600 + 40.0 * 60 + 40.0 * 60) / 720
    )
    assert summary["sample_macro"] == pytest.approx(30.0)
    assert summary["domain_macro"] == pytest.approx(25.0)  # (10 + 40) / 2
    assert len({summary["residue_pooled"], summary["sample_macro"],
                summary["domain_macro"]}) == 3


def test_residue_weighted_mean_reproduces_the_stage_b_micro_weighting():
    """Audit §3: this is the number results.csv prints, and it differs."""
    records = [
        record(arm="a", pair_id="p0", domain="d0", n_valid_residues=200, ca_rmsd=1.0),
        record(arm="a", pair_id="p1", domain="d0", n_valid_residues=20, ca_rmsd=5.0),
    ]
    weighted = stage_m.residue_weighted(records, "ca_rmsd", 1.0)
    sample_equal = float(np.mean([1.0, 5.0]))
    assert weighted == pytest.approx((1.0 * 200 + 5.0 * 20) / 220)
    assert abs(weighted - sample_equal) > 1.0   # they really are different


def test_domain_macro_counts_every_protein_once():
    values = {
        ("p0", "big"): 1.0, ("p1", "big"): 1.0, ("p2", "big"): 1.0,
        ("p3", "small"): 5.0,
    }
    assert stage_m.domain_macro(values) == pytest.approx(3.0)
    assert float(np.mean(list(values.values()))) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 5. no composite score
# --------------------------------------------------------------------------


def test_the_analysis_defines_no_weighted_composite_score():
    """Brief §14. A composite hides the trade-off the Pareto plots exist to show.

    Checked against the module's *namespace*, not its prose: the docstring says
    the word "composite" precisely to explain why there isn't one.
    """
    import report_phase1_6_extended as report

    for module in (stage_m, report):
        offenders = [
            name for name in dir(module)
            if any(token in name.lower()
                   for token in ("composite", "combined_score", "overall_score",
                                 "weighted_sum", "total_score"))
        ]
        assert offenders == [], (module.__name__, offenders)

    # The two co-primaries are reported as a pair and are never summed.
    assert stage_m.CO_PRIMARY == ("ca_rmsd", "drmsd_long_range")


def test_the_report_states_the_limits_the_brief_requires_it_to_state():
    """Brief §18, checked on the rendered text so it cannot drift out."""
    import report_phase1_6_extended as report

    lines = report._conclusions_section({}, stage_m.values_of, stage_m.paired_delta)
    # Emphasis markers are stripped so the assertion is on the sentence, not on
    # where the bold happens to fall.
    text = "\n".join(lines).lower().replace("*", "")
    for required in (
        "none of them is a thermodynamic or kinetic quantity",
        "not, on its own, an improvement in molecular dynamics",
        "no number is estimated for them",
        "is not a reproducible superiority",
        "no across-seed standard deviation is quoted",
    ):
        assert required in text, required


def test_a_comparison_that_was_not_run_produces_no_conclusion():
    """Brief §18's last ban, and the one an `if not better` fallback walks into.

    "not run" is also "not better", so a two-branch conclusion writes a finding
    about an experiment that never happened. Every arm here is absent except the
    P0/S0 pair, and every other bullet must say so and stop.
    """
    import report_phase1_6_extended as report

    scored = {
        arm: [
            record(arm=arm, pair_id=f"p{i}", domain=f"d{i % 5}",
                   ca_rmsd=2.0 - (0.3 if arm.startswith("P0") else 0.0))
            for i in range(20)
        ]
        for arm in ("S0_structure_history", "P0_pair_geometry_control")
    }
    text = "\n".join(
        report._conclusions_section(scored, stage_m.values_of, stage_m.paired_delta)
    )

    for label in ("P1 vs P0", "P2 vs P1", "Oracle vs S0"):
        line = next(l for l in text.splitlines() if label in l)
        assert "not run" in line, label
    assert "nothing is concluded from it" in text
    # The claims that would be wrong must be absent entirely.
    assert "is de-prioritised" not in text
    assert "does not beat" not in text
    # ... including the §18 bullet whose justification is a measured absence of
    # pair-physics gain. With P1 unscored there is no such measurement.
    assert "has not been shown to improve any structural endpoint" not in text
    assert "`P1` was not scored here" in text


def test_the_report_never_claims_an_oracle_failure_generalises_to_forces():
    """Brief §18's fourth banned reading, checked on the rendered sentence."""
    import report_phase1_6_extended as report

    scored = {
        "S0_structure_history": [
            record(arm="S0_structure_history", pair_id=f"p{i}", domain=f"d{i % 5}",
                   ca_rmsd=2.0)
            for i in range(20)
        ],
        "O_current_gt_force_oracle": [
            record(arm="O_current_gt_force_oracle", pair_id=f"p{i}",
                   domain=f"d{i % 5}", ca_rmsd=2.0)
            for i in range(20)
        ],
    }
    text = "\n".join(
        report._conclusions_section(scored, stage_m.values_of, stage_m.paired_delta)
    ).lower()
    assert "not about forces in general" in text
    assert "within this force encoding" in text
    for banned in ("forces are useless", "forces carry no information",
                   "information-theoretically impossible"):
        assert banned not in text, banned


# --------------------------------------------------------------------------
# 6. csv output
# --------------------------------------------------------------------------


def test_csv_writes_an_empty_cell_for_a_missing_value_not_a_zero(tmp_path):
    path = tmp_path / "out.csv"
    stage_m.write_csv(
        path,
        [{"a": 1.0, "b": None}, {"a": float("nan"), "b": "x,y"}],
        ["a", "b"],
    )
    text = path.read_text().splitlines()
    assert text[0] == "a,b"
    assert text[1] == "1.0,"
    assert text[2] == ',"x,y"'


def test_paired_delta_rows_record_the_bootstrap_settings(tmp_path):
    by_arm = stage_m.load_records(str(two_arm_fixture(tmp_path)))
    rows = stage_m.paired_delta_rows(by_arm, ["ca_rmsd", "contact_f1"])
    assert rows
    for row in rows:
        assert row["bootstrap_iterations"] == stage_m.BOOTSTRAP_ITERATIONS
        assert row["bootstrap_seed"] == stage_m.BOOTSTRAP_SEED
        assert "delta_positive_means_candidate_better" in row
        assert isinstance(row["higher_is_better"], bool)
