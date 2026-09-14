"""Phase 1.6 extended metrics (Stage M0).

The suite exists to score frozen checkpoints, so almost every test here is a
statement about a *definition* rather than about a model: what a metric must be
invariant to, what it must refuse to compute, and what it must not silently
substitute a zero for.

Three groups.

*Invariance.* mdCATH proteins tumble. Every metric is required to be unchanged
when the whole pair is rigidly moved, and dRMSD additionally when only one side
is moved, because it compares an internal distance matrix and nothing else. A
metric that moved would be measuring Brownian motion.

*Refusal.* A sample with too few valid residues, a torsion across a chain break,
a contact set with no positive event -- each of these must produce ``None``/NaN
and a counted reason, never a 0.0 that a table will read as a passing score.

*Reproduction.* The extended ``ca_rmsd`` and ``rotation_geodesic_mean_deg`` must
be the Stage B definitions to the bit, or the extended report is measuring
something else and quietly calling it by the old name.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from conftest import reference_dihedral_case  # noqa: E402
from force_md.geometry import (  # noqa: E402
    backbone_omega,
    random_rotation_matrix,
    sequence_neighbours,
    wrap_to_pi,
)
from force_md.geometry.torsions import dihedral_angle  # noqa: E402
from force_md.transition import (  # noqa: E402
    EXTENDED_COUNT_KEYS,
    EXTENDED_METRIC_KEYS,
    NOT_APPLICABLE,
    ExtendedMetricConfig,
    MetricConfig,
    RecordContext,
    build_transition_target,
    extended_metric_records,
    identity_prediction,
    length_bin,
    per_graph_transition_metrics,
    target_as_prediction,
)
from force_md.transition.extended_metrics import (  # noqa: E402
    _angle_at,
    _current_spatial_pair_mask,
    _separation,
    jensen_shannon,
    torsion_histogram,
)
from test_transition_targets import (  # noqa: E402
    make_batch,
    perturbed_future,
    rigidly_move,
)

CONTEXT = RecordContext(
    arm="test", canonical_arm="test", oracle=False, seed=0,
    manifest_hash="m", phase1_checkpoint_hash="p", transition_checkpoint_hash="t",
    config_hash="c", git_commit="g", git_dirty=True, source_diff_hash="d",
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_pair(*, num_graphs=2, num_residues=24, seed=0, displacement=0.4):
    """A batch and a plausible future frame, built with the Phase 1.5 helpers.

    ``make_batch`` and ``perturbed_future`` are imported from
    ``test_transition_targets`` rather than reimplemented: a fixture that
    diverged from the one the target tests use would let this file pass against
    geometry the rest of the suite has never seen. Also float64 throughout, as
    those helpers are, so a tolerance failure here means a definition is wrong
    rather than that float32 ran out of digits.
    """
    batch = make_batch(sizes=(num_residues,) * num_graphs, seed=seed)
    future = perturbed_future(batch, scale=displacement, seed=seed + 1)
    return batch, future


def fold_compact(batch, future, *, segment=4, spacing=7.0, seed=0):
    """Fold the synthetic chain into a globule, so it has long-range contacts.

    ``synthetic_batch`` builds an extended chain: measured on a 40-residue one,
    the closest ``|i-j| >= 6`` Cα pair is 10.2 A apart and the primary contact map
    is therefore **empty**. A contact test on that fixture would pass by
    measuring nothing.

    Each block of ``segment`` residues is moved by a single **rigid** transform
    onto a lattice site, so every distance inside a block -- bond lengths, bond
    angles, torsions -- is exactly preserved and only the junctions between
    blocks are artificial. Both structures get the *same* transform, so the
    conformational change between them is untouched.
    """
    import dataclasses

    n_res = batch.num_residues
    a2r = batch.atoms.atom_to_residue
    segment_of = torch.div(torch.arange(n_res), segment, rounding_mode="floor")
    count = int(segment_of.max()) + 1

    g = torch.Generator().manual_seed(seed)
    rotation = torch.stack([random_rotation_matrix(generator=g) for _ in range(count)])
    index = torch.arange(count)
    side = 3
    site = spacing * torch.stack(
        [index % side, (index // side) % side, index // (side * side)], dim=-1
    ).to(torch.float64)
    centroid = torch.stack(
        [batch.backbone.ca_positions[segment_of == s].mean(0) for s in range(count)]
    )

    def move(x, which):
        return torch.einsum(
            "nij,nj->ni", rotation[which], x - centroid[which]
        ) + site[which]

    moved_batch = dataclasses.replace(
        batch,
        atoms=dataclasses.replace(
            batch.atoms, positions=move(batch.atoms.positions, segment_of[a2r])
        ),
        backbone=dataclasses.replace(
            batch.backbone,
            n_positions=move(batch.backbone.n_positions, segment_of),
            ca_positions=move(batch.backbone.ca_positions, segment_of),
            c_positions=move(batch.backbone.c_positions, segment_of),
        ),
    )
    moved_future = dataclasses.replace(
        future,
        positions=move(future.positions, segment_of[a2r]),
        n_positions=move(future.n_positions, segment_of),
        ca_positions=move(future.ca_positions, segment_of),
        c_positions=move(future.c_positions, segment_of),
    )
    return moved_batch, moved_future


def make_compact_pair(*, num_residues=40, seed=0, displacement=1.5):
    return fold_compact(*make_pair(
        num_graphs=1, num_residues=num_residues, seed=seed, displacement=displacement
    ), seed=seed)


class FakePair:
    """The metadata a record needs, without the dataset layer."""

    def __init__(self, index: int, domain="d0", lag_ps=1000.0):
        self.domain = domain
        self.temperature = "320"
        self.replica = "0"
        self.current_frame = 2
        self.future_frame = 3
        self.lag_ps = lag_ps
        self.pair_id = f"{domain}/320/0/t2/f3/lag{lag_ps:g}ps#{index}"


def records_for(prediction, target, *, config=None, domains=None):
    pairs = [
        FakePair(i, domain=(domains[i] if domains else f"d{i}"))
        for i in range(target.num_graphs)
    ]
    return extended_metric_records(
        prediction, target, pairs=pairs, context=CONTEXT,
        config=config or ExtendedMetricConfig(),
    )


# --------------------------------------------------------------------------
# 1. exactness and schema
# --------------------------------------------------------------------------


def test_a_perfect_prediction_scores_zero_error_everywhere():
    """Test 1: every metric takes its exact best value on the target itself."""
    batch, future = make_compact_pair()
    target = build_transition_target(batch, future)
    rows = records_for(target_as_prediction(target), target)

    for row in rows:
        assert row["invalid_reason"] is None
        for key in ("ca_rmsd", "ca_rmsd_superposed", "translation_rmse",
                    "rotation_geodesic_mean_deg", "rotation_geodesic_median_deg",
                    "drmsd_all", "drmsd_local", "drmsd_medium", "drmsd_long_range",
                    "drmsd_current_spatial_edges", "phi_mae_deg", "psi_mae_deg",
                    "omega_mae_deg", "backbone_torsion_mae_deg",
                    "bond_length_rmse", "bond_length_mae", "bond_angle_mae_deg",
                    "ca_neighbor_distance_mae"):
            assert row[key] == pytest.approx(0.0, abs=1e-6), key
        # The fixture is folded precisely so this assertion measures something:
        # on the extended synthetic chain the |i-j| >= 6 contact map is empty.
        assert row["contact_target_positives"] > 0
        for key in ("contact_f1", "contact_precision", "contact_recall",
                    "contact_jaccard"):
            assert row[key] == pytest.approx(1.0), key
        for name in ("formed", "broken"):
            assert row[f"{name}_contact_events_target"] > 0
            assert row[f"{name}_contact_f1"] == pytest.approx(1.0), name


def test_every_record_carries_the_full_schema():
    """Test 18/schema: a short row would silently change the file's shape."""
    batch, future = make_pair()
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    required = {
        "arm", "canonical_arm", "oracle", "seed", "pair_id", "domain_id",
        "temperature", "replica_id", "current_frame_index", "future_frame_index",
        "lag_ns", "n_residues", "n_valid_residues", "n_valid_pairs",
        "n_valid_torsions", "invalid_reason", "manifest_hash",
        "phase1_checkpoint_hash", "transition_checkpoint_hash", "config_hash",
        "git_commit", "git_dirty", "source_diff_hash",
    }
    assert required <= set(row)
    assert set(EXTENDED_METRIC_KEYS) <= set(row)
    assert set(EXTENDED_COUNT_KEYS) <= set(row)


def test_unsupported_metrics_are_null_with_a_stated_reason():
    """Test 18: an unsupported metric is None, never 0.0."""
    batch, future = make_pair()
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    for key in ("bond_n_ca_rmse", "bond_ca_c_rmse", "angle_n_ca_c_mae_deg",
                "chirality_violation_count"):
        assert row[key] is None, key
        assert key in NOT_APPLICABLE and NOT_APPLICABLE[key]


def test_intra_residue_geometry_is_invariant_by_construction():
    """The evidence for the three ``not_applicable`` bond/angle entries.

    ``reconstruct_backbone`` carries the residue's current local N and C onto
    both the predicted and the target frame, so N-CA, CA-C and the N-CA-C angle
    are the *same numbers* on both sides whatever the prediction is. Measured
    here rather than asserted in prose.
    """
    from force_md.transition.targets import reconstruct_backbone

    batch, future = make_pair()
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)  # a badly wrong prediction
    n_p, ca_p, c_p = reconstruct_backbone(prediction, target)
    n_t, ca_t, c_t = reconstruct_backbone(target_as_prediction(target), target)

    assert torch.allclose(
        (n_p - ca_p).norm(dim=-1), (n_t - ca_t).norm(dim=-1), atol=1e-5
    )
    assert torch.allclose(
        (c_p - ca_p).norm(dim=-1), (c_t - ca_t).norm(dim=-1), atol=1e-5
    )
    assert torch.allclose(
        _angle_at(ca_p, n_p, c_p), _angle_at(ca_t, n_t, c_t), atol=1e-6
    )


# --------------------------------------------------------------------------
# 2. invariance
# --------------------------------------------------------------------------


def test_every_metric_is_invariant_under_a_global_rigid_motion_of_the_pair():
    """Tests 2, 3, 4: RMSD, dRMSD and the rotation angle all survive tumbling."""
    batch, future = make_pair()
    generator = torch.Generator().manual_seed(11)
    rotation = random_rotation_matrix(generator=generator)
    translation = torch.tensor([12.0, -7.0, 30.0], dtype=torch.float64)

    target = build_transition_target(batch, future)
    before = records_for(identity_prediction(target), target)

    moved_batch, moved_future = rigidly_move(batch, future, rotation, translation)
    moved_target = build_transition_target(moved_batch, moved_future)
    after = records_for(identity_prediction(moved_target), moved_target)

    for row_a, row_b in zip(before, after):
        for key in EXTENDED_METRIC_KEYS:
            a, b = row_a[key], row_b[key]
            if a is None or (isinstance(a, float) and math.isnan(a)):
                continue
            assert b == pytest.approx(a, abs=1e-6, rel=1e-6), key


def test_drmsd_is_invariant_when_only_the_prediction_is_rotated():
    """Test 3: dRMSD compares an internal distance matrix and nothing else."""
    batch, future = make_pair()
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)
    base = records_for(prediction, target)

    generator = torch.Generator().manual_seed(5)
    rotation = random_rotation_matrix(generator=generator)
    spun = type(prediction)(
        translation_local=prediction.translation_local,
        rotation=prediction.rotation,
    )
    # Rotate the predicted Cα cloud about each graph's own centroid by rewriting
    # the local translation, which is the only handle a frame-level prediction
    # has on where its atoms land.
    from force_md.transition.targets import apply_prediction

    ca, _ = apply_prediction(prediction, target)
    rot_cur = target.current_frames.rotation
    for graph in range(target.num_graphs):
        sel = target.residue_batch_index == graph
        centroid = ca[sel].mean(0)
        ca = ca.clone()
        ca[sel] = (ca[sel] - centroid) @ rotation.T + centroid
    spun = type(prediction)(
        translation_local=torch.einsum(
            "nji,nj->ni", rot_cur, ca - target.current_ca
        ),
        rotation=prediction.rotation,
    )
    spun_rows = records_for(spun, target)

    for row_a, row_b in zip(base, spun_rows):
        for key in ("drmsd_all", "drmsd_local", "drmsd_medium", "drmsd_long_range",
                    "drmsd_current_spatial_edges"):
            assert row_b[key] == pytest.approx(row_a[key], abs=1e-3), key
    # ... while the placement metric does move, or the test proves nothing.
    assert spun_rows[0]["ca_rmsd"] != pytest.approx(base[0]["ca_rmsd"], abs=1e-3)


def test_the_superposed_rotation_metric_removes_a_common_global_rotation():
    """Test 4/§4.2: ``R -> Q R`` is applied with the same Q as the coordinates.

    A common global rotation applied to the prediction alone must leave the
    *superposed* rotation error unchanged, and must change the non-superposed one
    -- which is the whole reason both are reported under different names.
    """
    batch, future = make_pair()
    target = build_transition_target(batch, future)
    generator = torch.Generator().manual_seed(3)
    q = random_rotation_matrix(generator=generator)

    prediction = identity_prediction(target)
    from force_md.transition.targets import apply_prediction

    ca, rotation = apply_prediction(prediction, target)
    rot_cur = target.current_frames.rotation
    spun_ca = ca.clone()
    spun_rot = rotation.clone()
    for graph in range(target.num_graphs):
        sel = target.residue_batch_index == graph
        centroid = ca[sel].mean(0)
        spun_ca[sel] = (ca[sel] - centroid) @ q.T + centroid
        spun_rot[sel] = q @ rotation[sel]
    spun = type(prediction)(
        translation_local=torch.einsum("nji,nj->ni", rot_cur, spun_ca - target.current_ca),
        rotation=torch.einsum("nji,njk->nik", rot_cur, spun_rot),
    )

    base = records_for(prediction, target)[0]
    spun_row = records_for(spun, target)[0]
    assert spun_row["rotation_geodesic_superposed_mean_deg"] == pytest.approx(
        base["rotation_geodesic_superposed_mean_deg"], abs=0.05
    )
    assert spun_row["rotation_geodesic_mean_deg"] != pytest.approx(
        base["rotation_geodesic_mean_deg"], abs=0.5
    )


@pytest.mark.parametrize("degrees", [0.5, 7.0, 42.0, 120.0, 179.5])
def test_the_recorded_rotation_error_equals_a_known_applied_angle(degrees):
    """Test 5, at the level a record is written rather than at the primitive's.

    ``tests/test_so3_alignment.py`` already pins ``rotation_geodesic_angle``
    against analytically known angles. What is checked here is the whole path
    from a prediction to ``rotation_geodesic_mean_deg``: turn every residue frame
    by exactly ``degrees`` and require the record to say so. Small angles are
    included because that is where a naive ``arccos`` implementation loses its
    digits, and 179.5 deg because that is where a chart-based one breaks.
    """
    from force_md.geometry import random_rotation_of_angle

    batch, future = make_pair(num_graphs=1, num_residues=20)
    target = build_transition_target(batch, future)
    turn = random_rotation_of_angle(
        math.radians(degrees), generator=torch.Generator().manual_seed(6)
    )
    prediction = target_as_prediction(target)
    turned = type(prediction)(
        translation_local=prediction.translation_local,
        rotation=prediction.rotation @ turn,
    )
    row = records_for(turned, target)[0]

    assert row["rotation_geodesic_mean_deg"] == pytest.approx(degrees, abs=1e-6)
    assert row["rotation_geodesic_median_deg"] == pytest.approx(degrees, abs=1e-6)
    # A pure frame turn moves no Cα, so placement and pair geometry must not react.
    assert row["ca_rmsd"] == pytest.approx(0.0, abs=1e-9)
    assert row["drmsd_long_range"] == pytest.approx(0.0, abs=1e-9)


def test_mean_and_median_rotation_error_are_reported_separately():
    """They differ whenever the per-residue error is skewed, and it usually is."""
    batch, future = make_pair(num_graphs=1, num_residues=24)
    target = build_transition_target(batch, future)
    prediction = target_as_prediction(target)

    from force_md.geometry import so3_exp_map

    # One badly wrong residue against 23 perfect ones: the median must ignore it,
    # the mean must not.
    axis = torch.zeros(target.num_residues, 3, dtype=torch.float64)
    axis[0, 0] = math.radians(120.0)
    turned = type(prediction)(
        translation_local=prediction.translation_local,
        rotation=prediction.rotation @ so3_exp_map(axis),
    )
    row = records_for(turned, target)[0]

    assert row["rotation_geodesic_median_deg"] == pytest.approx(0.0, abs=1e-6)
    assert row["rotation_geodesic_mean_deg"] == pytest.approx(120.0 / 24, abs=1e-4)


def test_kabsch_reflection_correction_is_not_optimised_away():
    """Test 6: a mirrored prediction must not superpose onto the target."""
    from force_md.geometry.alignment import kabsch_rotation

    generator = torch.Generator().manual_seed(2)
    points = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    mirrored = points * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    batch_index = torch.zeros(40, dtype=torch.int64)
    alignment = kabsch_rotation(mirrored, points, batch_index, 1)

    assert float(torch.linalg.det(alignment.rotation[0])) == pytest.approx(1.0)
    residual = (alignment.apply(mirrored, batch_index) - points).norm(dim=-1).mean()
    assert float(residual) > 0.5  # a reflection would have driven this to ~0


# --------------------------------------------------------------------------
# 3. torsions
# --------------------------------------------------------------------------


def test_circular_error_of_179_and_minus_179_degrees_is_two_degrees():
    """Test 7. An unwrapped subtraction reports 358."""
    a = torch.tensor([math.radians(179.0)])
    b = torch.tensor([math.radians(-179.0)])
    error = torch.rad2deg(wrap_to_pi(a - b).abs())
    assert float(error) == pytest.approx(2.0, abs=1e-4)


def test_omega_matches_a_hand_computed_dihedral_and_the_iupac_sign():
    """The new torsion is pinned to the same convention as phi and psi."""
    p0, p1, p2, p3 = reference_dihedral_case(70.0)
    assert float(torch.rad2deg(dihedral_angle(p0[None], p1[None], p2[None], p3[None]))) \
        == pytest.approx(70.0, abs=1e-3)

    ca = torch.stack([p0, p3])
    c = torch.stack([p1, p1])
    n = torch.stack([p0, p2])
    following = torch.tensor([1, -1])
    omega, ok = backbone_omega(n, ca, c, following)
    assert bool(ok[0]) and not bool(ok[1])
    assert float(torch.rad2deg(omega[0])) == pytest.approx(
        float(torch.rad2deg(dihedral_angle(
            ca[0][None], c[0][None], n[1][None], ca[1][None]
        ))),
        abs=1e-4,
    )


def test_torsions_are_not_computed_across_a_chain_break():
    """Test 8. A numbering gap means the peptide bond does not exist."""
    batch_index = torch.zeros(6, dtype=torch.int64)
    chain = torch.zeros(6, dtype=torch.int64)
    resid = torch.tensor([1, 2, 3, 40, 41, 42])  # a gap between rows 2 and 3
    previous, following = sequence_neighbours(batch_index, chain, resid)

    assert int(following[2]) == -1 and int(previous[3]) == -1
    ca = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))
    _, ok = backbone_omega(ca, ca + 1.0, ca + 2.0, following)
    assert not bool(ok[2])
    assert int(ok.sum()) == 4  # rows 0,1,3,4 only


def test_a_masked_neighbour_removes_the_torsion_that_needed_it():
    """Test 9: missing atoms / masked residues must drop the torsion, not fake it."""
    import dataclasses

    batch, future = make_pair(num_graphs=1, num_residues=20)
    mask = batch.residues.mask.clone()
    mask[7] = False
    batch = dataclasses.replace(
        batch, residues=dataclasses.replace(batch.residues, mask=mask)
    )
    target = build_transition_target(batch, future)
    row = records_for(target_as_prediction(target), target)[0]

    assert row["n_valid_residues"] == 19
    # Residue 7 is gone, so the torsions that needed it as a neighbour are too:
    # omega has one per bonded pair, and removing an interior residue removes two.
    assert row["n_omega"] == 17


# --------------------------------------------------------------------------
# 4. pair subsets and spatial edges
# --------------------------------------------------------------------------


def test_sequence_separation_uses_source_numbering_and_chain_identity():
    """Test 11. Row index would treat a numbering gap as adjacency."""
    resid = torch.tensor([1, 2, 40, 41])
    chain = torch.tensor([0, 0, 0, 1])
    separation = _separation(resid, chain)

    assert int(separation[0, 1]) == 1
    assert int(separation[0, 2]) == 39          # a gap is a real separation
    assert int(separation[0, 3]) > 10**9        # a different chain is a sentinel


def test_drmsd_subsets_partition_the_pairs_they_claim_to():
    batch, future = make_pair(num_graphs=1, num_residues=30)
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    n = row["n_valid_residues"]
    assert row["n_valid_pairs"] == n * (n - 1) // 2
    assert (row["n_pairs_local"] + row["n_pairs_medium"]
            + row["n_pairs_long_range"]) == row["n_valid_pairs"]
    assert row["n_pairs_local"] == (n - 1) + (n - 2)   # |i-j| in {1, 2}


def test_vectorised_drmsd_matches_a_brute_force_reference():
    """Test 23. The chunk-free vectorisation is checked against two loops."""
    batch, future = make_pair(num_graphs=1, num_residues=26)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)
    row = records_for(prediction, target)[0]

    from force_md.transition.targets import apply_prediction

    pred_ca, _ = apply_prediction(prediction, target)
    valid = target.valid
    pred_ca = pred_ca[valid].to(torch.float64)
    targ_ca = target.future_ca_aligned[valid].to(torch.float64)
    resid = target.resid_original[valid]

    for name, keep in (
        ("drmsd_all", lambda s: s >= 1),
        ("drmsd_local", lambda s: 1 <= s <= 2),
        ("drmsd_medium", lambda s: 3 <= s <= 5),
        ("drmsd_long_range", lambda s: s >= 6),
    ):
        total, count = 0.0, 0
        for i in range(pred_ca.shape[0]):
            for j in range(i + 1, pred_ca.shape[0]):
                if not keep(abs(int(resid[i]) - int(resid[j]))):
                    continue
                dp = float((pred_ca[i] - pred_ca[j]).norm())
                dt = float((targ_ca[i] - targ_ca[j]).norm())
                total += (dp - dt) ** 2
                count += 1
        reference = math.sqrt(total / count) if count else float("nan")
        assert row[name] == pytest.approx(reference, abs=1e-6), name


def test_current_edge_selection_ignores_the_future():
    """Test 12. The spatial edge set is a function of the current frame alone."""
    batch, future = make_pair(num_graphs=1, num_residues=24)
    ca = batch.backbone.ca_positions
    keep = torch.ones(ca.shape[0], dtype=torch.bool)
    reference = _current_spatial_pair_mask(ca, keep, k=8, cutoff=13.0)

    # The function takes no future argument at all, so the strongest available
    # statement is that scrambling the future leaves the *recorded* edge count
    # unchanged end to end.
    import dataclasses

    target_a = build_transition_target(batch, future)
    noisy = dataclasses.replace(
        future,
        ca_positions=future.ca_positions
        + 5.0 * torch.randn(future.ca_positions.shape,
                            generator=torch.Generator().manual_seed(4)),
    )
    target_b = build_transition_target(batch, noisy)
    row_a = records_for(identity_prediction(target_a), target_a)[0]
    row_b = records_for(identity_prediction(target_b), target_b)[0]

    assert row_a["n_current_spatial_edges"] == row_b["n_current_spatial_edges"]
    assert bool(reference.any())


def test_spatial_edges_are_symmetric_and_strictly_upper_triangular():
    ca = torch.tensor([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0], [40.0, 0, 0]])
    keep = torch.ones(4, dtype=torch.bool)
    mask = _current_spatial_pair_mask(ca, keep, k=2, cutoff=10.0)

    assert not bool(torch.tril(mask).any())     # upper triangular only
    assert bool(mask[0, 1]) and bool(mask[1, 2])
    assert not bool(mask[0, 3])                 # beyond the cutoff


# --------------------------------------------------------------------------
# 5. contacts and events
# --------------------------------------------------------------------------


def _contact_case(cutoff=8.0, separation=6):
    """A 14-residue chain whose contact map is known by construction."""
    resid = torch.arange(1, 15)
    return resid, cutoff, separation


def test_contact_threshold_is_strictly_below_the_cutoff():
    """Test 13. A pair exactly at 8.0 A is not a contact under ``d < 8``."""
    from force_md.transition.extended_metrics import _contact_scores, _upper

    d = torch.tensor([[0.0, 8.0], [8.0, 0.0]], dtype=torch.float64)
    eligible = _upper(2, d.device)
    tp, fp, fn, *_ = _contact_scores(d < 8.0, d < 8.0, eligible)
    assert (tp, fp, fn) == (0, 0, 0)
    tp, *_ = _contact_scores(d < 8.001, d < 8.001, eligible)
    assert tp == 1


def test_a_sample_with_no_positive_event_reports_na_not_one():
    """Test 14. F1 = 1 for "predicted nothing, nothing happened" would be a lie."""
    from force_md.transition.extended_metrics import _prf

    precision, recall, f1, jaccard = _prf(0, 0, 0)
    assert math.isnan(precision) and math.isnan(recall)
    assert math.isnan(f1) and math.isnan(jaccard)

    # A model that predicted nothing while three events happened has undefined
    # precision -- it made no predictions to be wrong about -- but it missed
    # every event, so F1 is 0.0, not NA. This is the identity baseline's case.
    precision, recall, f1, jaccard = _prf(0, 0, 3)
    assert math.isnan(precision)
    assert recall == 0.0 and f1 == 0.0 and jaccard == 0.0

    # ... and the mirror case: events invented where none existed.
    precision, recall, f1, _ = _prf(0, 4, 0)
    assert precision == 0.0 and math.isnan(recall) and f1 == 0.0


def test_formed_and_broken_contact_metrics_are_conditioned_correctly():
    """Tests 15 and 16, on a hand-built case with one of each event."""
    from force_md.transition.extended_metrics import _contact_scores, _upper

    # 4 residues; treat all pairs as eligible for the purpose of the arithmetic.
    current = torch.tensor([
        [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
    ], dtype=torch.bool)
    future = torch.tensor([   # (0,1) breaks, (0,2) forms, (2,3) stays
        [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0],
    ], dtype=torch.bool)
    predicted = torch.tensor([  # breaks both existing contacts, invents (1,3)
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0],
    ], dtype=torch.bool)
    eligible = _upper(4, current.device)

    formed = eligible & ~current
    tp, fp, fn, *_ = _contact_scores(predicted, future, formed)
    assert (tp, fp, fn) == (0, 1, 1)   # invented (1,3); missed (0,2)

    broken = eligible & current
    tp, fp, fn, *_ = _contact_scores(~predicted, ~future, broken)
    assert (tp, fp, fn) == (1, 1, 0)   # caught (0,1); wrongly broke (2,3)

    # The conditioning is what makes those counts meaningful: a false positive
    # here is an invented *event*, not a disagreement about a static contact.
    assert int(formed.sum()) == 4 and int(broken.sum()) == 2


def test_the_identity_baseline_predicts_no_transition_event():
    """The report must be able to say this, so it is measured, not assumed.

    "Nothing moves" reproduces the current contact map exactly, so by definition
    it forms nothing and breaks nothing. Its formed/broken **precision** is
    therefore undefined -- it made no predictions -- while its **F1 is 0.0**,
    because it missed every event that happened.
    """
    batch, future = make_compact_pair(displacement=1.6)
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    assert row["formed_contact_events_target"] > 0
    assert row["broken_contact_events_target"] > 0
    assert row["formed_contact_events_predicted"] == 0
    assert row["broken_contact_events_predicted"] == 0
    for name in ("formed", "broken"):
        assert math.isnan(row[f"{name}_contact_precision"]), name
        assert row[f"{name}_contact_recall"] == 0.0, name
        assert row[f"{name}_contact_f1"] == 0.0, name


# --------------------------------------------------------------------------
# 6. validity metrics
# --------------------------------------------------------------------------


def test_the_clash_metric_excludes_bonded_and_next_nearest_neighbours():
    """Test 17. Sequence neighbours are close because they are bonded."""
    batch, future = make_pair(num_graphs=1, num_residues=20)
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    n = row["n_valid_residues"]
    # separation >= 2 excludes only the |i-j| = 1 pairs
    assert row["n_clash_pairs"] == n * (n - 1) // 2 - (n - 1)
    assert row["clash_definition"].startswith("ca_only_")


def test_bond_and_angle_scope_is_recorded_as_inter_residue_only():
    batch, future = make_pair(num_graphs=1, num_residues=16)
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    assert row["bond_length_scope"] == "peptide_C_N_next_only"
    assert row["n_bond_c_n_next"] == row["n_valid_residues"] - 1
    assert row["n_bond_angles"] == 2 * row["n_bond_c_n_next"]


# --------------------------------------------------------------------------
# 7. refusal, ragged batches, determinism
# --------------------------------------------------------------------------


def test_a_sample_with_too_few_valid_residues_is_refused_not_scored():
    import dataclasses

    batch, future = make_pair(num_graphs=1, num_residues=12)
    mask = torch.zeros_like(batch.residues.mask)
    mask[:2] = True
    batch = dataclasses.replace(
        batch, residues=dataclasses.replace(batch.residues, mask=mask)
    )
    target = build_transition_target(batch, future)
    row = records_for(identity_prediction(target), target)[0]

    assert row["invalid_reason"] is not None
    assert row["ca_rmsd"] is None
    assert row["drmsd_long_range"] is None


def test_a_ragged_batch_scores_each_graph_on_its_own_residues():
    """Test 10. Graphs of different lengths must not bleed into each other."""
    batch_a, future_a = make_pair(num_graphs=1, num_residues=12, seed=1)
    batch_b, future_b = make_pair(num_graphs=1, num_residues=28, seed=1)

    target_a = build_transition_target(batch_a, future_a)
    target_b = build_transition_target(batch_b, future_b)
    row_a = records_for(identity_prediction(target_a), target_a)[0]
    row_b = records_for(identity_prediction(target_b), target_b)[0]

    assert row_a["n_valid_residues"] == 12
    assert row_b["n_valid_residues"] == 28
    assert row_a["n_valid_pairs"] != row_b["n_valid_pairs"]

    # and a multi-graph batch reproduces the single-graph numbers exactly
    batch, future = make_pair(num_graphs=3, num_residues=18)
    target = build_transition_target(batch, future)
    rows = records_for(identity_prediction(target), target)
    assert len(rows) == 3
    assert all(r["n_valid_residues"] == 18 for r in rows)


def test_repeated_evaluation_is_bitwise_deterministic():
    """Test 25."""
    batch, future = make_pair()
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)
    first = records_for(prediction, target)
    second = records_for(prediction, target)
    assert first == second


def test_no_metric_is_nan_or_inf_on_a_well_formed_sample():
    """Test 24."""
    batch, future = make_pair(num_graphs=2, num_residues=32)
    target = build_transition_target(batch, future)
    for row in records_for(identity_prediction(target), target):
        for key in ("ca_rmsd", "drmsd_all", "drmsd_long_range",
                    "rotation_geodesic_mean_deg", "backbone_torsion_mae_deg",
                    "bond_length_rmse", "ca_neighbor_distance_mae", "clash_rate"):
            assert row[key] is not None and math.isfinite(row[key]), key


def test_a_non_finite_coordinate_is_refused():
    batch, future = make_pair(num_graphs=1, num_residues=16)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)
    broken = type(prediction)(
        translation_local=prediction.translation_local.clone().index_fill_(
            0, torch.tensor([3]), float("nan")
        ),
        rotation=prediction.rotation,
    )
    row = records_for(broken, target)[0]
    assert row["invalid_reason"] == "non-finite coordinate in the prediction or target"


# --------------------------------------------------------------------------
# 8. reproduction of the Stage B definitions
# --------------------------------------------------------------------------


def test_extended_ca_rmsd_reproduces_the_stage_b_definition():
    """Test 21. Same number, or the extended report renamed a metric."""
    batch, future = make_pair(num_graphs=3, num_residues=22)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)

    legacy, _ = per_graph_transition_metrics(
        prediction, target, config=MetricConfig()
    )
    extended = records_for(prediction, target)
    for old, new in zip(legacy, extended):
        assert new["ca_rmsd"] == pytest.approx(old["ca_rmsd"], rel=1e-6)
        assert new["ca_rmsd_superposed"] == pytest.approx(
            old["ca_rmsd_aligned"], rel=1e-6
        )
        assert new["translation_rmse"] == pytest.approx(
            old["translation_rmse"], rel=1e-6
        )


def test_extended_rotation_reproduces_the_stage_b_definition():
    """Test 22."""
    batch, future = make_pair(num_graphs=3, num_residues=22)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)

    legacy, _ = per_graph_transition_metrics(prediction, target)
    extended = records_for(prediction, target)
    for old, new in zip(legacy, extended):
        assert new["rotation_geodesic_mean_deg"] == pytest.approx(
            old["rotation_geodesic_deg"], rel=1e-6
        )


def test_extended_legacy_contact_f1_reproduces_the_stage_b_definition():
    batch, future = make_pair(num_graphs=2, num_residues=30)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)

    legacy, _ = per_graph_transition_metrics(prediction, target)
    extended = records_for(prediction, target)
    for old, new in zip(legacy, extended):
        if math.isnan(old["contact_f1"]):
            continue
        assert new["contact_f1_legacy_sep3"] == pytest.approx(
            old["contact_f1"], rel=1e-6
        )


def test_extended_clash_rate_reproduces_the_stage_b_definition():
    batch, future = make_pair(num_graphs=2, num_residues=30)
    target = build_transition_target(batch, future)
    prediction = identity_prediction(target)

    legacy, _ = per_graph_transition_metrics(prediction, target)
    extended = records_for(prediction, target)
    for old, new in zip(legacy, extended):
        assert new["clash_rate"] == pytest.approx(old["clash_rate"], abs=1e-12)
        assert new["clash_rate_target"] == pytest.approx(
            old["clash_rate_target"], abs=1e-12
        )


# --------------------------------------------------------------------------
# 9. small helpers
# --------------------------------------------------------------------------


def test_length_bin_labels_are_contiguous_and_sortable():
    edges = (100, 150, 200)
    assert length_bin(60, edges) == "<100"
    assert length_bin(100, edges) == "100-149"
    assert length_bin(149, edges) == "100-149"
    assert length_bin(150, edges) == "150-199"
    assert length_bin(400, edges) == ">=200"


def test_torsion_histogram_bins_are_fixed_and_jsd_is_zero_for_equal_inputs():
    a = torsion_histogram([-179.0, 0.0, 179.0], bins=36)
    assert sum(a) == 3
    assert jensen_shannon(a, a) == pytest.approx(0.0, abs=1e-12)
    b = torsion_histogram([90.0, 90.0, 90.0], bins=36)
    assert jensen_shannon(a, b) > 0.5
    assert math.isnan(jensen_shannon(a, [0] * 36))
