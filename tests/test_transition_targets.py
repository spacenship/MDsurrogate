"""Transition targets and geometry metrics (Phase 1.5, Checkpoint 2).

The property that matters most here is what the target does **not** contain. A
protein in mdCATH tumbles and diffuses, and over 1-4 ns that motion is several
times larger than the conformational change being modelled. So the central tests
rotate and translate the future structure -- by itself, and together with the
current one -- and require every target and every metric to be unchanged. A
target that moved would be measuring Brownian motion.

The second group checks that the target is a *label*: it is refused whenever the
future rows do not correspond to the current ones, because Kabsch pairs points by
row and would happily return a rotation for a mismatched ordering.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.data.contracts import FrameGeometry  # noqa: E402
from force_md.geometry import (  # noqa: E402
    build_residue_frames,
    random_rotation_matrix,
    rotation_geodesic_angle,
    so3_exp_map,
)
from force_md.transition import (  # noqa: E402
    MetricConfig,
    aggregate_metric_records,
    build_transition_target,
    identity_prediction,
    metric_records,
    per_graph_transition_metrics,
    reconstruct_backbone,
    target_as_prediction,
    transition_metrics,
    TransitionPrediction,
)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_batch(sizes=(8, 6), seed: int = 0):
    return synthetic_batch(
        [SyntheticSpec(n) for n in sizes], seed=seed, plm_dim=32, dtype=torch.float64
    )


def frame_from_positions(batch, positions: torch.Tensor, *, offset: int = 4) -> FrameGeometry:
    """A future frame carrying ``positions``, row-aligned with ``batch``.

    Backbone N/CA/C are gathered from the atom tensor exactly as
    ``link_backbone_to_atom_positions`` does, so the frame is self-consistent.
    """
    from force_md.geometry import frame_atom_indices

    indices, complete = frame_atom_indices(batch)
    safe = indices.clamp(min=0)
    return FrameGeometry(
        positions=positions,
        n_positions=positions[safe[:, 0]],
        ca_positions=positions[safe[:, 1]],
        c_positions=positions[safe[:, 2]],
        frame_valid=batch.backbone.frame_valid & complete,
        atom_batch_index=batch.atoms.batch_index,
        residue_batch_index=batch.residues.batch_index,
        frame_index=batch.frame_index + offset,
    )


def perturbed_future(batch, *, scale: float = 0.3, seed: int = 1, offset: int = 4):
    """A plausible future: every residue rigidly displaced and rotated a little."""
    g = generator(seed)
    n_res = batch.num_residues
    shift = torch.randn(n_res, 3, dtype=torch.float64, generator=g) * scale
    turn = so3_exp_map(torch.randn(n_res, 3, dtype=torch.float64, generator=g) * 0.1)
    a2r = batch.atoms.atom_to_residue
    ca = batch.backbone.ca_positions
    local = batch.atoms.positions - ca[a2r]
    moved = torch.einsum("nij,nj->ni", turn[a2r], local) + ca[a2r] + shift[a2r]
    return frame_from_positions(batch, moved, offset=offset)


def rigidly_move(batch, future, rotation, translation, *, move_future_only: bool = False):
    """Apply a global rigid motion, optionally to the future alone."""
    from force_md.geometry import apply_rigid_transform

    moved_future = dataclasses.replace(
        future,
        positions=future.positions @ rotation.T + translation,
        n_positions=future.n_positions @ rotation.T + translation,
        ca_positions=future.ca_positions @ rotation.T + translation,
        c_positions=future.c_positions @ rotation.T + translation,
    )
    if move_future_only:
        return batch, moved_future
    return apply_rigid_transform(batch, rotation, translation), moved_future


# --------------------------------------------------------------------------
# the target
# --------------------------------------------------------------------------


def test_an_unchanged_future_gives_a_zero_target():
    batch = make_batch()
    future = frame_from_positions(batch, batch.atoms.positions.clone())
    target = build_transition_target(batch, future)

    assert bool(target.valid.all())
    assert float(target.translation_local.abs().max()) < 1e-9
    assert float(torch.rad2deg(rotation_geodesic_angle(target.rotation)).max()) < 1e-6


def test_delta_r_global_is_r_current_times_delta_r_local():
    """The relation every downstream head depends on."""
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    rebuilt = torch.einsum(
        "nij,nj->ni", target.current_frames.rotation, target.translation_local
    )
    assert torch.allclose(rebuilt, target.translation_global, atol=1e-10)


def test_rotation_target_composes_back_to_the_aligned_future_frame():
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    composed = target.current_frames.rotation @ target.rotation
    assert torch.allclose(composed, target.future_frames_aligned.rotation, atol=1e-10)


def test_a_global_rigid_motion_of_the_future_alone_is_removed():
    """Tumbling between the two frames must not appear in the target at all."""
    batch = make_batch()
    future = perturbed_future(batch)
    reference = build_transition_target(batch, future)

    g = generator(2)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    translation = torch.tensor([120.0, -35.0, 8.0], dtype=torch.float64)
    _, moved_future = rigidly_move(batch, future, rotation, translation, move_future_only=True)
    moved = build_transition_target(batch, moved_future)

    assert torch.allclose(moved.translation_local, reference.translation_local, atol=1e-8)
    assert torch.allclose(moved.rotation, reference.rotation, atol=1e-8)


def test_targets_are_invariant_under_a_global_rigid_motion_of_the_pair():
    batch = make_batch()
    future = perturbed_future(batch)
    reference = build_transition_target(batch, future)

    g = generator(3)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    translation = torch.tensor([-60.0, 15.0, 240.0], dtype=torch.float64)
    moved_batch, moved_future = rigidly_move(batch, future, rotation, translation)
    moved = build_transition_target(moved_batch, moved_future)

    assert torch.allclose(moved.translation_local, reference.translation_local, atol=1e-8)
    assert torch.allclose(moved.rotation, reference.rotation, atol=1e-8)
    # the global-frame translation is equivariant, not invariant
    assert torch.allclose(
        moved.translation_global, reference.translation_global @ rotation.T, atol=1e-8
    )


def test_alignment_is_a_proper_rotation():
    batch = make_batch()
    mirrored = batch.atoms.positions.clone()
    mirrored[:, 0] *= -1
    target = build_transition_target(batch, frame_from_positions(batch, mirrored))
    assert torch.allclose(
        torch.linalg.det(target.alignment.rotation),
        torch.ones(batch.num_graphs, dtype=torch.float64),
        atol=1e-9,
    )


def test_reconstruction_recovers_ca_exactly_and_n_c_closely():
    batch = make_batch()
    future = perturbed_future(batch)
    target = build_transition_target(batch, future)
    n, ca, c = reconstruct_backbone(target_as_prediction(target), target)

    index = target.residue_batch_index
    true_ca = target.alignment.apply(future.ca_positions, index)
    true_n = target.alignment.apply(future.n_positions, index)
    assert torch.allclose(ca, true_ca, atol=1e-9)
    # N and C carry the *current* internal geometry, so they are close but not exact.
    assert float((n - true_n).norm(dim=-1).mean()) < 0.5


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_future_with_the_wrong_number_of_residues_is_refused():
    batch = make_batch()
    future = perturbed_future(batch)
    truncated = dataclasses.replace(
        future,
        ca_positions=future.ca_positions[:-1],
        n_positions=future.n_positions[:-1],
        c_positions=future.c_positions[:-1],
        frame_valid=future.frame_valid[:-1],
        residue_batch_index=future.residue_batch_index[:-1],
    )
    with pytest.raises(ValueError, match="residues"):
        build_transition_target(batch, truncated)


def test_a_future_whose_residues_belong_to_other_graphs_is_refused():
    batch = make_batch()
    future = perturbed_future(batch)
    scrambled = dataclasses.replace(
        future, residue_batch_index=torch.flip(future.residue_batch_index, dims=(0,))
    )
    with pytest.raises(ValueError, match="residue_batch_index"):
        build_transition_target(batch, scrambled)


def test_a_future_that_is_not_later_than_the_current_frame_is_refused():
    batch = make_batch()
    future = perturbed_future(batch, offset=0)
    with pytest.raises(ValueError, match="not after the current one"):
        build_transition_target(batch, future)


def test_masked_residues_are_excluded_from_the_target():
    batch = make_batch(sizes=(8,))
    mask = batch.residues.mask.clone()
    mask[2] = False
    batch = dataclasses.replace(
        batch, residues=dataclasses.replace(batch.residues, mask=mask)
    )
    target = build_transition_target(batch, perturbed_future(batch))
    assert not bool(target.valid[2])
    assert float(target.translation_local[2].abs().max()) == 0.0
    assert torch.allclose(target.rotation[2], torch.eye(3, dtype=torch.float64))


def test_a_degenerate_future_frame_invalidates_that_residue():
    batch = make_batch(sizes=(8,))
    future = perturbed_future(batch)
    collapsed = dataclasses.replace(
        future, c_positions=future.c_positions.clone()
    )
    collapsed.c_positions[3] = collapsed.ca_positions[3]  # zero-length e1
    target = build_transition_target(batch, collapsed)
    assert not bool(target.valid[3])
    assert bool(target.valid[[0, 1, 2, 4]].all())


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_a_perfect_prediction_scores_zero_error_and_perfect_contacts():
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    metrics = transition_metrics(target_as_prediction(target), target)

    assert metrics["ca_rmsd"] < 1e-9
    assert metrics["translation_rmse"] < 1e-9
    assert metrics["rotation_geodesic_deg"] < 1e-6
    assert metrics["pair_distance_mae"] < 1e-9
    assert metrics["contact_f1"] == pytest.approx(1.0)
    assert metrics["phi_mae_deg"] < 1e-6
    assert metrics["psi_mae_deg"] < 1e-6


def test_ca_rmsd_and_translation_rmse_are_the_same_quantity():
    """Both are ``|R_cur (t_pred - t_target)|``; the rotation preserves the norm."""
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    metrics = transition_metrics(identity_prediction(target), target)
    assert metrics["ca_rmsd"] == pytest.approx(metrics["translation_rmse"], rel=1e-9)


def test_metrics_are_invariant_under_a_global_rigid_motion():
    batch = make_batch()
    future = perturbed_future(batch)
    target = build_transition_target(batch, future)
    reference = transition_metrics(identity_prediction(target), target)

    g = generator(4)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    translation = torch.tensor([300.0, -90.0, 45.0], dtype=torch.float64)
    moved_batch, moved_future = rigidly_move(batch, future, rotation, translation)
    moved_target = build_transition_target(moved_batch, moved_future)
    moved = transition_metrics(identity_prediction(moved_target), moved_target)

    for key, value in reference.items():
        if value != value:  # NaN
            continue
        assert moved[key] == pytest.approx(value, abs=1e-6, rel=1e-6), key


def test_the_identity_baseline_is_reported_and_relative_to_itself_is_one():
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    metrics = transition_metrics(identity_prediction(target), target)
    for key in ("ca_rmsd", "rotation_geodesic_deg", "pair_distance_mae"):
        assert metrics[f"{key}_identity"] == pytest.approx(metrics[key], rel=1e-9)
        assert metrics[f"{key}_relative"] == pytest.approx(1.0, rel=1e-9)


def test_a_worse_than_nothing_prediction_has_relative_above_one():
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    wrong = TransitionPrediction(
        translation_local=-target.translation_local,
        rotation=target.rotation.transpose(-1, -2),
    )
    metrics = transition_metrics(wrong, target)
    assert metrics["ca_rmsd_relative"] > 1.0
    assert metrics["rotation_geodesic_deg_relative"] > 1.0


def test_pair_distances_and_contacts_react_to_an_expansion():
    """Moving every residue outward leaves Ca RMSD finite but destroys contacts."""
    batch = make_batch(sizes=(12,))
    target = build_transition_target(batch, perturbed_future(batch))
    centre = target.current_ca.mean(dim=0, keepdim=True)
    outward = (target.current_ca - centre) * 0.5
    expanded = TransitionPrediction(
        translation_local=torch.einsum(
            "nji,nj->ni", target.current_frames.rotation, outward
        ),
        rotation=target.rotation,
    )
    metrics = transition_metrics(expanded, target)
    baseline = transition_metrics(identity_prediction(target), target)
    assert metrics["pair_distance_mae"] > baseline["pair_distance_mae"]
    assert metrics["contact_recall"] < baseline["contact_recall"]
    assert metrics["rotation_geodesic_deg"] < 1e-6  # orientation untouched


def test_clash_rate_detects_residues_driven_into_each_other():
    batch = make_batch(sizes=(12,))
    target = build_transition_target(batch, perturbed_future(batch))
    assert transition_metrics(identity_prediction(target), target)["clash_rate"] == 0.0

    collapse = TransitionPrediction(
        translation_local=torch.einsum(
            "nji,nj->ni",
            target.current_frames.rotation,
            (target.current_ca.mean(dim=0, keepdim=True) - target.current_ca) * 0.95,
        ),
        rotation=target.rotation,
    )
    assert transition_metrics(collapse, target)["clash_rate"] > 0.5


def test_a_real_structure_has_no_clashes_by_this_definition():
    """3.6 A between non-neighbouring Ca is below a *bonded* Ca-Ca distance."""
    batch = make_batch(sizes=(14,))
    target = build_transition_target(batch, perturbed_future(batch))
    metrics = transition_metrics(identity_prediction(target), target)
    assert metrics["clash_rate_target"] == 0.0


def test_metrics_ignore_masked_residues():
    batch = make_batch(sizes=(10,))
    future = perturbed_future(batch)
    reference = transition_metrics(
        identity_prediction(build_transition_target(batch, future)),
        build_transition_target(batch, future),
    )

    mask = batch.residues.mask.clone()
    mask[4] = False
    masked_batch = dataclasses.replace(
        batch, residues=dataclasses.replace(batch.residues, mask=mask)
    )
    masked_target = build_transition_target(masked_batch, future)
    masked = transition_metrics(identity_prediction(masked_target), masked_target)

    assert masked["residue_count"] == reference["residue_count"] - 1
    assert masked["ca_rmsd"] != pytest.approx(reference["ca_rmsd"], rel=1e-12)


def test_per_graph_metrics_are_reported_separately():
    batch = make_batch(sizes=(8, 6, 10))
    target = build_transition_target(batch, perturbed_future(batch))
    rows, counts = per_graph_transition_metrics(identity_prediction(target), target)
    assert len(rows) == 3 and counts == [8, 6, 10]
    assert all(row["ca_rmsd"] == row["ca_rmsd"] for row in rows)  # not NaN


def test_records_and_aggregation_split_by_lag_and_report_both_averages():
    batch = make_batch(sizes=(8, 6, 10, 7))
    target = build_transition_target(batch, perturbed_future(batch))
    records = metric_records(
        identity_prediction(target), target,
        domains=["a", "a", "b", "b"], lag_ps=[1000.0, 4000.0, 1000.0, 4000.0],
        split="val",
    )
    assert len(records) == 4
    assert {r["lag_ns"] for r in records} == {1.0, 4.0}

    rows = aggregate_metric_records(records)
    assert len(rows) == 2
    for row in rows:
        assert row["split"] == "val"
        assert row["domain_count"] == 2
        assert row["graph_count"] == 2
        assert row["ca_rmsd_micro"] == row["ca_rmsd_micro"]
        assert row["ca_rmsd_domain_macro"] == row["ca_rmsd_domain_macro"]


def test_domain_macro_and_micro_differ_when_domains_have_different_sizes():
    """The whole reason both are reported: a big protein can carry the micro mean."""
    batch = make_batch(sizes=(4, 20))
    target = build_transition_target(batch, perturbed_future(batch, scale=0.1))
    big_error = TransitionPrediction(
        translation_local=target.translation_local.clone(),
        rotation=target.rotation.clone(),
    )
    big = target.residue_batch_index == 1
    big_error.translation_local[big] += 5.0

    records = metric_records(
        big_error, target, domains=["small", "big"], lag_ps=[1000.0, 1000.0]
    )
    row = aggregate_metric_records(records)[0]
    assert row["ca_rmsd_micro"] > row["ca_rmsd_domain_macro"]


def test_metric_records_refuse_mismatched_metadata():
    batch = make_batch(sizes=(8, 6))
    target = build_transition_target(batch, perturbed_future(batch))
    with pytest.raises(ValueError, match="metadata length mismatch"):
        metric_records(identity_prediction(target), target,
                       domains=["only-one"], lag_ps=[1000.0, 4000.0])


def test_contact_cutoff_is_configurable():
    batch = make_batch(sizes=(12,))
    target = build_transition_target(batch, perturbed_future(batch))
    loose = transition_metrics(
        identity_prediction(target), target,
        config=MetricConfig(contact_cutoff=20.0), with_baseline=False,
    )
    tight = transition_metrics(
        identity_prediction(target), target,
        config=MetricConfig(contact_cutoff=6.0), with_baseline=False,
    )
    assert loose["contact_f1"] != pytest.approx(tight["contact_f1"], rel=1e-6)
