"""SO(3), proper-rotation Kabsch and backbone torsions (Phase 1.5, Checkpoint 2).

Every function here has a region where the obvious implementation is wrong, and
these tests are aimed at those regions rather than at the easy middle:

* the geodesic angle is checked against **analytically known** angles, not against
  a second implementation of the same formula. The first implementation here was
  off by a factor of two in ``sin(theta)`` and was exact at 0, 90 and 180 degrees
  -- every round number one would spot-check -- while reporting a 30 degree
  rotation as 49.1;
* the log map is checked within 0.001 degrees of a half turn, where the
  antisymmetric part carries no axis at all;
* Kabsch is fed a **mirrored** structure, where the reflection-including solution
  fits perfectly and is the wrong answer for a chiral molecule;
* the dihedral is checked against the canonical alpha-helix values, which are
  signed, so a mirrored convention fails.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.geometry import (  # noqa: E402
    MIN_CORRESPONDENCES,
    backbone_torsions,
    dihedral_angle,
    is_proper_rotation,
    kabsch_rotation,
    random_rotation_matrix,
    random_rotation_of_angle,
    relative_rotation,
    rotation_from_6d,
    rotation_geodesic_angle,
    rotation_to_6d,
    sequence_neighbours,
    so3_exp_map,
    so3_log_map,
    wrap_to_pi,
)

from conftest import dihedral, reference_dihedral_case  # noqa: E402

DEG = [0.0, 1e-4, 0.5, 5.0, 30.0, 45.0, 90.0, 120.0, 150.0, 179.0, 179.999, 180.0]


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------
# geodesic angle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", DEG)
def test_geodesic_angle_matches_a_known_rotation(degrees):
    rotation = random_rotation_of_angle(math.radians(degrees), generator=generator())
    measured = float(torch.rad2deg(rotation_geodesic_angle(rotation)))
    assert measured == pytest.approx(degrees, abs=1e-9)


def test_geodesic_angle_of_the_identity_is_exactly_zero():
    eye = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    assert torch.equal(rotation_geodesic_angle(eye), torch.zeros(4, dtype=torch.float64))


@pytest.mark.parametrize("degrees", [5.0, 60.0, 150.0])
def test_geodesic_distance_is_symmetric_and_left_invariant(degrees):
    g = generator(1)
    a = random_rotation_matrix(g, dtype=torch.float64)
    delta = random_rotation_of_angle(math.radians(degrees), generator=g)
    b = a @ delta
    forward = rotation_geodesic_angle(a, b)
    backward = rotation_geodesic_angle(b, a)
    assert float(torch.rad2deg(forward)) == pytest.approx(degrees, abs=1e-9)
    assert float(forward) == pytest.approx(float(backward), abs=1e-12)
    # left-invariance: pre-multiplying both by the same rotation changes nothing
    q = random_rotation_matrix(g, dtype=torch.float64)
    assert float(rotation_geodesic_angle(q @ a, q @ b)) == pytest.approx(
        float(forward), abs=1e-12
    )


def test_relative_rotation_uses_right_multiplication():
    g = generator(2)
    source = random_rotation_matrix(g, dtype=torch.float64)
    destination = random_rotation_matrix(g, dtype=torch.float64)
    rel = relative_rotation(source, destination)
    assert torch.allclose(source @ rel, destination, atol=1e-12)


# --------------------------------------------------------------------------
# log / exp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", DEG)
def test_log_exp_round_trip(degrees):
    rotation = random_rotation_of_angle(math.radians(degrees), generator=generator(3))
    vector = so3_log_map(rotation)
    assert float(torch.rad2deg(vector.norm())) == pytest.approx(degrees, abs=1e-9)
    # Near a half turn the axis branch is an approximation; everywhere else this
    # is exact to machine precision.
    tolerance = 1e-4 if degrees > 179.99 else 1e-10
    assert torch.allclose(so3_exp_map(vector), rotation, atol=tolerance)


def test_log_map_of_the_identity_is_zero_and_finite():
    vector = so3_log_map(torch.eye(3, dtype=torch.float64).expand(3, 3, 3))
    assert torch.all(torch.isfinite(vector))
    assert float(vector.abs().max()) == 0.0


def test_exp_map_of_zero_is_the_identity():
    rotation = so3_exp_map(torch.zeros(5, 3, dtype=torch.float64))
    assert torch.allclose(rotation, torch.eye(3, dtype=torch.float64).expand(5, 3, 3))
    assert bool(is_proper_rotation(rotation).all())


def test_exp_map_produces_proper_rotations():
    vectors = torch.randn(32, 3, dtype=torch.float64, generator=generator(4)) * 2.0
    assert bool(is_proper_rotation(so3_exp_map(vectors)).all())


# --------------------------------------------------------------------------
# 6D chart
# --------------------------------------------------------------------------


def test_6d_round_trip_is_exact():
    g = generator(5)
    rotations = torch.stack([random_rotation_matrix(g, dtype=torch.float64) for _ in range(8)])
    assert torch.allclose(rotation_from_6d(rotation_to_6d(rotations)), rotations, atol=1e-12)


def test_6d_maps_arbitrary_vectors_to_proper_rotations():
    """The point of the chart: no input can produce a reflection or a NaN."""
    vectors = torch.randn(64, 6, dtype=torch.float64, generator=generator(6)) * 10.0
    rotations = rotation_from_6d(vectors)
    assert bool(is_proper_rotation(rotations).all())
    assert torch.all(torch.isfinite(rotations))


def test_is_proper_rotation_rejects_a_reflection():
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    assert not bool(is_proper_rotation(reflection))
    assert bool(is_proper_rotation(torch.eye(3, dtype=torch.float64)))


# --------------------------------------------------------------------------
# Kabsch
# --------------------------------------------------------------------------


def test_kabsch_recovers_a_known_rigid_motion():
    g = generator(7)
    reference = torch.randn(24, 3, dtype=torch.float64, generator=g)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    translation = torch.tensor([12.0, -5.0, 3.0], dtype=torch.float64)
    mobile = reference @ rotation.T + translation

    batch_index = torch.zeros(24, dtype=torch.int64)
    alignment = kabsch_rotation(mobile, reference, batch_index, 1)
    assert torch.allclose(alignment.rotation[0], rotation.T, atol=1e-10)
    assert torch.allclose(alignment.apply(mobile, batch_index), reference, atol=1e-10)


def test_kabsch_never_returns_a_reflection():
    """A mirrored structure fits perfectly *with* a reflection. It must not use one."""
    g = generator(8)
    reference = torch.randn(20, 3, dtype=torch.float64, generator=g)
    mirrored = reference.clone()
    mirrored[:, 0] *= -1

    alignment = kabsch_rotation(mirrored, reference, torch.zeros(20, dtype=torch.int64), 1)
    assert float(torch.linalg.det(alignment.rotation[0])) == pytest.approx(1.0, abs=1e-10)
    assert bool(is_proper_rotation(alignment.rotation).all())
    # and therefore it cannot fit the mirror image
    residual = (alignment.apply(mirrored, torch.zeros(20, dtype=torch.int64)) - reference)
    assert float(residual.norm(dim=-1).mean()) > 0.1


def test_kabsch_fits_each_graph_independently():
    g = generator(9)
    reference = torch.randn(30, 3, dtype=torch.float64, generator=g)
    batch_index = torch.repeat_interleave(torch.arange(3), 10)
    rotations = [random_rotation_matrix(g, dtype=torch.float64) for _ in range(3)]
    mobile = torch.cat(
        [reference[batch_index == i] @ rotations[i].T + i * 7.0 for i in range(3)]
    )
    alignment = kabsch_rotation(mobile, reference, batch_index, 3)
    for i in range(3):
        assert torch.allclose(alignment.rotation[i], rotations[i].T, atol=1e-10)
    assert torch.allclose(alignment.apply(mobile, batch_index), reference, atol=1e-10)


def test_kabsch_ignores_zero_weighted_rows():
    g = generator(10)
    reference = torch.randn(20, 3, dtype=torch.float64, generator=g)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    mobile = reference @ rotation.T
    poisoned = mobile.clone()
    poisoned[:5] = 1e3  # rows that would wreck the fit if they counted

    weights = torch.ones(20, dtype=torch.float64)
    weights[:5] = 0.0
    batch_index = torch.zeros(20, dtype=torch.int64)
    alignment = kabsch_rotation(poisoned, reference, batch_index, 1, weights=weights)
    assert torch.allclose(alignment.rotation[0], rotation.T, atol=1e-10)
    assert int(alignment.num_used[0]) == 15


def test_kabsch_marks_an_underdetermined_graph_invalid():
    """Two points do not determine a rotation; say so instead of inventing one."""
    reference = torch.randn(2, 3, dtype=torch.float64, generator=generator(11))
    alignment = kabsch_rotation(
        reference, reference, torch.zeros(2, dtype=torch.int64), 1
    )
    assert not bool(alignment.valid[0])
    assert torch.allclose(alignment.rotation[0], torch.eye(3, dtype=torch.float64))
    assert int(alignment.num_used[0]) < MIN_CORRESPONDENCES


def test_kabsch_of_a_structure_onto_itself_is_the_identity():
    points = torch.randn(15, 3, dtype=torch.float64, generator=generator(12))
    batch_index = torch.zeros(15, dtype=torch.int64)
    alignment = kabsch_rotation(points, points, batch_index, 1)
    assert torch.allclose(alignment.rotation[0], torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(alignment.apply(points, batch_index), points, atol=1e-10)


def test_rotating_frames_agrees_with_rebuilding_them_from_aligned_points():
    """``apply_frames`` is the equivariance shortcut; check it against the definition."""
    from force_md.geometry import build_residue_frames

    batch = synthetic_batch([SyntheticSpec(8)], seed=0, plm_dim=32, dtype=torch.float64)
    g = generator(13)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    shift = torch.tensor([4.0, -1.0, 2.0], dtype=torch.float64)
    moved = {
        name: getattr(batch.backbone, name) @ rotation.T + shift
        for name in ("n_positions", "ca_positions", "c_positions")
    }
    batch_index = batch.residues.batch_index
    alignment = kabsch_rotation(
        moved["ca_positions"], batch.backbone.ca_positions, batch_index, batch.num_graphs
    )
    aligned = {k: alignment.apply(v, batch_index) for k, v in moved.items()}
    rebuilt = build_residue_frames(
        aligned["n_positions"], aligned["ca_positions"], aligned["c_positions"]
    )
    moved_frames = build_residue_frames(
        moved["n_positions"], moved["ca_positions"], moved["c_positions"]
    )
    shortcut = alignment.apply_frames(moved_frames.rotation, batch_index)
    assert torch.allclose(rebuilt.rotation, shortcut, atol=1e-10)


# --------------------------------------------------------------------------
# torsions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", [-179.0, -90.0, -1.0, 0.0, 1.0, 90.0, 179.0])
def test_dihedral_matches_the_reference_convention(degrees):
    """Pin the vectorised implementation to the self-tested one in conftest."""
    p0, p1, p2, p3 = reference_dihedral_case(degrees)
    batched = float(
        torch.rad2deg(dihedral_angle(p0[None], p1[None], p2[None], p3[None]))[0]
    )
    assert batched == pytest.approx(dihedral(p0, p1, p2, p3), abs=1e-4)
    assert batched == pytest.approx(degrees, abs=1e-4)


def test_backbone_torsions_of_the_synthetic_helix_are_canonical():
    """phi = -57, psi = -47 is a right-handed alpha helix. Mirrored gives +57/+47."""
    batch = synthetic_batch([SyntheticSpec(10)], seed=0, plm_dim=32, dtype=torch.float64)
    previous, following = sequence_neighbours(
        batch.residues.batch_index, batch.residues.chain_index,
        batch.residues.resid_original,
    )
    phi, psi, phi_valid, psi_valid = backbone_torsions(
        batch.backbone.n_positions, batch.backbone.ca_positions,
        batch.backbone.c_positions, previous, following,
    )
    assert int(phi_valid.sum()) == 9 and int(psi_valid.sum()) == 9
    assert torch.allclose(
        torch.rad2deg(phi[phi_valid]), torch.full((9,), -57.0, dtype=torch.float64), atol=0.5
    )
    assert torch.allclose(
        torch.rad2deg(psi[psi_valid]), torch.full((9,), -47.0, dtype=torch.float64), atol=0.5
    )


def test_torsions_are_invariant_under_a_global_rigid_motion():
    batch = synthetic_batch([SyntheticSpec(9)], seed=1, plm_dim=32, dtype=torch.float64)
    previous, following = sequence_neighbours(
        batch.residues.batch_index, batch.residues.chain_index,
        batch.residues.resid_original,
    )
    args = (batch.backbone.n_positions, batch.backbone.ca_positions,
            batch.backbone.c_positions)
    phi, psi, _, _ = backbone_torsions(*args, previous, following)

    g = generator(14)
    rotation = random_rotation_matrix(g, dtype=torch.float64)
    shift = torch.tensor([100.0, -50.0, 25.0], dtype=torch.float64)
    moved = tuple(x @ rotation.T + shift for x in args)
    phi_moved, psi_moved, _, _ = backbone_torsions(*moved, previous, following)
    assert torch.allclose(phi, phi_moved, atol=1e-9)
    assert torch.allclose(psi, psi_moved, atol=1e-9)


def test_sequence_neighbours_respect_chains_and_numbering_gaps():
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1])
    chain_index = torch.tensor([0, 0, 1, 1, 0, 0])
    resid = torch.tensor([10, 11, 40, 42, 5, 6])
    previous, following = sequence_neighbours(batch_index, chain_index, resid)
    assert following.tolist() == [1, -1, -1, -1, 5, -1]
    assert previous.tolist() == [-1, 0, -1, -1, -1, 4]


def test_sequence_neighbours_may_ignore_numbering_gaps_when_asked():
    batch_index = torch.zeros(3, dtype=torch.int64)
    chain_index = torch.zeros(3, dtype=torch.int64)
    resid = torch.tensor([10, 40, 41])
    _, strict = sequence_neighbours(batch_index, chain_index, resid)
    _, loose = sequence_neighbours(
        batch_index, chain_index, resid, require_contiguous_resid=False
    )
    assert strict.tolist() == [-1, 2, -1]
    assert loose.tolist() == [1, 2, -1]


@pytest.mark.parametrize(
    "raw,expected",
    [(0.0, 0.0), (math.pi, -math.pi), (-math.pi + 1e-9, -math.pi + 1e-9),
     (3 * math.pi, -math.pi), (math.radians(359.0), math.radians(-1.0)),
     (math.radians(-359.0), math.radians(1.0))],
)
def test_wrap_to_pi(raw, expected):
    """The range is the half-open ``[-pi, pi)``: exactly ``pi`` comes back ``-pi``."""
    assert float(wrap_to_pi(torch.tensor(raw, dtype=torch.float64))) == pytest.approx(
        expected, abs=1e-9
    )


def test_wrap_to_pi_preserves_the_magnitude_of_a_half_turn():
    """The property the metrics rely on, which the sign convention does not affect."""
    for raw in (math.pi, -math.pi, 3 * math.pi):
        wrapped = wrap_to_pi(torch.tensor(raw, dtype=torch.float64))
        assert float(wrapped.abs()) == pytest.approx(math.pi, abs=1e-9)


def test_wrapped_difference_of_near_opposite_angles_is_small():
    """179 and -179 degrees are two degrees apart, not 358."""
    a = torch.tensor(math.radians(179.0), dtype=torch.float64)
    b = torch.tensor(math.radians(-179.0), dtype=torch.float64)
    assert float(torch.rad2deg(wrap_to_pi(a - b).abs())) == pytest.approx(2.0, abs=1e-9)
