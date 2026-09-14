"""Equivariance of the heavy-atom stages, checked rather than argued.

Every claim this repository makes about H1b being "equivariant by construction"
reduces to two facts that are easy to state and easy to break silently:

1. everything the refiner *looks at* is invariant under a global rigid motion, and
2. everything it *emits* is expressed in the coarse predicted frame.

Together those force the refined frames to rotate with the input. Separately,
either one can be violated by a one-line change -- a feature block that uses a
global coordinate, a correction applied on the wrong side -- and nothing else in
the pipeline would notice, because the training data is not rotation-augmented
and the metrics are computed in a canonical frame.

Also pins the commutation claim H1a rests on: a Delta-chi rotation and a rigid
frame correction commute, so H1a and H1b can be built in either order and
composed in either order.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.geometry import (  # noqa: E402
    apply_rigid_transform,
    random_rotation_matrix,
    so3_exp_map,
)
from force_md.heavy.backmapping import local_atom_coordinates, place_on_frames  # noqa: E402
from force_md.heavy.builder import rotate_about_axis  # noqa: E402
from force_md.heavy.refiner import (  # noqa: E402
    BackboneFrameConstraintRefiner,
    FrameCorrection,
    RefinerConfig,
    compose,
    reconstruct_from_frames,
    refiner_feature_dim,
    refiner_features,
    soft_overlap_penalty,
)
from force_md.transition import (  # noqa: E402
    TransitionPrediction,
    build_transition_target,
)
from test_transition_targets import (  # noqa: E402
    frame_from_positions,
    generator,
    make_batch,
    perturbed_future,
    rigidly_move,
)

TOLERANCE = 1e-9


def a_prediction(n_residues: int, seed: int = 3) -> TransitionPrediction:
    """A plausible coarse prediction. Local by definition, so a global rigid
    motion of the structure leaves it **numerically unchanged** -- which is why
    the same object is passed to both sides of every test below."""
    g = generator(seed)
    return TransitionPrediction(
        translation_local=torch.randn(n_residues, 3, dtype=torch.float64, generator=g) * 0.3,
        rotation=so3_exp_map(
            torch.randn(n_residues, 3, dtype=torch.float64, generator=g) * 0.1
        ),
    )


def a_motion(seed: int = 7):
    g = generator(seed)
    rotation = random_rotation_matrix(generator=g, dtype=torch.float64)
    return rotation, torch.tensor([3.0, -2.0, 11.0], dtype=torch.float64)


# --------------------------------------------------------------------------
# H1b features
# --------------------------------------------------------------------------


def test_refiner_features_are_invariant_under_a_global_rigid_motion():
    batch = make_batch()
    future = perturbed_future(batch)
    rotation, translation = a_motion()
    moved_batch, moved_future = rigidly_move(batch, future, rotation, translation)

    prediction = a_prediction(batch.num_residues)
    lag = torch.full((batch.num_graphs,), 1.0, dtype=torch.float64)

    before = refiner_features(
        prediction, build_transition_target(batch, future), lag
    )
    after = refiner_features(
        prediction, build_transition_target(moved_batch, moved_future), lag
    )
    assert torch.allclose(before, after, atol=1e-8), (
        "a refiner feature changed under a global rotation, so at least one "
        "block is reading a global coordinate"
    )


def test_refiner_features_have_the_width_the_block_table_declares():
    batch = make_batch()
    target = build_transition_target(batch, perturbed_future(batch))
    features = refiner_features(
        a_prediction(batch.num_residues),
        target,
        torch.full((batch.num_graphs,), 1.0, dtype=torch.float64),
    )
    assert features.shape == (batch.num_residues, refiner_feature_dim())


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_compose_is_equivariant():
    g = generator(11)
    n = 12
    origin = torch.randn(n, 3, dtype=torch.float64, generator=g)
    frames = so3_exp_map(torch.randn(n, 3, dtype=torch.float64, generator=g))
    correction = FrameCorrection(
        delta_translation=torch.randn(n, 3, dtype=torch.float64, generator=g) * 0.1,
        delta_rotation=so3_exp_map(
            torch.randn(n, 3, dtype=torch.float64, generator=g) * 0.05
        ),
    )
    rotation, translation = a_motion()

    plain_origin, plain_rotation = compose(origin, frames, correction)
    moved_origin, moved_rotation = compose(
        origin @ rotation.T + translation, rotation @ frames, correction
    )

    assert torch.allclose(moved_origin, plain_origin @ rotation.T + translation, atol=TOLERANCE)
    assert torch.allclose(moved_rotation, rotation @ plain_rotation, atol=TOLERANCE)


def test_compose_keeps_the_rotation_proper():
    """A correction that drifted off SO(3) would mirror the structure."""
    g = generator(13)
    n = 20
    frames = so3_exp_map(torch.randn(n, 3, dtype=torch.float64, generator=g))
    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(), RefinerConfig()
    ).double()
    features = torch.randn(n, refiner_feature_dim(), dtype=torch.float64, generator=g)
    _, rotation = compose(
        torch.zeros(n, 3, dtype=torch.float64), frames, refiner(features)
    )
    assert torch.allclose(
        torch.linalg.det(rotation), torch.ones(n, dtype=torch.float64), atol=1e-10
    )


def test_an_untrained_refiner_is_the_identity():
    """Zero-init means H1b starts exactly at the coarse prediction, so every
    later number is a departure from it and not from a random pose."""
    g = generator(17)
    n = 9
    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(), RefinerConfig()
    ).double()
    with torch.no_grad():
        correction = refiner(
            torch.randn(n, refiner_feature_dim(), dtype=torch.float64, generator=g)
        )
    assert float(correction.delta_translation.abs().max()) == 0.0
    identity = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
    assert torch.allclose(correction.delta_rotation, identity, atol=TOLERANCE)


def test_the_correction_caps_bound_the_magnitude_not_the_components():
    """A per-component ``tanh`` lets the norm reach ``sqrt(3)`` times the cap.

    Regression: it did. The first H1b overfit probe reported a maximum
    correction of 1.72 A and 25.96 deg against caps of 1.0 A and 15 deg -- both
    exactly ``sqrt(3)`` over, the signature of capping each axis instead of the
    length. The caps are set against measured quantities, so the factor matters.
    """
    from force_md.geometry import rotation_geodesic_angle

    config = RefinerConfig(max_translation=1.0, max_rotation_rad=math.radians(15.0))
    refiner = BackboneFrameConstraintRefiner(refiner_feature_dim(), config).double()
    with torch.no_grad():
        # Drive the head hard and along the diagonal, where the bug shows.
        refiner.out.weight.fill_(0.0)
        refiner.out.bias.copy_(
            torch.tensor([50.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=torch.float64)
        )
        correction = refiner(torch.ones(8, refiner_feature_dim(), dtype=torch.float64))

    assert float(correction.delta_translation.norm(dim=-1).max()) <= 1.0 + 1e-9
    angle = torch.rad2deg(rotation_geodesic_angle(correction.delta_rotation))
    assert float(angle.max()) <= 15.0 + 1e-6


def test_the_cap_is_the_identity_below_the_cap():
    """``|v_out| = min(|v|, cap)`` -- untouched inside the ball.

    The point of clipping rather than squashing. A ``tanh`` on the norm also
    respects the bound but compresses everything beneath it (76% of the cap at
    ``|v| = 1``), so the head cannot use the range it was given and the gradient
    thins out exactly where the correction matters.
    """
    cap = BackboneFrameConstraintRefiner._cap_norm
    inside = torch.tensor([[0.3, -0.4, 0.0], [0.0, 0.0, 0.6]], dtype=torch.float64)
    assert torch.allclose(cap(inside, 1.0), inside, atol=1e-12)

    outside = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)  # norm 5
    capped = cap(outside, 1.0)
    assert torch.allclose(capped.norm(dim=-1), torch.ones(1, dtype=torch.float64))
    # direction preserved
    assert torch.allclose(capped[0] * 5.0, outside[0], atol=1e-12)

    assert torch.equal(
        cap(torch.zeros(4, 3, dtype=torch.float64), 1.0),
        torch.zeros(4, 3, dtype=torch.float64),
    )


def test_the_zero_init_refiner_still_has_a_gradient():
    """Zero output at step 0 must not mean zero gradient at step 0.

    Regression, and the nastier half of the cap fix: writing the norm cap as
    ``tanh(n) / clamp(n, 1e-12)`` makes the scale ``0/1e-12 = 0`` at the origin,
    so a zero-init head sits at exactly zero correction forever. It trains
    without error, the loss even falls (the coarse arm is doing the work), and
    the only visible symptom is ``|dt| 0.0000 A`` on every logged step.
    """
    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(), RefinerConfig()
    ).double()
    features = torch.randn(16, refiner_feature_dim(), dtype=torch.float64)
    correction = refiner(features)
    # A target that wants any motion at all.
    (correction.delta_translation.pow(2).sum()
     - correction.delta_translation.sum()).backward()

    gradients = [
        p.grad.abs().max() for p in refiner.parameters() if p.grad is not None
    ]
    assert gradients, "no parameter received a gradient"
    assert float(max(gradients)) > 0.0, (
        "the zero-init refiner has no gradient, so it can never leave the "
        "identity: the correction cap has a dead fixed point at the origin"
    )


def test_reconstruct_from_frames_is_equivariant():
    g = generator(19)
    n = 10
    origin = torch.randn(n, 3, dtype=torch.float64, generator=g)
    frames = so3_exp_map(torch.randn(n, 3, dtype=torch.float64, generator=g))
    local_n = torch.randn(n, 3, dtype=torch.float64, generator=g)
    local_c = torch.randn(n, 3, dtype=torch.float64, generator=g)
    rotation, translation = a_motion()

    plain = reconstruct_from_frames(origin, frames, local_n, local_c)
    moved = reconstruct_from_frames(
        origin @ rotation.T + translation, rotation @ frames, local_n, local_c
    )
    for a, b in zip(moved, plain):
        assert torch.allclose(a, b @ rotation.T + translation, atol=TOLERANCE)


# --------------------------------------------------------------------------
# H1a's rotation engine
# --------------------------------------------------------------------------


def test_a_chi_rotation_commutes_with_a_rigid_frame_correction():
    """The claim that lets H1a and H1b be built and applied in either order.

    A Delta-chi rotation turns a residue's atoms about an axis defined by two of
    that residue's **own** atoms. Move the residue rigidly and the axis moves
    with it, so rotating-then-moving and moving-then-rotating land in the same
    place. If this failed, the two stages would have to be trained jointly.
    """
    g = generator(23)
    points = torch.randn(14, 3, dtype=torch.float64, generator=g)
    origin = torch.randn(3, dtype=torch.float64, generator=g).expand(14, 3)
    direction = torch.randn(3, dtype=torch.float64, generator=g).expand(14, 3)
    angle = torch.full((14,), 0.7, dtype=torch.float64)
    rotation, translation = a_motion()

    rotate_then_move = (
        rotate_about_axis(points, origin, direction, angle) @ rotation.T + translation
    )
    move_then_rotate = rotate_about_axis(
        points @ rotation.T + translation,
        origin @ rotation.T + translation,
        direction @ rotation.T,
        angle,
    )
    assert torch.allclose(rotate_then_move, move_then_rotate, atol=1e-12)


def test_local_atom_coordinates_are_invariant_and_round_trip():
    batch = make_batch()
    future = perturbed_future(batch)
    rotation, translation = a_motion()
    moved_batch, _ = rigidly_move(batch, future, rotation, translation)

    target = build_transition_target(batch, future)
    moved_target = build_transition_target(
        moved_batch, frame_from_positions(moved_batch, moved_batch.atoms.positions)
    )
    a2r = batch.atoms.atom_to_residue

    plain = local_atom_coordinates(
        batch.atoms.positions, target.current_frames.rotation, target.current_ca, a2r
    )
    moved = local_atom_coordinates(
        moved_batch.atoms.positions,
        moved_target.current_frames.rotation,
        moved_target.current_ca,
        a2r,
    )
    assert torch.allclose(plain, moved, atol=1e-8)

    back = place_on_frames(
        plain, target.current_frames.rotation, target.current_ca, a2r
    )
    assert torch.allclose(back, batch.atoms.positions, atol=1e-9)


def test_the_overlap_penalty_is_invariant_under_a_rigid_motion():
    g = generator(29)
    n = 40
    positions = torch.randn(n, 3, dtype=torch.float64, generator=g) * 2.0
    radii = torch.full((n,), 1.7, dtype=torch.float64)
    excluded = torch.zeros(n, n, dtype=torch.bool)
    valid = torch.ones(n, dtype=torch.bool)
    rotation, translation = a_motion()

    plain = soft_overlap_penalty(positions, radii, excluded, valid)
    moved = soft_overlap_penalty(
        positions @ rotation.T + translation, radii, excluded, valid
    )
    assert float(plain) > 0.0, "the fixture must actually contain overlaps"
    assert torch.allclose(plain, moved, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refined_frames_rotate_with_the_input_end_to_end(seed: int):
    """Features invariant + correction in the local frame => refined frames
    equivariant. Checked through the actual head, with non-zero weights."""
    batch = make_batch(seed=seed)
    future = perturbed_future(batch, seed=seed + 1)
    rotation, translation = a_motion(seed=seed + 5)
    moved_batch, moved_future = rigidly_move(batch, future, rotation, translation)

    prediction = a_prediction(batch.num_residues, seed=seed + 2)
    lag = torch.full((batch.num_graphs,), 1.0, dtype=torch.float64)
    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(), RefinerConfig()
    ).double()
    with torch.no_grad():  # break the zero-init so the test is not vacuous
        refiner.out.weight.normal_(std=0.05, generator=generator(seed + 9))
        refiner.out.bias.normal_(std=0.05, generator=generator(seed + 10))

    def refined(a_batch, a_future):
        target = build_transition_target(a_batch, a_future)
        origin = target.current_ca + torch.einsum(
            "nij,nj->ni", target.current_frames.rotation, prediction.translation_local
        )
        frames = target.current_frames.rotation @ prediction.rotation
        correction = refiner(refiner_features(prediction, target, lag))
        return compose(origin, frames, correction)

    plain_origin, plain_rotation = refined(batch, future)
    moved_origin, moved_rotation = refined(moved_batch, moved_future)

    assert torch.allclose(
        moved_origin, plain_origin @ rotation.T + translation, atol=1e-7
    )
    assert torch.allclose(moved_rotation, rotation @ plain_rotation, atol=1e-8)
