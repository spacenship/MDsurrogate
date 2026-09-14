"""H1a torsion head, H1b frame refiner, and the H2 force arms.

These modules are the ones that will eventually be *trained*, so the tests
concentrate on the properties that make a training run interpretable rather than
on numerical accuracy: an untrained head must be the identity baseline, a
correction must be a proper rotation, the four H2 arms must be the same size, and
no path may read the future.

The equivariance tests matter most. A chi angle is an internal coordinate and
must not change when the molecule is rotated; a force is a vector and must
rotate with it. Getting either backwards produces a model that trains fine and
is wrong everywhere.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from force_md.geometry.frames import random_rotation_matrix  # noqa: E402
from force_md.geometry.so3 import is_proper_rotation  # noqa: E402
from force_md.geometry.torsions import wrap_to_pi  # noqa: E402
from force_md.heavy.atom_force import (  # noqa: E402
    H2_ARMS,
    AtomForceConditioner,
    AtomForceConditionerConfig,
    AtomForcePredictor,
    aggregate_hydrogen_forces,
    heteroscedastic_nll,
    shuffle_forces_within_strata,
)
from force_md.heavy.refiner import (  # noqa: E402
    BackboneFrameConstraintRefiner,
    RefinerConfig,
    compose,
    correction_magnitude,
    current_peptide_geometry,
    peptide_geometry_loss,
    soft_overlap_penalty,
)
from force_md.heavy.torsion_decoder import (  # noqa: E402
    CHI_STATE_EDGES,
    SidechainTorsionHead,
    TorsionHeadConfig,
    chi_state_accuracy_3bin,
    circular_mae,
    circular_mixture_nll,
    von_mises_nll,
)

PARAMETERISATIONS = ("mixture", "state_plus_residual", "von_mises", "sincos")


# --------------------------------------------------------------------------
# H1a -- torsion head
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PARAMETERISATIONS)
def test_an_untrained_torsion_head_predicts_no_change(mode):
    """Zero-init must mean the identity baseline, in every parameterisation.

    ``state_plus_residual`` needed a bias to get here: with tied logits
    ``argmax`` picks bin 0, whose centre is -120 degrees, so an untrained head
    would have turned every torsion by 120 degrees.
    """
    head = SidechainTorsionHead(16, TorsionHeadConfig(parameterisation=mode))
    output = head(torch.randn(11, 16))
    estimate = head.point_estimate(output)
    assert estimate.shape == (11, head.config.max_chi)
    assert torch.allclose(estimate, torch.zeros_like(estimate), atol=1e-6)


@pytest.mark.parametrize("mode", PARAMETERISATIONS)
def test_the_torsion_head_is_invariant_to_the_features_being_scalars(mode):
    """Chi is an internal coordinate; the head must take invariants only.

    Checked structurally: the head's input is a plain feature matrix with no
    notion of orientation, so the same features give the same answer however the
    molecule they came from was posed.
    """
    torch.manual_seed(0)
    head = SidechainTorsionHead(16, TorsionHeadConfig(parameterisation=mode))
    for parameter in head.out.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    features = torch.randn(5, 16)
    first = head.point_estimate(head(features))
    second = head.point_estimate(head(features.clone()))
    assert torch.equal(first, second)


def test_circular_losses_are_finite_at_high_concentration():
    """``i0`` overflows float32 near kappa = 90; a trained head gets there."""
    mean = torch.zeros(4, 2)
    target = torch.zeros(4, 2)
    mask = torch.ones(4, 2, dtype=torch.bool)
    periodicity = torch.ones(4, 2)
    for log_kappa in (0.0, 5.0, 8.0):
        value = von_mises_nll(
            mean, torch.full((4, 2), log_kappa), target, mask, periodicity
        )
        assert torch.isfinite(value), log_kappa


def test_mixture_nll_prefers_the_component_that_matches():
    """A two-mode target is exactly what a single mean cannot represent."""
    log_weight = torch.log(torch.tensor([[[0.5, 0.5]]]))
    mean = torch.tensor([[[0.0, 2.0]]])
    log_concentration = torch.full((1, 1, 2), 2.0)
    mask = torch.ones(1, 1, dtype=torch.bool)
    periodicity = torch.ones(1, 1)

    on_mode = circular_mixture_nll(
        log_weight, mean, log_concentration, torch.tensor([[2.0]]), mask, periodicity
    )
    between = circular_mixture_nll(
        log_weight, mean, log_concentration, torch.tensor([[1.0]]), mask, periodicity
    )
    assert float(on_mode) < float(between)


def test_circular_mae_folds_pi_periodic_torsions():
    mask = torch.ones(1, 1, dtype=torch.bool)
    predicted = torch.zeros(1, 1)
    target = torch.full((1, 1), math.pi)
    assert float(circular_mae(predicted, target, mask, torch.ones(1, 1))) == \
        pytest.approx(math.pi, abs=1e-6)
    assert float(circular_mae(predicted, target, mask, torch.full((1, 1), 2.0))) == \
        pytest.approx(0.0, abs=1e-6)


def test_circular_mae_treats_179_and_minus_179_as_two_degrees():
    mask = torch.ones(1, 1, dtype=torch.bool)
    error = circular_mae(
        torch.full((1, 1), math.radians(179.0)),
        torch.full((1, 1), math.radians(-179.0)),
        mask, torch.ones(1, 1),
    )
    assert math.degrees(float(error)) == pytest.approx(2.0, abs=1e-4)


def test_chi_state_bins_are_fixed_and_named_honestly():
    """Not rotamer recovery: there is no library, so the name says what it is."""
    assert CHI_STATE_EDGES[:2] == (-math.pi / 3.0, math.pi / 3.0)
    predicted = torch.tensor([[-2.0, 0.0, 2.0]])
    target = torch.tensor([[-2.0, 0.0, 2.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert float(chi_state_accuracy_3bin(predicted, target, mask)) == 1.0
    assert float(chi_state_accuracy_3bin(predicted, -target, mask)) < 1.0


def test_torsion_head_gradients_are_finite():
    head = SidechainTorsionHead(16)
    output = head(torch.randn(6, 16, requires_grad=True))
    loss = circular_mixture_nll(
        output["log_weight"], output["mean"], output["log_concentration"],
        torch.rand(6, head.config.max_chi) * 4 - 2,
        torch.ones(6, head.config.max_chi, dtype=torch.bool),
        torch.ones(6, head.config.max_chi),
    )
    loss.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


# --------------------------------------------------------------------------
# H1b -- frame refiner
# --------------------------------------------------------------------------


def test_an_untrained_refiner_is_the_identity_map():
    refiner = BackboneFrameConstraintRefiner(12)
    correction = refiner(torch.randn(8, 12))
    assert torch.allclose(
        correction.delta_translation, torch.zeros(8, 3), atol=1e-7
    )
    assert torch.allclose(
        correction.delta_rotation, torch.eye(3).expand(8, 3, 3), atol=1e-7
    )


def test_zero_correction_reproduces_the_coarse_prediction():
    """Brief §4: the refiner must start from the coarse output exactly."""
    refiner = BackboneFrameConstraintRefiner(12)
    torch.manual_seed(0)
    origin = torch.randn(8, 3)
    rotation = torch.stack([random_rotation_matrix() for _ in range(8)]).float()
    moved_origin, moved_rotation = compose(
        origin, rotation, refiner(torch.randn(8, 12))
    )
    assert torch.allclose(moved_origin, origin, atol=1e-6)
    assert torch.allclose(moved_rotation, rotation, atol=1e-6)


def test_the_correction_is_always_a_proper_rotation_and_respects_its_caps():
    torch.manual_seed(1)
    config = RefinerConfig(max_translation=0.8, max_rotation_rad=math.radians(10.0))
    refiner = BackboneFrameConstraintRefiner(12, config)
    for parameter in refiner.out.parameters():
        torch.nn.init.normal_(parameter, std=3.0)
    correction = refiner(torch.randn(64, 12))
    assert bool(is_proper_rotation(correction.delta_rotation).all())
    magnitude = correction_magnitude(correction)
    assert float(magnitude["correction_translation_norm"].max()) <= (
        math.sqrt(3) * config.max_translation + 1e-5
    )
    assert float(magnitude["correction_rotation_deg"].max()) <= (
        math.degrees(math.sqrt(3) * config.max_rotation_rad) + 1e-3
    )


def test_frame_composition_is_se3_equivariant():
    """Rotate the input pose, and the corrected pose rotates the same way."""
    torch.manual_seed(2)
    refiner = BackboneFrameConstraintRefiner(12)
    for parameter in refiner.out.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    features = torch.randn(10, 12)
    correction = refiner(features)

    origin = torch.randn(10, 3)
    rotation = torch.stack([random_rotation_matrix() for _ in range(10)]).float()
    q = random_rotation_matrix().float()
    translation = torch.tensor([3.0, -1.0, 2.0])

    base_origin, base_rotation = compose(origin, rotation, correction)
    moved_origin, moved_rotation = compose(
        origin @ q.T + translation, q @ rotation, correction
    )
    assert torch.allclose(moved_origin, base_origin @ q.T + translation, atol=1e-4)
    assert torch.allclose(moved_rotation, q @ base_rotation, atol=1e-5)


def test_peptide_geometry_loss_is_zero_against_the_structure_it_measured():
    """The reference is the current frame, so the current frame scores zero."""
    torch.manual_seed(3)
    n, ca, c = (torch.randn(7, 3, dtype=torch.float64) for _ in range(3))
    following = torch.tensor([1, 2, 3, 4, 5, 6, -1])
    geometry = current_peptide_geometry(n, ca, c, following)
    losses = peptide_geometry_loss(geometry, geometry)
    for name, value in losses.items():
        assert float(value) == pytest.approx(0.0, abs=1e-18), name


def test_peptide_geometry_excludes_the_residue_with_no_successor():
    n, ca, c = (torch.randn(5, 3) for _ in range(3))
    following = torch.tensor([1, -1, 3, -1, -1])
    geometry = current_peptide_geometry(n, ca, c, following)
    assert geometry["valid"].tolist() == [True, False, True, False, False]


def test_peptide_geometry_is_invariant_under_a_global_rigid_motion():
    torch.manual_seed(4)
    n, ca, c = (torch.randn(6, 3, dtype=torch.float64) for _ in range(3))
    following = torch.tensor([1, 2, 3, 4, 5, -1])
    q = random_rotation_matrix()
    shift = torch.tensor([5.0, -2.0, 1.0], dtype=torch.float64)
    before = current_peptide_geometry(n, ca, c, following)
    after = current_peptide_geometry(
        n @ q.T + shift, ca @ q.T + shift, c @ q.T + shift, following
    )
    for key in ("bond_c_n", "angle_ca_c_n", "angle_c_n_ca"):
        assert torch.allclose(before[key], after[key], atol=1e-10), key


def test_soft_overlap_penalty_is_zero_when_nothing_overlaps_and_positive_when_it_does():
    positions = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    radii = torch.full((3,), 1.7)
    excluded = torch.zeros(3, 3, dtype=torch.bool)
    valid = torch.ones(3, dtype=torch.bool)
    assert float(soft_overlap_penalty(positions, radii, excluded, valid)) == 0.0

    close = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [20.0, 0, 0]])
    assert float(soft_overlap_penalty(close, radii, excluded, valid)) > 0.0
    # ... and a bonded pair at the same distance is excluded, so it is zero again
    bonded = excluded.clone()
    bonded[0, 1] = bonded[1, 0] = True
    assert float(soft_overlap_penalty(close, radii, bonded, valid)) == 0.0


def test_soft_overlap_penalty_has_a_finite_gradient():
    positions = torch.tensor(
        [[0.0, 0, 0], [2.0, 0, 0]], requires_grad=True
    )
    penalty = soft_overlap_penalty(
        positions, torch.full((2,), 1.7), torch.zeros(2, 2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    penalty.backward()
    assert torch.isfinite(positions.grad).all()
    assert float(positions.grad.abs().sum()) > 0


# --------------------------------------------------------------------------
# H2 -- force arms
# --------------------------------------------------------------------------


def test_all_h2_arms_have_identical_parameter_counts():
    """Brief §6.7. A capacity difference would masquerade as a force effect."""
    counts = {
        arm: AtomForceConditioner(AtomForceConditionerConfig(arm=arm)).parameter_count()
        for arm in H2_ARMS
    }
    assert len(set(counts.values())) == 1, counts
    assert all(v > 0 for v in counts.values())


def test_the_geometry_control_still_has_a_force_encoder():
    """Removing it would make H2G a smaller model, not a controlled one."""
    control = AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2G_atom_geometry_control")
    )
    assert any(p.requires_grad for p in control.force_encoder.parameters())
    fed = control.force_input(
        local_force=None, log_variance=None, local_torque=None,
        n_atoms=5, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert fed.shape == (5, 7)
    assert float(fed.abs().sum()) == 0.0     # including the has_force flag


def test_the_has_force_flag_separates_no_information_from_zero_force():
    oracle = AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2O_current_gt_atom_force_oracle")
    )
    zero_force = oracle.force_input(
        local_force=torch.zeros(4, 3), log_variance=None, local_torque=None,
        n_atoms=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    control = AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2G_atom_geometry_control")
    ).force_input(
        local_force=None, log_variance=None, local_torque=None,
        n_atoms=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert not torch.equal(zero_force, control)
    assert float(zero_force[:, 6].min()) == 1.0
    assert float(control[:, 6].max()) == 0.0


def test_only_the_oracle_and_the_shuffled_control_are_non_deployable():
    assert not AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2O_current_gt_atom_force_oracle")
    ).deployable
    assert not AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2S_shuffled_force_control")
    ).deployable
    assert AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2P_predicted_atom_force")
    ).deployable
    assert AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2G_atom_geometry_control")
    ).deployable


def test_the_shuffled_control_refuses_to_run_without_strata():
    """Shuffling across the whole batch would not preserve the distribution."""
    arm = AtomForceConditioner(
        AtomForceConditionerConfig(arm="H2S_shuffled_force_control")
    )
    with pytest.raises(ValueError, match="needs strata"):
        arm.force_input(
            local_force=torch.randn(6, 3), log_variance=None, local_torque=None,
            n_atoms=6, device=torch.device("cpu"), dtype=torch.float32,
        )


def test_shuffling_preserves_the_distribution_within_every_stratum():
    torch.manual_seed(5)
    forces = torch.randn(80, 3)
    strata = torch.arange(80) % 5
    shuffled = shuffle_forces_within_strata(forces, strata, seed=11)
    for value in torch.unique(strata):
        mask = strata == value
        assert torch.allclose(
            forces[mask].norm(dim=-1).sort().values,
            shuffled[mask].norm(dim=-1).sort().values,
        )
    assert int((forces != shuffled).any(-1).sum()) > 50   # correspondence destroyed


def test_shuffling_is_reproducible_for_a_fixed_seed():
    forces = torch.randn(40, 3)
    strata = torch.zeros(40, dtype=torch.int64)
    assert torch.equal(
        shuffle_forces_within_strata(forces, strata, seed=3),
        shuffle_forces_within_strata(forces, strata, seed=3),
    )
    assert not torch.equal(
        shuffle_forces_within_strata(forces, strata, seed=3),
        shuffle_forces_within_strata(forces, strata, seed=4),
    )


def test_hydrogen_aggregation_conserves_total_force():
    """Brief §6.6. A partition must not lose or duplicate anything."""
    torch.manual_seed(6)
    atomic_number = torch.tensor([6, 1, 1, 7, 1, 8, 6, 1])
    parent = torch.tensor([-1, 0, 0, -1, 3, -1, -1, 6])
    raw_to_batch = torch.tensor([0, -1, -1, 1, -1, 2, 3, -1])
    forces = torch.randn(8, 3, dtype=torch.float64)
    positions = torch.randn(8, 3, dtype=torch.float64)

    aggregated = aggregate_hydrogen_forces(forces, positions, parent, raw_to_batch)
    assert torch.allclose(aggregated.force.sum(0), forces.sum(0), atol=1e-12)
    assert aggregated.n_hydrogens.tolist() == [2, 1, 0, 1]


def test_heavy_only_mode_drops_hydrogen_force_and_says_so():
    forces = torch.ones(4, 3, dtype=torch.float64)
    positions = torch.zeros(4, 3, dtype=torch.float64)
    parent = torch.tensor([-1, 0, -1, 2])
    raw_to_batch = torch.tensor([0, -1, 1, -1])
    aggregated = aggregate_hydrogen_forces(
        forces, positions, parent, raw_to_batch, mode="heavy_only"
    )
    assert aggregated.mode == "heavy_only"
    assert aggregated.torque is None
    assert float(aggregated.force.sum()) == pytest.approx(6.0)   # 2 heavy atoms only


def test_hydrogen_torque_is_computed_about_the_parent():
    forces = torch.tensor(
        [[0.0, 0, 0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    positions = torch.tensor([[0.0, 0, 0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    parent = torch.tensor([-1, 0])
    raw_to_batch = torch.tensor([0, -1])
    aggregated = aggregate_hydrogen_forces(
        forces, positions, parent, raw_to_batch,
        mode="heavy_plus_bonded_hydrogen_and_torque",
    )
    # r x F with r = +x, F = +y gives +z
    assert torch.allclose(
        aggregated.torque[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )


def test_a_hydrogen_whose_parent_is_dropped_raises_rather_than_losing_force():
    forces = torch.randn(3, 3)
    positions = torch.randn(3, 3)
    parent = torch.tensor([-1, 0, -1])
    raw_to_batch = torch.tensor([-1, -1, 0])     # atom 0 is the parent and is dropped
    with pytest.raises(ValueError, match="not in the heavy-atom representation"):
        aggregate_hydrogen_forces(forces, positions, parent, raw_to_batch)


def test_an_untrained_force_predictor_predicts_zero_with_unit_variance():
    predictor = AtomForcePredictor(20)
    output = predictor(torch.randn(9, 20))
    assert torch.allclose(output["mean_local"], torch.zeros(9, 3), atol=1e-7)
    assert torch.allclose(output["log_variance"], torch.zeros(9, 3), atol=1e-7)


def test_the_force_nll_is_finite_and_rewards_the_right_variance():
    target = torch.randn(50, 3)
    mask = torch.ones(50, 3, dtype=torch.bool)
    honest = heteroscedastic_nll(
        torch.zeros(50, 3), torch.zeros(50, 3), target, mask
    )
    overconfident = heteroscedastic_nll(
        torch.zeros(50, 3), torch.full((50, 3), -5.0), target, mask
    )
    assert torch.isfinite(honest) and torch.isfinite(overconfident)
    assert float(honest) < float(overconfident)


def test_h2_arms_reject_an_unknown_name():
    with pytest.raises(ValueError, match="unknown H2 arm"):
        AtomForceConditionerConfig(arm="H2X_made_up")
