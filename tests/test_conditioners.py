"""The A-E conditioners, force moments and local-frame projection (Checkpoint 4).

The tests are organised around the four ways a conditioner can be quietly wrong:

* **it is not invariant** -- rotating the protein changes the conditioning, so the
  probe learns to undo tumbling instead of predicting a transition;
* **it depends on atom order** -- a permutation of a residue's atoms changes the
  answer, which means the pooling is not a set operation;
* **it has already thrown the signal away** -- summing forces before the
  nonlinearity leaves a residue whose internal forces cancel indistinguishable
  from one with no forces at all. The ``(+f, -f)`` test is aimed exactly there;
* **it reads a label** -- the oracle bundle reaching a production arm.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.geometry import apply_rigid_transform, random_rotation_matrix  # noqa: E402
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.transition import (  # noqa: E402
    CONDITIONER_ARMS,
    ConditionerConfig,
    FrozenPhase1Extractor,
    IrrepsLocalFrame,
    build_conditioner,
    force_moments,
    precision_weights,
    residue_shape,
)

PLM_DIM = 32
ROOT = os.path.dirname(os.path.dirname(__file__))


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def small_config() -> LocalPhysicsConfig:
    return LocalPhysicsConfig(
        encoder=EncoderConfig(
            plm_dim=PLM_DIM,
            num_cycles=1,
            irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
        ),
        use_energy_branch=False,
    )


@pytest.fixture(scope="module")
def extractor(tmp_path_factory):
    config = small_config()
    model = LocalPhysicsModel(config)
    path = str(tmp_path_factory.mktemp("ck") / "phase1.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config,
            "step": 1,
            "latent_contract": model.latent_contract(),
        },
        path,
    )
    return FrozenPhase1Extractor.from_checkpoint(path)


def make_batch(sizes=(6, 5), seed: int = 0):
    return synthetic_batch(
        [SyntheticSpec(n) for n in sizes], seed=seed, plm_dim=PLM_DIM,
        include_hydrogens=True,
    )


@pytest.fixture
def bundles(extractor):
    batch = make_batch()
    return batch, extractor(batch), extractor.oracle_bundle(batch)


def all_arms(extractor, config=None):
    config = config or ConditionerConfig()
    irreps = extractor.contract["physics_latent_irreps"]
    torch.manual_seed(0)
    return {arm: build_conditioner(arm, config, irreps=irreps) for arm in CONDITIONER_ARMS}


def run(conditioner, bundle, oracle):
    return conditioner(oracle if conditioner.requires_oracle else bundle)


# --------------------------------------------------------------------------
# the local-frame projection
# --------------------------------------------------------------------------


def test_l1_wigner_d_is_the_rotation_matrix():
    """Pins the assumption that lets the l=1 block skip e3nn entirely."""
    from e3nn import o3

    rotations = torch.stack(
        [random_rotation_matrix(generator(i), dtype=torch.float64) for i in range(4)]
    )
    assert torch.allclose(o3.Irrep(1, -1).D_from_matrix(rotations), rotations, atol=1e-5)


def test_local_frame_projection_is_invariant_under_a_global_rotation():
    irreps = "8x0e+4x1o+2x2e"
    projection = IrrepsLocalFrame(irreps)
    from e3nn import o3

    g = generator(1)
    frames = torch.stack(
        [random_rotation_matrix(g, dtype=torch.float64) for _ in range(6)]
    )
    features = torch.randn(6, projection.dim, dtype=torch.float64, generator=g)

    q = random_rotation_matrix(g, dtype=torch.float64)
    rotated_features = features @ o3.Irreps(irreps).D_from_matrix(q).T.to(torch.float64)
    rotated_frames = q @ frames

    assert torch.allclose(
        projection(features, frames),
        projection(rotated_features, rotated_frames),
        atol=1e-4,
    )


def test_local_frame_projection_leaves_scalars_untouched():
    projection = IrrepsLocalFrame("8x0e+4x1o")
    g = generator(2)
    frames = torch.stack(
        [random_rotation_matrix(g, dtype=torch.float64) for _ in range(5)]
    )
    features = torch.randn(5, projection.dim, dtype=torch.float64, generator=g)
    out = projection(features, frames)
    assert torch.equal(out[:, :8], features[:, :8])
    assert not torch.allclose(out[:, 8:], features[:, 8:])


def test_local_frame_projection_of_the_identity_frame_is_a_no_op():
    projection = IrrepsLocalFrame("4x0e+2x1o+1x2e")
    features = torch.randn(3, projection.dim, dtype=torch.float64, generator=generator(3))
    eye = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
    assert torch.allclose(projection(features, eye), features, atol=1e-6)


def test_local_frame_projection_checks_its_widths():
    projection = IrrepsLocalFrame("4x0e+2x1o")
    with pytest.raises(ValueError, match="need"):
        projection(torch.zeros(3, 5), torch.eye(3).expand(3, 3, 3))
    with pytest.raises(ValueError, match="rotations for"):
        projection(torch.zeros(3, projection.dim), torch.eye(3).expand(2, 3, 3))


# --------------------------------------------------------------------------
# force moments
# --------------------------------------------------------------------------


def test_cancelling_forces_leave_a_zero_net_force_but_a_nonzero_moment():
    """The central claim of arm D, as an explicit two-atom example.

    Two atoms of one residue pulled in exactly opposite directions: the residue is
    being stretched, and a conditioner that only sees the sum cannot tell that
    from a residue with no forces at all.
    """
    y = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    f = torch.tensor([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], dtype=torch.float64)
    index = torch.zeros(2, dtype=torch.int64)

    moments = force_moments(y, f, index, 1)
    assert float(moments.net_force.abs().max()) == 0.0
    assert float(moments.torque.abs().max()) == 0.0
    # ... but the moment tensor is not zero: this is an extension
    assert float(moments.isotropic[0, 0]) > 0.0
    assert float(moments.symmetric_traceless.abs().max()) > 0.0

    compressed = force_moments(y, -f, index, 1)
    assert float(compressed.isotropic[0, 0]) < 0.0


def test_a_pure_couple_shows_up_as_torque_not_as_compression():
    y = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    f = torch.tensor([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]], dtype=torch.float64)
    moments = force_moments(y, f, torch.zeros(2, dtype=torch.int64), 1)
    assert float(moments.net_force.abs().max()) == 0.0
    assert float(moments.torque.norm()) == pytest.approx(4.0)
    assert float(moments.isotropic.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_moment_features_are_permutation_invariant():
    g = generator(4)
    y = torch.randn(9, 3, dtype=torch.float64, generator=g)
    f = torch.randn(9, 3, dtype=torch.float64, generator=g)
    index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    reference = force_moments(y, f, index, 3).as_features()
    order = torch.randperm(9, generator=g)
    shuffled = force_moments(y[order], f[order], index[order], 3).as_features()
    assert torch.allclose(reference, shuffled, atol=1e-12)


def test_moments_scale_with_the_weights():
    y = torch.randn(6, 3, dtype=torch.float64, generator=generator(5))
    f = torch.randn(6, 3, dtype=torch.float64, generator=generator(6))
    index = torch.zeros(6, dtype=torch.int64)
    full = force_moments(y, f, index, 1)
    half = force_moments(y, f, index, 1, weights=torch.full((6,), 0.5, dtype=torch.float64))
    assert torch.allclose(half.net_force, full.net_force * 0.5, atol=1e-12)
    assert torch.allclose(half.torque, full.torque * 0.5, atol=1e-12)


def test_residue_shape_measures_size_and_anisotropy():
    sphere = torch.tensor(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]],
        dtype=torch.float64,
    )
    flat = sphere.clone()
    flat[:, 2] = 0.0
    index = torch.zeros(6, dtype=torch.int64)

    round_shape = residue_shape(sphere, index, 1)
    flat_shape = residue_shape(flat, index, 1)
    assert float(round_shape.anisotropy.abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(flat_shape.anisotropy.abs().max()) > 0.1
    assert float(round_shape.max_radius[0, 0]) == pytest.approx(1.0)
    assert float(round_shape.radius_of_gyration[0, 0]) > float(
        flat_shape.radius_of_gyration[0, 0]
    )


# --------------------------------------------------------------------------
# uncertainty gating
# --------------------------------------------------------------------------


def test_precision_weight_falls_as_predicted_uncertainty_rises():
    config = ConditionerConfig()
    logvar = torch.tensor([[-4.0] * 3, [0.0] * 3, [4.0] * 3])
    weights = precision_weights(logvar, config)
    assert weights[0] > weights[1] > weights[2]
    assert float(weights[1]) == pytest.approx(1.0, rel=1e-6)
    assert float(weights[0]) <= config.max_precision_weight + 1e-6


def test_precision_weight_can_be_switched_off():
    config = dataclasses.replace(ConditionerConfig(), uncertainty_gating=False)
    logvar = torch.tensor([[-4.0] * 3, [4.0] * 3])
    assert torch.allclose(precision_weights(logvar, config), torch.ones(2))


def test_invalid_atoms_get_zero_weight():
    config = ConditionerConfig()
    logvar = torch.zeros(3, 3)
    valid = torch.tensor([True, False, True])
    weights = precision_weights(logvar, config, valid=valid)
    assert float(weights[1]) == 0.0
    assert float(weights[0]) > 0.0


def test_uncertain_forces_move_the_conditioner_less(extractor, bundles):
    """The gate must be visible in the output, not just in the helper."""
    _, bundle, _ = bundles
    conditioner = build_conditioner(
        "force_pattern_shape", ConditionerConfig(),
        irreps=extractor.contract["physics_latent_irreps"],
    )
    confident = dataclasses.replace(
        bundle, atom_force_logvar=torch.full_like(bundle.atom_force_logvar, -4.0)
    )
    uncertain = dataclasses.replace(
        bundle, atom_force_logvar=torch.full_like(bundle.atom_force_logvar, 4.0)
    )
    with torch.no_grad():
        assert not torch.allclose(conditioner(confident), conditioner(uncertain), atol=1e-4)


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


def test_every_arm_emits_the_same_shape(extractor, bundles):
    _, bundle, oracle = bundles
    outputs = {
        arm: run(cond, bundle, oracle) for arm, cond in all_arms(extractor).items()
    }
    assert set(outputs) == set(CONDITIONER_ARMS)
    shapes = {tuple(v.shape) for v in outputs.values()}
    assert shapes == {(bundle.num_residues, ConditionerConfig().d_cond)}


def test_the_zero_arm_is_exactly_zero_and_has_no_parameters(extractor, bundles):
    _, bundle, oracle = bundles
    zero = all_arms(extractor)["structure_only"]
    assert zero.parameter_count() == 0
    assert float(run(zero, bundle, oracle).abs().max()) == 0.0


def test_arms_differ_from_each_other(extractor, bundles):
    """If two arms produced the same conditioning the ablation would be vacuous."""
    _, bundle, oracle = bundles
    with torch.no_grad():
        outputs = {arm: run(c, bundle, oracle) for arm, c in all_arms(extractor).items()}
    for arm, value in outputs.items():
        if arm == "structure_only":
            continue
        assert float(value.abs().max()) > 0.0, arm
    assert not torch.allclose(outputs["force_torque"], outputs["physics_latent"])
    assert not torch.allclose(outputs["physics_latent"], outputs["force_pattern_shape"])


def test_every_arm_is_invariant_under_a_global_rigid_motion(extractor):
    batch = make_batch()
    g = generator(7)
    rotation = random_rotation_matrix(g, dtype=batch.atoms.positions.dtype)
    translation = torch.tensor([37.0, -12.0, 5.0], dtype=batch.atoms.positions.dtype)
    moved = apply_rigid_transform(batch, rotation, translation)

    arms = all_arms(extractor)
    with torch.no_grad():
        before = extractor(batch), extractor.oracle_bundle(batch)
        after = extractor(moved), extractor.oracle_bundle(moved)
        for arm, conditioner in arms.items():
            a = run(conditioner, before[0], before[1])
            b = run(conditioner, after[0], after[1])
            # Measured worst case across the arms is 8.6e-7 in float32, against an
            # output scale of ~0.13 -- i.e. machine precision. A tolerance loose
            # enough to hide a real frame bug would make this test decorative.
            assert torch.allclose(a, b, atol=1e-5), arm


def test_every_arm_is_invariant_to_atom_order_within_a_residue(extractor):
    """A residue's atoms are a set. Reordering them must change nothing."""
    batch = make_batch(sizes=(6,))
    order = torch.arange(batch.num_atoms)
    # reverse the atoms of residue 2, keeping the required non-decreasing grouping
    rows = (batch.atoms.atom_to_residue == 2).nonzero(as_tuple=True)[0]
    order[rows] = rows.flip(0)

    atoms = dataclasses.replace(
        batch.atoms,
        positions=batch.atoms.positions[order],
        atom_to_residue=batch.atoms.atom_to_residue[order],
        batch_index=batch.atoms.batch_index[order],
        atomic_number=batch.atoms.atomic_number[order],
        atom_name_id=batch.atoms.atom_name_id[order],
        is_backbone=batch.atoms.is_backbone[order],
        is_cap=batch.atoms.is_cap[order],
        forces=batch.atoms.forces[order],
        force_valid=batch.atoms.force_valid[order],
    )
    permuted = dataclasses.replace(batch, atoms=atoms)

    arms = all_arms(extractor)
    with torch.no_grad():
        base = extractor(batch), extractor.oracle_bundle(batch)
        perm = extractor(permuted), extractor.oracle_bundle(permuted)
        for arm, conditioner in arms.items():
            assert torch.allclose(
                run(conditioner, base[0], base[1]),
                run(conditioner, perm[0], perm[1]),
                atol=1e-4,
            ), arm


def test_masked_residues_get_a_zero_condition(extractor):
    batch = make_batch(sizes=(6,))
    mask = batch.residues.mask.clone()
    mask[3] = False
    masked = dataclasses.replace(
        batch, residues=dataclasses.replace(batch.residues, mask=mask)
    )
    bundle = extractor(masked)
    oracle = extractor.oracle_bundle(masked)
    with torch.no_grad():
        for arm, conditioner in all_arms(extractor).items():
            out = run(conditioner, bundle, oracle)
            assert float(out[3].abs().max()) == 0.0, arm


def test_production_arms_refuse_the_oracle_bundle(extractor, bundles):
    _, _, oracle = bundles
    for arm, conditioner in all_arms(extractor).items():
        if conditioner.requires_oracle:
            continue
        with pytest.raises(TypeError, match="OracleFeatureBundle reached a production"):
            conditioner(oracle)


def test_the_oracle_arm_refuses_a_production_bundle(extractor, bundles):
    _, bundle, _ = bundles
    oracle_arm = all_arms(extractor)["oracle_force"]
    with pytest.raises(TypeError, match="needs an OracleFeatureBundle"):
        oracle_arm(bundle)


def test_the_oracle_arm_uses_the_labels_not_the_predictions(extractor, bundles):
    """Arm E must not collapse into arm D."""
    batch, bundle, oracle = bundles
    config = ConditionerConfig()
    irreps = extractor.contract["physics_latent_irreps"]
    torch.manual_seed(0)
    predicted = build_conditioner("force_pattern_shape", config, irreps=irreps)
    torch.manual_seed(0)
    labelled = build_conditioner("oracle_force", config, irreps=irreps)

    with torch.no_grad():
        assert not torch.allclose(predicted(bundle), labelled(oracle), atol=1e-4)


def test_arm_parameter_counts_are_recorded_and_comparable(extractor):
    counts = {arm: c.parameter_count() for arm, c in all_arms(extractor).items()}
    assert counts["structure_only"] == 0
    assert counts["oracle_force"] == counts["force_pattern_shape"]
    # the learned arms sit within one order of magnitude of each other
    learned = [v for k, v in counts.items() if v > 0]
    assert max(learned) < 10 * min(learned)


def test_conditioner_gradients_flow_to_the_conditioner_only(extractor, bundles):
    _, bundle, oracle = bundles
    for arm, conditioner in all_arms(extractor).items():
        out = run(conditioner, bundle, oracle)
        if conditioner.parameter_count() == 0:
            continue
        conditioner.zero_grad(set_to_none=True)
        out.pow(2).mean().backward()
        assert any(p.grad is not None for p in conditioner.parameters()), arm
        assert all(p.grad is None for p in extractor.phase1.parameters()), arm


def test_unknown_arm_is_rejected(extractor):
    with pytest.raises(ValueError, match="unknown arm"):
        build_conditioner("magic", ConditionerConfig(), irreps="4x0e")


def test_d_cond_is_configurable_and_shared(extractor, bundles):
    _, bundle, oracle = bundles
    config = dataclasses.replace(ConditionerConfig(), d_cond=32)
    for arm, conditioner in all_arms(extractor, config).items():
        assert conditioner.d_cond == 32
        assert run(conditioner, bundle, oracle).shape[1] == 32
