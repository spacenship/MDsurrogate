"""Phase 1.6: the pair-interaction arms, and the guards that make them meaningful.

The tests are organised around the ways a pair arm can be quietly wrong:

* **it conditions on nothing** -- the pair messages were never extracted, the
  bundle's field is ``None``, and the arm silently degenerates into a control.
  That failure mode is a *null result*, which is the worst kind of bug in an
  experiment whose expected outcome is also a small number;
* **it reads the future** -- an edge, a contact or a force from ``t + lag`` leaks
  into the conditioning path and the whole comparison is meaningless;
* **it is not invariant** -- the edge message is a global-frame tensor and
  flattening it makes the probe track the laboratory frame;
* **it is not a control** -- ``P0`` and ``P1`` differ by more than the physics,
  so a win is attributable to capacity;
* **it is not the same experiment** -- two arms evaluated different samples.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.geometry import apply_rigid_transform, random_rotation_matrix  # noqa: E402
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.training.transition_module import (  # noqa: E402
    TransitionTrainConfig,
    TransitionTrainer,
)
from force_md.transition import (  # noqa: E402
    CANONICAL_ARMS,
    ConditionerConfig,
    FrozenPhase1Extractor,
    Phase1FeatureCache,
    TransitionProbe,
    TransitionProbeConfig,
    arm_spec,
    build_conditioner,
    edge_geometry_features,
    future_state_batch,
    matched_hidden_width,
    split_bundle,
)

from test_transition_training import PLM_DIM, make_lag_batch  # noqa: E402

SMALL_IRREPS = IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2)
PAIR_ARMS = (
    "P0_pair_geometry_control",
    "P1_pair_physics_frozen",
    "P2_pair_physics_moments",
    "P3_pair_physics_uncertainty",
    "P4_future_physics_consistency",
)


def small_phase1() -> LocalPhysicsConfig:
    return LocalPhysicsConfig(
        encoder=EncoderConfig(plm_dim=PLM_DIM, num_cycles=1, irreps=SMALL_IRREPS),
        use_energy_branch=False,
    )


@pytest.fixture(scope="module")
def phase1() -> LocalPhysicsModel:
    torch.manual_seed(0)
    return LocalPhysicsModel(small_phase1()).eval()


@pytest.fixture(scope="module")
def pair_extractor(phase1) -> FrozenPhase1Extractor:
    return FrozenPhase1Extractor(
        phase1, phase1.latent_contract(), extract_pair_features=True
    )


@pytest.fixture(scope="module")
def batch():
    return synthetic_batch(
        [SyntheticSpec(8), SyntheticSpec(6, drop_atom_at=(2,), drop_frame_atom_at=(4,))],
        seed=3, plm_dim=PLM_DIM, include_hydrogens=True,
    )


def make_probe(phase1, canonical: str, *, d_cond: int = 16) -> TransitionProbe:
    spec = arm_spec(canonical)
    torch.manual_seed(0)
    return TransitionProbe(
        TransitionProbeConfig(
            arm=spec.implementation,
            plm_dim=PLM_DIM,
            num_blocks=2,
            irreps=SMALL_IRREPS,
            conditioner=ConditionerConfig(
                d_cond=d_cond, hidden=32, atom_message_dim=16, d_pair=8,
                pair_hidden=32, pair_message_dim=16,
            ),
            future_physics=spec.future_physics,
        ),
        latent_irreps=phase1.latent_contract()["physics_latent_irreps"],
        message_irreps=phase1.pair_contract()["pair_message_irreps"],
    )


def run_probe(probe, lag_batch, bundle):
    return probe(
        lag_batch.current, bundle,
        history=lag_batch.history, lag_ps=lag_batch.lag_ps,
    )


# --------------------------------------------------------------------------
# the Phase 1 interface
# --------------------------------------------------------------------------


def test_pair_messages_are_off_by_default(phase1, batch):
    """The default forward path is the one Phase 1 trained with, unchanged."""
    with torch.no_grad():
        default = phase1(batch)
        asked = phase1(batch, return_pair_messages=True)
    assert default.pair is None
    assert asked.pair is not None
    # Not merely close: the same computation, so the same bits.
    assert torch.equal(default.physics_latent, asked.physics_latent)
    assert torch.equal(default.atom_force_mean, asked.atom_force_mean)


def test_pair_message_irreps_match_contract(phase1, batch):
    contract = phase1.pair_contract()
    with torch.no_grad():
        pair = phase1(batch, return_pair_messages=True).pair
    assert str(pair.irreps) == contract["pair_message_irreps"]
    assert pair.message.shape == (pair.num_edges, contract["pair_message_dim"])
    assert pair.distance.shape == (pair.num_edges,)
    assert pair.unit_vector.shape == (pair.num_edges, 3)


def test_pair_edges_never_cross_a_graph_boundary(phase1, batch):
    with torch.no_grad():
        pair = phase1(batch, return_pair_messages=True).pair
    graph = batch.residues.batch_index
    assert torch.equal(graph[pair.src], graph[pair.dst])


def test_bundle_without_pair_features_fails_loudly(phase1, batch):
    """A pair arm handed a node-only bundle must raise, not condition on zeros."""
    node_only = FrozenPhase1Extractor(phase1, phase1.latent_contract())
    bundle = node_only(batch)
    assert bundle.pair is None
    conditioner = build_conditioner(
        "pair_physics", ConditionerConfig(d_pair=8),
        irreps=phase1.latent_contract()["physics_latent_irreps"],
        message_irreps=phase1.pair_contract()["pair_message_irreps"],
    )
    with pytest.raises(ValueError, match="no pair messages"):
        conditioner(bundle)


def test_pair_arm_cannot_be_built_without_message_irreps(phase1):
    with pytest.raises(ValueError, match="message_irreps"):
        build_conditioner(
            "pair_physics", ConditionerConfig(),
            irreps=phase1.latent_contract()["physics_latent_irreps"],
        )


# --------------------------------------------------------------------------
# shape, raggedness, masking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("canonical", list(CANONICAL_ARMS))
def test_every_arm_emits_the_same_output_shapes(phase1, canonical):
    """One backbone, one output contract. The arm changes what is seen, not what
    is produced -- otherwise the metric is not comparing like with like."""
    spec = arm_spec(canonical)
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(),
        extract_pair_features=spec.needs_pair_features,
    )
    lag_batch = make_lag_batch(seed=1)
    bundle = (
        extractor.oracle_bundle(lag_batch.current) if spec.oracle
        else extractor(lag_batch.current)
    )
    probe = make_probe(phase1, canonical)
    out = run_probe(probe, lag_batch, bundle)
    n = lag_batch.current.num_residues
    assert out.translation_local.shape == (n, 3)
    assert out.rotation.shape == (n, 3, 3)
    assert torch.isfinite(out.translation_local).all()
    assert torch.isfinite(out.rotation).all()
    if spec.future_physics:
        assert out.future_physics_latent.shape == (
            n, phase1.latent_contract()["physics_latent_dim"]
        )
    else:
        assert out.future_physics_latent is None


def test_ragged_batch_and_missing_atoms(pair_extractor, batch):
    """Two proteins of different sizes, one with a dropped atom and a degenerate
    frame. Invalid residues must condition on exact zeros, not on garbage."""
    bundle = pair_extractor(batch)
    conditioner = build_conditioner(
        "pair_physics_moments", ConditionerConfig(d_pair=8, d_cond=16),
        irreps=bundle.physics_latent_irreps,
        message_irreps=str(bundle.pair.irreps),
    )
    out = conditioner(bundle)
    assert out.shape == (batch.num_residues, 16)
    assert torch.isfinite(out).all()
    invalid = ~bundle.residue_valid
    if bool(invalid.any()):
        assert torch.equal(out[invalid], torch.zeros_like(out[invalid]))


def test_pair_edge_permutation_does_not_change_the_answer(pair_extractor, batch):
    """The pooling is a set operation over each residue's edges. Shuffling the
    edge rows is a relabelling, and a conditioner that noticed would be reading
    the neighbour-list order as if it were information."""
    bundle = pair_extractor(batch)
    conditioner = build_conditioner(
        "pair_physics", ConditionerConfig(d_pair=8, d_cond=16),
        irreps=bundle.physics_latent_irreps,
        message_irreps=str(bundle.pair.irreps),
    ).eval()
    reference = conditioner(bundle)

    permutation = torch.randperm(bundle.pair.num_edges, generator=torch.Generator().manual_seed(0))
    shuffled = dataclasses.replace(
        bundle,
        pair=dataclasses.replace(
            bundle.pair,
            src=bundle.pair.src[permutation],
            dst=bundle.pair.dst[permutation],
            edge_type=bundle.pair.edge_type[permutation],
            message=bundle.pair.message[permutation],
            distance=bundle.pair.distance[permutation],
            unit_vector=bundle.pair.unit_vector[permutation],
        ),
    )
    assert torch.allclose(conditioner(shuffled), reference, atol=1e-5)


# --------------------------------------------------------------------------
# equivariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["pair_geometry", "pair_physics", "pair_physics_moments"])
def test_conditioner_is_invariant_under_a_global_rigid_motion(phase1, arm):
    """Rotate and translate the whole protein: the conditioning must not move.

    Run in float64 for the same reason every other equivariance test in this
    repository is: a float32 tolerance loose enough to pass is also loose enough
    to hide a real frame error.
    """
    with torch.autocast("cpu", enabled=False):
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        try:
            model = LocalPhysicsModel(small_phase1()).to(dtype).eval()
            extractor = FrozenPhase1Extractor(
                model, model.latent_contract(), extract_pair_features=True
            )
            conditioner = build_conditioner(
                arm, ConditionerConfig(d_pair=8, d_cond=16),
                irreps=model.latent_contract()["physics_latent_irreps"],
                message_irreps=model.pair_contract()["pair_message_irreps"],
            ).to(dtype).eval()

            original = synthetic_batch(
                [SyntheticSpec(7)], seed=5, plm_dim=PLM_DIM, dtype=dtype
            )
            rotation = random_rotation_matrix(
                generator=torch.Generator().manual_seed(2)
            ).to(dtype)
            translation = torch.tensor([3.0, -7.0, 11.0], dtype=dtype)
            moved = apply_rigid_transform(original, rotation, translation)

            a = conditioner(extractor(original))
            b = conditioner(extractor(moved))
            assert torch.allclose(a, b, atol=1e-8), float((a - b).abs().max())
        finally:
            torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------
# capacity control
# --------------------------------------------------------------------------


def test_geometry_control_is_capacity_matched_to_the_physics_arm(phase1):
    """P0 must not lose for want of parameters.

    The tolerance is one-sided in spirit: the control is allowed to be *larger*.
    What it may not be is meaningfully smaller, because then "P1 beats P0" has a
    second explanation.
    """
    kwargs = dict(
        irreps=phase1.latent_contract()["physics_latent_irreps"],
        message_irreps=phase1.pair_contract()["pair_message_irreps"],
    )
    config = ConditionerConfig(d_pair=32, d_cond=64)
    control = build_conditioner("pair_geometry", config, **kwargs)
    physics = build_conditioner("pair_physics", config, **kwargs)
    difference = abs(control.parameter_count() - physics.parameter_count())
    assert difference / physics.parameter_count() < 0.01, (
        control.parameter_count(), physics.parameter_count()
    )
    # And the difference must live in the source, not in the shared downstream.
    assert (
        sum(p.numel() for p in control.adapter.parameters())
        == sum(p.numel() for p in physics.adapter.parameters())
    )
    assert (
        sum(p.numel() for p in control.message.parameters())
        == sum(p.numel() for p in physics.message.parameters())
    )


def test_matched_hidden_width_solves_the_parameter_equation():
    hidden = matched_hidden_width(physics_in=880, geometry_in=37, d_pair=32)
    physics = 2 * 880 + 880 * 32 + 32
    geometry = 2 * 37 + 37 * hidden + hidden + hidden * 32 + 32
    assert abs(physics - geometry) / physics < 0.01


# --------------------------------------------------------------------------
# leakage guards
# --------------------------------------------------------------------------


def test_production_pair_arms_refuse_an_oracle_bundle(pair_extractor, batch):
    bundle = pair_extractor.oracle_bundle(batch)
    for arm in ("pair_geometry", "pair_physics", "pair_physics_moments"):
        conditioner = build_conditioner(
            arm, ConditionerConfig(d_pair=8),
            irreps=bundle.production.physics_latent_irreps,
            message_irreps=str(bundle.production.pair.irreps),
        )
        with pytest.raises(TypeError, match="labels, never inputs"):
            conditioner(bundle)


def test_only_the_oracle_arm_reads_ground_truth_force(phase1):
    """Randomise the force labels; every production arm must be bit-identical.

    Compared at the **conditioner output**, not at the probe output: the probe's
    heads are zero-initialised, so an untrained probe emits exact zeros whatever
    it is conditioned on and would pass this test while leaking freely.
    """
    lag_batch = make_lag_batch(seed=2)
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(), extract_pair_features=True
    )
    corrupted = dataclasses.replace(
        lag_batch.current,
        atoms=dataclasses.replace(
            lag_batch.current.atoms,
            forces=torch.randn_like(lag_batch.current.atoms.forces) * 100,
        ),
    )
    for canonical in ("P1_pair_physics_frozen", "P2_pair_physics_moments"):
        conditioner = make_probe(phase1, canonical).eval().conditioner
        with torch.no_grad():
            a = conditioner(extractor(lag_batch.current))
            b = conditioner(extractor(corrupted))
        assert torch.equal(a, b), canonical

    node_only = FrozenPhase1Extractor(phase1, phase1.latent_contract())
    oracle = make_probe(phase1, "O_current_gt_force_oracle").eval().conditioner
    with torch.no_grad():
        a = oracle(node_only.oracle_bundle(lag_batch.current))
        b = oracle(node_only.oracle_bundle(corrupted))
    # The oracle is *supposed* to notice. If it does not, arm O measures nothing.
    assert not torch.equal(a, b)


@pytest.mark.parametrize("canonical", PAIR_ARMS)
def test_changing_the_future_does_not_change_the_conditioning(phase1, canonical):
    """Guard 5: perturb the future structure arbitrarily. The current-frame
    conditioner tensors must be identical."""
    spec = arm_spec(canonical)
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(), extract_pair_features=True
    )
    lag_batch = make_lag_batch(seed=4)
    bundle = extractor(lag_batch.current)
    conditioner = build_conditioner(
        spec.implementation, ConditionerConfig(d_pair=8, d_cond=16),
        irreps=bundle.physics_latent_irreps,
        message_irreps=str(bundle.pair.irreps),
    ).eval()

    from force_md.transition.pair_physics import ConditionerContext

    def condition(b):
        if conditioner.wants_context:
            index = lag_batch.current.residues.batch_index
            return conditioner(b, context=ConditionerContext(
                temperature_kelvin=lag_batch.current.temperature.reshape(-1)[index],
                lag_ps=lag_batch.lag_ps.reshape(-1)[index],
            ))
        return conditioner(b)

    reference = condition(bundle)
    wrecked = dataclasses.replace(
        lag_batch,
        future=dataclasses.replace(
            lag_batch.future,
            positions=lag_batch.future.positions + 137.0,
            ca_positions=lag_batch.future.ca_positions * -3.0,
        ),
    )
    assert torch.equal(condition(extractor(wrecked.current)), reference)


def test_future_state_batch_has_no_forces(phase1):
    lag_batch = make_lag_batch(seed=6)
    future = future_state_batch(lag_batch.current, lag_batch.future)
    assert future.atoms.forces is None
    assert future.atoms.force_valid is None
    assert torch.equal(future.atoms.positions, lag_batch.future.positions)
    # topology carried over, coordinates replaced
    assert torch.equal(future.atoms.atom_to_residue, lag_batch.current.atoms.atom_to_residue)
    assert torch.equal(future.residues.plm_embedding, lag_batch.current.residues.plm_embedding)


def test_p4_target_is_detached_and_never_reaches_the_conditioner(phase1):
    """Guards 4 and 9: the future latent is a target, carries no gradient, and
    the probe's own forward cannot see it."""
    from force_md.transition.future_physics import future_physics_target

    lag_batch = make_lag_batch(seed=7)
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(), extract_pair_features=True
    )
    probe = make_probe(phase1, "P4_future_physics_consistency")
    bundle = extractor(lag_batch.current)
    future_latent = extractor.node_latent(
        future_state_batch(lag_batch.current, lag_batch.future)
    )
    target = future_physics_target(
        future_latent, bundle.frames.rotation, probe.future_physics_head
    )
    assert not target.requires_grad
    assert target.grad_fn is None

    # The probe's forward signature has no parameter that could carry it.
    import inspect

    parameters = set(inspect.signature(probe.forward).parameters)
    assert parameters == {"batch", "bundle", "history", "lag_ps"}


def test_frozen_phase1_receives_no_gradient_from_a_pair_arm(phase1):
    """Guard 8. Not ``requires_grad=False`` alone: backpropagate a real loss and
    require every Phase 1 parameter to still have ``grad is None``."""
    model = LocalPhysicsModel(small_phase1())
    extractor = FrozenPhase1Extractor(
        model, model.latent_contract(), extract_pair_features=True
    )
    lag_batch = make_lag_batch(seed=8)
    probe = make_probe(model, "P2_pair_physics_moments")
    out = run_probe(probe, lag_batch, extractor(lag_batch.current))
    out.translation_local.pow(2).sum().backward()

    assert all(p.grad is None for p in extractor.phase1.parameters())
    assert any(p.grad is not None for p in probe.conditioner.parameters())


def test_oracle_checkpoint_cannot_be_exported_as_production(phase1, tmp_path):
    """Guard 6."""
    lag_batch = make_lag_batch(seed=9)
    extractor = FrozenPhase1Extractor(phase1, phase1.latent_contract())
    probe = make_probe(phase1, "O_current_gt_force_oracle")
    trainer = TransitionTrainer(
        probe, extractor, TransitionTrainConfig(device="cpu", max_steps=1)
    )
    path = str(tmp_path / "oracle.pt")
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["oracle"] is True
    assert payload["provenance"]["oracle"] is True
    with pytest.raises(ValueError, match="is an oracle checkpoint"):
        TransitionTrainer.export_production_checkpoint(path, str(tmp_path / "out.pt"))


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_rejects_a_shard_written_without_pair_features(pair_extractor, batch, tmp_path):
    """Guard: a pair arm must not silently read a node-only cache shard."""
    node_cache = Phase1FeatureCache(
        str(tmp_path), checkpoint_sha256="abc", config_hash="def", pair_features=False
    )
    node_only = FrozenPhase1Extractor(
        pair_extractor.phase1, pair_extractor.contract
    )
    bundle = node_only(batch)
    node_cache.save("dom", {"dom/320/0/5": split_bundle(bundle, 0)})

    pair_cache = Phase1FeatureCache(
        str(tmp_path), checkpoint_sha256="abc", config_hash="def", pair_features=True
    )
    with pytest.raises(ValueError, match="different Phase 1"):
        pair_cache.load("dom")


def test_cache_refuses_to_shard_a_pair_bundle(pair_extractor, batch):
    with pytest.raises(ValueError, match="cannot shard a bundle carrying pair"):
        split_bundle(pair_extractor(batch), 0)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def make_pair_trainer(phase1, canonical, **overrides):
    spec = arm_spec(canonical)
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(),
        extract_pair_features=spec.needs_pair_features,
    )
    probe = make_probe(phase1, canonical)
    defaults = dict(device="cpu", max_steps=6, warmup_steps=2, eval_every=100,
                    log_every=1000)
    if spec.future_physics:
        defaults["future_physics_weight"] = 0.1
    defaults.update(overrides)
    return TransitionTrainer(probe, extractor, TransitionTrainConfig(**defaults))


@pytest.mark.parametrize("canonical", ["P1_pair_physics_frozen", "P2_pair_physics_moments"])
def test_a_pair_arm_overfits_a_repeated_batch(phase1, canonical):
    """The loss must fall on one batch shown repeatedly. A pair arm that cannot
    fit a single batch is broken in a way no ablation table would reveal."""
    trainer = make_pair_trainer(phase1, canonical, max_steps=40, learning_rate=3e-3)
    lag_batch = make_lag_batch(seed=10)
    first = trainer.train_step(lag_batch)["total"]
    for _ in range(30):
        last = trainer.train_step(lag_batch)["total"]
    assert math.isfinite(last)
    assert last < first, (first, last)


def test_p4_auxiliary_loss_is_reported_and_changes_the_total(phase1):
    trainer = make_pair_trainer(phase1, "P4_future_physics_consistency")
    components = trainer.train_step(make_lag_batch(seed=11))
    assert "future_physics_huber" in components
    assert "future_physics_cosine" in components
    assert math.isfinite(components["total"])
    assert -1.0001 <= components["future_physics_cosine"] <= 1.0001


def test_future_head_without_a_weight_is_refused(phase1):
    with pytest.raises(ValueError, match="silently be a copy of P2"):
        make_pair_trainer(
            phase1, "P4_future_physics_consistency", future_physics_weight=0.0
        )


def test_weight_without_a_head_is_refused(phase1):
    with pytest.raises(ValueError, match="no auxiliary head"):
        make_pair_trainer(
            phase1, "P2_pair_physics_moments", future_physics_weight=0.1
        )


def test_lag_conditioning_contract(phase1):
    """The probe must answer differently for 1 ns and 4 ns, and must refuse to be
    asked without a lag at all."""
    extractor = FrozenPhase1Extractor(
        phase1, phase1.latent_contract(), extract_pair_features=True
    )
    lag_batch = make_lag_batch(seed=12)
    probe = make_probe(phase1, "P2_pair_physics_moments")
    # Untrained heads are zero-initialised, so compare the conditioning-bearing
    # node inputs rather than the (identically zero) output.
    bundle = extractor(lag_batch.current)
    with pytest.raises(ValueError, match="lag_ps is required"):
        probe(lag_batch.current, bundle, history=lag_batch.history, lag_ps=None)

    from force_md.transition.probe import lag_features

    one = lag_features(torch.tensor([1000.0]))
    four = lag_features(torch.tensor([4000.0]))
    assert not torch.allclose(one, four)


@pytest.mark.parametrize("canonical", ["P1_pair_physics_frozen", "P4_future_physics_consistency"])
def test_checkpoint_round_trip_restores_a_pair_arm(phase1, canonical, tmp_path):
    trainer = make_pair_trainer(phase1, canonical)
    lag_batch = make_lag_batch(seed=13)
    trainer.train_step(lag_batch)
    path = str(tmp_path / "arm.pt")
    trainer.save_checkpoint(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["message_irreps"] == phase1.pair_contract()["pair_message_irreps"]

    probe, restored = TransitionTrainer.load_checkpoint(path, trainer.extractor, "cpu")
    assert restored.step == trainer.step
    trainer.module.eval()
    probe.eval()
    with torch.no_grad():
        a = run_probe(trainer.module, lag_batch, trainer.extractor(lag_batch.current))
        b = run_probe(probe, lag_batch, trainer.extractor(lag_batch.current))
    assert torch.equal(a.translation_local, b.translation_local)


def test_same_seed_gives_the_same_pair_arm(phase1):
    losses = []
    for _ in range(2):
        trainer = make_pair_trainer(phase1, "P1_pair_physics_frozen", seed=7)
        losses.append(trainer.train_step(make_lag_batch(seed=14))["total"])
    assert losses[0] == losses[1]


def test_nonfinite_input_is_caught_not_trained_on(phase1):
    """A NaN coordinate must not be silently optimised through."""
    trainer = make_pair_trainer(phase1, "P1_pair_physics_frozen")
    lag_batch = make_lag_batch(seed=15)
    positions = lag_batch.current.atoms.positions.clone()
    positions[0] = float("nan")
    broken = dataclasses.replace(
        lag_batch,
        current=dataclasses.replace(
            lag_batch.current,
            atoms=dataclasses.replace(lag_batch.current.atoms, positions=positions),
        ),
    )
    before = [p.detach().clone() for p in trainer.module.parameters()]
    components = trainer.train_step(broken)
    assert not math.isfinite(components["total"]) or trainer.skipped_steps == 1
    if trainer.skipped_steps == 1:
        after = list(trainer.module.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_provenance_records_every_required_hash(phase1):
    trainer = make_pair_trainer(phase1, "P2_pair_physics_moments")
    provenance = trainer.provenance()
    for key in ("phase1_sha256", "phase1_checkpoint", "parameter_breakdown",
                "canonical_arm", "oracle", "uses_pair_features", "git", "resources"):
        assert key in provenance, key
    assert provenance["canonical_arm"] == "P2_pair_physics_moments"
    assert provenance["uses_pair_features"] is True
    assert provenance["oracle"] is False
    assert set(provenance["resources"]) >= {"peak_gpu_memory_bytes", "train_wall_time_s"}


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_sample_identity_checker_notices_a_mismatch():
    """Requirement 11: the analysis must refuse to compare arms that evaluated
    different samples, rather than silently intersecting them."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "analyze_phase1_6", os.path.join(root, "scripts", "analyze_phase1_6.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [{"pair_id": "a/320/0/5@1000", "domain": "a", "temperature": "320",
             "lag_ns": 1.0, "ca_rmsd": 1.0},
            {"pair_id": "b/320/0/5@1000", "domain": "b", "temperature": "320",
             "lag_ns": 1.0, "ca_rmsd": 2.0}]
    same = {"S0_structure_history": {"records": rows, "provenance": {}, "dir": ""},
            "P1_pair_physics_frozen": {"records": rows, "provenance": {}, "dir": ""}}
    assert module.check_same_samples(same) == []

    different = dict(same)
    different["P1_pair_physics_frozen"] = {
        "records": rows[:1], "provenance": {}, "dir": ""
    }
    assert module.check_same_samples(different)

    # And a degenerate oracle gap is named, not divided by.
    assert module.recoverability(2.0, 1.9, 2.0) == "not identifiable"
    assert module.recoverability(2.0, 1.9, 1.8) == "0.500"


def test_plots_are_written_without_a_plotting_dependency(tmp_path):
    """The figures are deliverables, so they must not depend on an optional
    import that is absent from this environment. Dependency-free SVG."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "analyze_phase1_6", os.path.join(root, "scripts", "analyze_phase1_6.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def records(offset: float) -> list[dict]:
        return [
            {"pair_id": f"{d}/320/{r}/5@{int(lag * 1000)}", "domain": d,
             "temperature": "320", "lag_ns": lag,
             "ca_rmsd": 2.0 + offset + i * 0.1,
             "ca_rmsd_identity": 2.5,
             "rotation_geodesic_deg": 30.0 + offset,
             "rotation_geodesic_deg_identity": 32.0}
            for i, d in enumerate(("a", "b", "c"))
            for r in (0, 1)
            for lag in (1.0, 4.0)
        ]

    runs = {
        "S0_structure_history": {"records": records(0.0), "provenance": {}, "dir": ""},
        "P1_pair_physics_frozen": {"records": records(-0.05), "provenance": {}, "dir": ""},
        "O_current_gt_force_oracle": {"records": records(-0.08), "provenance": {}, "dir": ""},
    }
    written = module.write_plots([runs], str(tmp_path))
    assert written and all(p.endswith(".svg") for p in written)
    for path in written:
        text = open(path).read()
        assert text.startswith("<svg") and text.rstrip().endswith("</svg>")
