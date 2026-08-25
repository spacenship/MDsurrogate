"""The minimal transition probe and its losses (Phase 1.5, Checkpoint 5).

Four properties carry the experiment, and each has its own group below.

**Fairness.** Every arm must be the same network fed a different conditioning
block: same backbone parameter count, same output contract, same starting point.
If arm D began from a different place than arm A, the comparison would measure
initialisation.

**Equivariance.** Both outputs live in the residue's own frame, so a global rigid
motion of the whole pair must leave them bit-for-bit comparable. The probe never
sees a Kabsch-aligned input.

**No leakage.** The future frame is a label. The probe's forward signature does
not accept one, and the conditioners refuse an oracle bundle unless the arm is
the oracle arm.

**It can actually learn.** A model that satisfies the three properties above and
cannot fit anything is useless. Measured on one real mdCATH pair, 600 steps at
lr 3e-3 take the loss from 1.26 to 0.0031 (Ca RMSD 2.90 -> 0.05 A, frame rotation
23.4 -> 0.6 degrees), so the smoke test below asks for a large drop rather than a
token one.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.data.contracts import FrameGeometry  # noqa: E402
from force_md.geometry import (  # noqa: E402
    apply_rigid_transform,
    frame_atom_indices,
    is_proper_rotation,
    random_rotation_matrix,
    so3_exp_map,
)
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.transition import (  # noqa: E402
    CONDITIONER_ARMS,
    ConditionerConfig,
    FrozenPhase1Extractor,
    TransitionLossWeights,
    TransitionProbe,
    TransitionProbeConfig,
    DISPLACEMENT_FEATURE_DIM,
    build_transition_target,
    displacement_features,
    identity_prediction,
    lag_features,
    transition_loss,
    transition_metrics,
)

PLM_DIM = 32


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def extractor(tmp_path_factory):
    config = LocalPhysicsConfig(
        encoder=EncoderConfig(
            plm_dim=PLM_DIM,
            num_cycles=1,
            irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
        ),
        use_energy_branch=False,
    )
    model = LocalPhysicsModel(config)
    path = str(tmp_path_factory.mktemp("ck") / "phase1.pt")
    torch.save(
        {"state_dict": model.state_dict(), "model_config": config, "step": 1,
         "latent_contract": model.latent_contract()},
        path,
    )
    return FrozenPhase1Extractor.from_checkpoint(path)


def probe_config(**overrides) -> TransitionProbeConfig:
    base = dict(
        plm_dim=PLM_DIM,
        num_blocks=2,
        irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
        conditioner=ConditionerConfig(d_cond=16, hidden=32, atom_message_dim=16),
    )
    base.update(overrides)
    return TransitionProbeConfig(**base)


def make_probe(extractor, **overrides) -> TransitionProbe:
    torch.manual_seed(0)
    return TransitionProbe(
        probe_config(**overrides), latent_irreps=extractor.contract["physics_latent_irreps"]
    )


def frame_from_positions(batch, positions, *, offset: int) -> FrameGeometry:
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


def displaced(batch, *, scale: float, seed: int, offset: int) -> FrameGeometry:
    """A rigidly-perturbed copy of the structure, as a history or future frame."""
    g = generator(seed)
    n_res = batch.num_residues
    shift = torch.randn(n_res, 3, dtype=batch.atoms.positions.dtype, generator=g) * scale
    turn = so3_exp_map(
        torch.randn(n_res, 3, dtype=batch.atoms.positions.dtype, generator=g) * scale * 0.3
    )
    a2r = batch.atoms.atom_to_residue
    ca = batch.backbone.ca_positions
    local = batch.atoms.positions - ca[a2r]
    moved = torch.einsum("nij,nj->ni", turn[a2r], local) + ca[a2r] + shift[a2r]
    return frame_from_positions(batch, moved, offset=offset)


@pytest.fixture
def episode(extractor):
    """A batch, its history and future frames, the target and the bundles."""
    batch = synthetic_batch(
        [SyntheticSpec(7), SyntheticSpec(6)], seed=0, plm_dim=PLM_DIM,
        include_hydrogens=True,
    )
    history = (displaced(batch, scale=0.15, seed=11, offset=-1),)
    future = displaced(batch, scale=0.4, seed=12, offset=4)
    target = build_transition_target(batch, future)
    lag = torch.full((batch.num_graphs,), 4000.0, dtype=batch.atoms.positions.dtype)
    return {
        "batch": batch,
        "history": history,
        "future": future,
        "target": target,
        "lag": lag,
        "bundle": extractor(batch),
        "oracle": extractor.oracle_bundle(batch),
    }


def run(probe, episode):
    bundle = episode["oracle"] if probe.conditioner.requires_oracle else episode["bundle"]
    return probe(episode["batch"], bundle, history=episode["history"], lag_ps=episode["lag"])


# --------------------------------------------------------------------------
# lag encoding
# --------------------------------------------------------------------------


def test_lag_features_separate_the_two_lags():
    one, four = lag_features(torch.tensor([1000.0])), lag_features(torch.tensor([4000.0]))
    assert one.shape == four.shape
    assert not torch.allclose(one, four)


def test_lag_features_are_ordered_and_continuous():
    """Continuous, not a two-way one-hot: 2 ns must sit between 1 and 4 ns."""
    values = lag_features(torch.tensor([1000.0, 2000.0, 4000.0]))
    assert float((values[1] - values[0]).norm()) < float((values[2] - values[0]).norm())


# --------------------------------------------------------------------------
# fairness across arms
# --------------------------------------------------------------------------


def test_every_arm_shares_one_backbone_and_output_contract(extractor, episode):
    shapes, backbones, translations = set(), set(), set()
    for arm in CONDITIONER_ARMS:
        probe = make_probe(extractor, arm=arm)
        with torch.no_grad():
            prediction = run(probe, episode)
        shapes.add((tuple(prediction.translation_local.shape), tuple(prediction.rotation.shape)))
        backbones.add(probe.parameter_breakdown()["blocks"])
        translations.add(probe.parameter_breakdown()["heads"])
    n = episode["batch"].num_residues
    assert shapes == {((n, 3), (n, 3, 3))}
    assert len(backbones) == 1, "arms must share one backbone size"
    assert len(translations) == 1, "arms must share one head size"


def test_every_arm_starts_at_the_identity_baseline(extractor, episode):
    """Zero-initialised heads mean no arm gets a head start."""
    baseline = identity_prediction(episode["target"])
    for arm in CONDITIONER_ARMS:
        probe = make_probe(extractor, arm=arm)
        with torch.no_grad():
            prediction = run(probe, episode)
        assert float(prediction.translation_local.abs().max()) == 0.0, arm
        assert torch.allclose(prediction.rotation, baseline.rotation, atol=1e-6), arm


def test_the_predicted_rotation_is_always_a_proper_rotation(extractor, episode):
    probe = make_probe(extractor, arm="force_pattern_shape")
    # perturb the heads away from zero so this is not a test of the identity
    with torch.no_grad():
        for parameter in probe.rotation_head.parameters():
            parameter.normal_(generator=generator(3))
        prediction = run(probe, episode)
    assert bool(is_proper_rotation(prediction.rotation, tolerance=1e-4).all())


def test_parameter_breakdown_adds_up(extractor):
    probe = make_probe(extractor, arm="force_pattern_shape")
    parts = probe.parameter_breakdown()
    assert parts["total"] == probe.parameter_count()
    assert parts["total"] == sum(v for k, v in parts.items() if k != "total")


# --------------------------------------------------------------------------
# equivariance
# --------------------------------------------------------------------------


def test_predictions_are_invariant_under_a_global_rigid_motion(extractor, episode):
    g = generator(5)
    dtype = episode["batch"].atoms.positions.dtype
    rotation = random_rotation_matrix(g, dtype=dtype)
    translation = torch.tensor([53.0, -21.0, 9.0], dtype=dtype)

    moved_batch = apply_rigid_transform(episode["batch"], rotation, translation)
    moved = dict(episode)
    moved["batch"] = moved_batch
    moved["history"] = tuple(
        dataclasses.replace(
            frame,
            positions=frame.positions @ rotation.T + translation,
            n_positions=frame.n_positions @ rotation.T + translation,
            ca_positions=frame.ca_positions @ rotation.T + translation,
            c_positions=frame.c_positions @ rotation.T + translation,
        )
        for frame in episode["history"]
    )
    moved["bundle"] = extractor(moved_batch)
    moved["oracle"] = extractor.oracle_bundle(moved_batch)

    for arm in CONDITIONER_ARMS:
        probe = make_probe(extractor, arm=arm)
        # non-trivial heads, so this tests the network and not a pair of zeros
        with torch.no_grad():
            for head in (probe.translation_head, probe.rotation_head):
                for parameter in head.parameters():
                    parameter.normal_(generator=generator(6))
            before = run(probe, episode)
            after = run(probe, moved)
        assert torch.allclose(
            before.translation_local, after.translation_local, atol=1e-4
        ), arm
        assert torch.allclose(before.rotation, after.rotation, atol=1e-4), arm


# --------------------------------------------------------------------------
# inputs: history, lag, and what must not be readable
# --------------------------------------------------------------------------


def test_the_probe_takes_no_future_frame(extractor, episode):
    """The forward signature cannot be handed a label."""
    import inspect

    parameters = set(inspect.signature(TransitionProbe.forward).parameters)
    assert "future" not in parameters and "target" not in parameters
    assert parameters == {"self", "batch", "bundle", "history", "lag_ps"}


def test_production_arms_refuse_the_oracle_bundle(extractor, episode):
    for arm in CONDITIONER_ARMS:
        probe = make_probe(extractor, arm=arm)
        if probe.conditioner.requires_oracle:
            continue
        with pytest.raises(TypeError, match="OracleFeatureBundle reached a production"):
            probe(episode["batch"], episode["oracle"], history=episode["history"],
                  lag_ps=episode["lag"])


def test_changing_the_lag_changes_the_prediction_but_not_its_contract(extractor, episode):
    probe = make_probe(extractor, arm="physics_latent")
    with torch.no_grad():
        for parameter in probe.translation_head.parameters():
            parameter.normal_(generator=generator(7))
        one = run(probe, episode)
        four = probe(
            episode["batch"], episode["bundle"], history=episode["history"],
            lag_ps=torch.full_like(episode["lag"], 1000.0),
        )
    assert one.translation_local.shape == four.translation_local.shape
    assert one.rotation.shape == four.rotation.shape
    assert not torch.allclose(one.translation_local, four.translation_local)


def test_a_missing_lag_is_refused(extractor, episode):
    probe = make_probe(extractor, arm="structure_only")
    with pytest.raises(ValueError, match="lag_ps is required"):
        probe(episode["batch"], episode["bundle"], history=episode["history"])


def test_the_history_length_must_match_the_configuration(extractor, episode):
    probe = make_probe(extractor, arm="structure_only", history_length=2)
    with pytest.raises(ValueError, match="past frame"):
        probe(episode["batch"], episode["bundle"], history=(), lag_ps=episode["lag"])


def test_history_length_one_needs_no_past_frame(extractor, episode):
    probe = make_probe(extractor, arm="structure_only", history_length=1)
    with torch.no_grad():
        prediction = probe(
            episode["batch"], episode["bundle"], history=(), lag_ps=episode["lag"]
        )
    assert prediction.translation_local.shape == (episode["batch"].num_residues, 3)


def test_history_changes_the_prediction(extractor, episode):
    """If history were ignored, the ablation's `history_length=1` arm would be free."""
    probe = make_probe(extractor, arm="structure_only")
    with torch.no_grad():
        for parameter in probe.translation_head.parameters():
            parameter.normal_(generator=generator(8))
        a = run(probe, episode)
        other = dict(episode)
        other["history"] = (displaced(episode["batch"], scale=0.5, seed=99, offset=-1),)
        b = run(probe, other)
    assert not torch.allclose(a.translation_local, b.translation_local)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def test_a_perfect_prediction_has_almost_zero_loss(episode):
    from force_md.transition import target_as_prediction

    total, components = transition_loss(target_as_prediction(episode["target"]), episode["target"])
    assert float(total) < 1e-6
    assert components["translation_rmse_angstrom"] < 1e-6
    assert components["rotation_error_deg"] < 1e-4


def test_loss_components_are_reported_in_physical_units(episode):
    _, components = transition_loss(identity_prediction(episode["target"]), episode["target"])
    metrics = transition_metrics(
        identity_prediction(episode["target"]), episode["target"], with_baseline=False
    )
    assert components["rotation_error_deg"] == pytest.approx(
        metrics["rotation_geodesic_deg"], rel=0.2
    )
    assert components["valid_residues"] == float(episode["target"].valid.sum())


def test_translation_and_rotation_enter_at_comparable_scale(episode):
    """Angstrom and radians are not commensurable; the scales make them so."""
    _, components = transition_loss(identity_prediction(episode["target"]), episode["target"])
    assert 0.005 < components["translation"] < 20.0
    assert 0.005 < components["rotation"] < 20.0


def test_both_rotation_losses_agree_on_the_ordering(episode):
    from force_md.transition import target_as_prediction

    target = episode["target"]
    good = target_as_prediction(target)
    bad = dataclasses.replace(
        good, rotation=good.rotation.transpose(-1, -2) @ good.rotation.transpose(-1, -2)
    )
    for kind in ("chordal", "geodesic"):
        weights = TransitionLossWeights(rotation_loss=kind)
        good_loss, _ = transition_loss(good, target, weights=weights)
        bad_loss, _ = transition_loss(bad, target, weights=weights)
        assert float(bad_loss) > float(good_loss), kind


def test_an_unknown_rotation_loss_is_refused(episode):
    with pytest.raises(ValueError, match="rotation_loss must be"):
        transition_loss(
            identity_prediction(episode["target"]), episode["target"],
            weights=TransitionLossWeights(rotation_loss="quaternion"),
        )


def test_the_clash_penalty_is_off_by_default_and_can_be_switched_on(episode):
    weights = TransitionLossWeights()
    assert weights.clash == 0.0
    _, components = transition_loss(
        identity_prediction(episode["target"]), episode["target"],
        weights=TransitionLossWeights(clash=1.0),
    )
    assert components["clash"] >= 0.0


def test_a_fully_masked_batch_is_refused(episode):
    target = dataclasses.replace(
        episode["target"], valid=torch.zeros_like(episode["target"].valid)
    )
    with pytest.raises(ValueError, match="no valid residue"):
        transition_loss(identity_prediction(target), target)


# --------------------------------------------------------------------------
# it learns, and it does not touch Phase 1
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_overfitting_one_batch_drives_the_loss_down(extractor, episode):
    """Measured on a real pair: 600 steps at 3e-3 reach loss 0.003 from 1.26.

    Here the budget is smaller, so the bar is a large *relative* drop rather than
    a near-zero loss. A probe that satisfies every contract above and cannot fit
    one batch would pass all of them and be useless.
    """
    torch.set_num_threads(1)  # deterministic; see tests/test_training.py
    probe = make_probe(extractor, arm="force_pattern_shape")
    optimiser = torch.optim.AdamW(probe.parameters(), lr=3e-3)

    first = None
    for _ in range(200):
        optimiser.zero_grad(set_to_none=True)
        total, components = transition_loss(run(probe, episode), episode["target"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 10.0)
        optimiser.step()
        if first is None:
            first = components

    last = components
    assert last["total"] < 0.5 * first["total"], f"{first['total']} -> {last['total']}"
    assert last["translation_rmse_angstrom"] < first["translation_rmse_angstrom"]
    assert last["rotation_error_deg"] < first["rotation_error_deg"]


def test_training_the_probe_leaves_phase_1_untouched(extractor, episode):
    before = [p.detach().clone() for p in extractor.phase1.parameters()]
    probe = make_probe(extractor, arm="force_pattern_shape")
    optimiser = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    for _ in range(3):
        optimiser.zero_grad(set_to_none=True)
        total, _ = transition_loss(run(probe, episode), episode["target"])
        total.backward()
        optimiser.step()

    assert all(p.grad is None for p in extractor.phase1.parameters())
    for old, new in zip(before, extractor.phase1.parameters()):
        assert torch.equal(old, new)


def test_gradients_reach_every_part_of_the_probe(extractor, episode):
    probe = make_probe(extractor, arm="force_pattern_shape")
    total, _ = transition_loss(run(probe, episode), episode["target"])
    total.backward()
    missing = [
        name for name, parameter in probe.named_parameters() if parameter.grad is None
    ]
    assert not missing, missing


# --------------------------------------------------------------------------
# history displacement encoding
#
# The defect these pin down: the raw Angstrom displacement used to seed the l=1
# channels directly. Each BackboneInteractionBlock adds a body-order-3 term
# (h <- h + square_mix(h (x) h)), so magnitude is squared once per block and
# num_blocks blocks compound it to roughly |h|**(2**num_blocks). Measured over 60
# real steps at num_blocks=3, the raw seed gave a median gradient norm of 1.7e6
# and a peak of 1.4e11; the same run with the history input removed gave 0.01 and
# 0.74. Phase 1 never hits this because it interleaves its two backbone blocks
# with pooling and injection layers instead of stacking them.
# --------------------------------------------------------------------------


def test_displacement_features_are_bounded_for_any_physical_magnitude():
    """Every channel stays in [0, 1] -- including well past the 44 A measured max."""
    magnitude = torch.tensor([0.0, 0.5, 1.4, 5.6, 20.0, 44.0, 500.0]).unsqueeze(-1)
    features = displacement_features(magnitude)
    assert features.shape == (7, DISPLACEMENT_FEATURE_DIM)
    assert torch.isfinite(features).all()
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_displacement_features_are_ordered_and_continuous():
    """Continuous in |dr|: 3 A must sit between 1 A and 10 A."""
    values = displacement_features(torch.tensor([1.0, 3.0, 10.0]).unsqueeze(-1))
    assert float((values[1] - values[0]).norm()) < float((values[2] - values[0]).norm())


def test_the_history_seed_vector_has_unit_norm_regardless_of_displacement(
    extractor, episode
):
    """The l=1 seed is a direction. A 100x larger step must not scale it."""
    from force_md.geometry.frames import (
        build_residue_frames,
        link_backbone_to_atom_positions,
    )

    probe = make_probe(extractor, num_blocks=3)
    batch = link_backbone_to_atom_positions(episode["batch"])
    frames = build_residue_frames(
        batch.backbone.n_positions,
        batch.backbone.ca_positions,
        batch.backbone.c_positions,
        prior_valid=batch.backbone.frame_valid,
    )

    seeds = []
    for scale in (1.0, 100.0):
        past = episode["history"][0]
        stretched = dataclasses.replace(
            past,
            positions=batch.atoms.positions
            + (past.positions - batch.atoms.positions) * scale,
            ca_positions=batch.backbone.ca_positions
            + (past.ca_positions - batch.backbone.ca_positions) * scale,
            n_positions=past.n_positions,
            c_positions=past.c_positions,
        )
        _, vectors = probe._inputs(
            batch, episode["bundle"], (stretched,), episode["lag"], frames
        )
        seeds.append(vectors[0])

    for seed in seeds:
        norms = seed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # A pure scaling of the displacement leaves the direction unchanged.
    assert torch.allclose(seeds[0], seeds[1], atol=1e-5)


@pytest.mark.parametrize("num_blocks", [2, 3, 4])
def test_a_large_history_step_does_not_blow_the_gradient_up(
    extractor, episode, num_blocks
):
    """The regression itself: depth must not amplify a large real displacement.

    A 30 A CA step over the history frame is physical at 450 K (the measured
    per-residue maximum over a 630-pair sample is 44 A). Before the fix this
    produced gradient norms of 1e5 at two blocks and 1e11 at three.
    """
    probe = make_probe(extractor, num_blocks=num_blocks)
    batch = episode["batch"]
    past = episode["history"][0]
    direction = past.ca_positions - batch.backbone.ca_positions
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    far = dataclasses.replace(
        past,
        positions=past.positions,
        ca_positions=batch.backbone.ca_positions + direction * 30.0,
    )

    prediction = probe(
        batch, episode["bundle"], history=(far,), lag_ps=episode["lag"]
    )
    total, _ = transition_loss(prediction, episode["target"])
    total.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), float("inf"))
    assert torch.isfinite(grad_norm), f"non-finite gradient at num_blocks={num_blocks}"
    assert float(grad_norm) < 1e3, (
        f"gradient norm {float(grad_norm):.4g} at num_blocks={num_blocks}: the "
        "history displacement is being amplified by the body-order-3 term again"
    )
