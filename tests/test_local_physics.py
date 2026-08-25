"""LocalPhysicsModel integration: contract, ablations, label hygiene."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import (
    apply_rigid_transform,
    frame_atom_indices,
    frames_from_batch,
    link_backbone_to_atom_positions,
    random_rotation_matrix,
)
from force_md.graph import GraphConfig, build_hierarchical_graph
from force_md.models import LocalPhysicsConfig, LocalPhysicsModel
from force_md.nn import EncoderConfig, IrrepsConfig
from force_md.physics import (
    LossWeights,
    ResidueSumProjector,
    omitted_atom_residual,
    phase1_loss,
)

PLM_DIM = 32


def make_config(**overrides) -> LocalPhysicsConfig:
    encoder_kwargs = overrides.pop("encoder_kwargs", {})
    return LocalPhysicsConfig(
        encoder=EncoderConfig(plm_dim=PLM_DIM, **encoder_kwargs), **overrides
    )


def make_model(**overrides) -> LocalPhysicsModel:
    torch.manual_seed(0)
    return LocalPhysicsModel(make_config(**overrides)).to(torch.float64)


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(6), SyntheticSpec(4)], seed=0,
                           include_hydrogens=True, plm_dim=PLM_DIM, dtype=torch.float64)


# --------------------------------------------------------------------------
# forward contract
# --------------------------------------------------------------------------


def test_forward_produces_the_full_output_contract(batch):
    out = make_model()(batch)
    n_a, n_r, b = batch.num_atoms, batch.num_residues, batch.num_graphs
    assert out.atom_force_mean.shape == (n_a, 3)
    assert out.atom_force_residual.shape == (n_a, 3)
    assert out.atom_force_logvar.shape == (n_a, 3)
    assert out.atom_force_conservative.shape == (n_a, 3)
    assert out.residue_explained_force.shape == (n_r, 3)
    assert out.residue_hidden_force.shape == (n_r, 3)
    assert out.residue_force_mean.shape == (n_r, 3)
    assert out.residue_torque_mean.shape == (n_r, 3)
    assert out.residue_force_logvar.shape == (n_r, 3)
    assert out.residue_torque_logvar.shape == (n_r, 3)
    assert out.aggregated_atom_force.shape == (n_r, 3)
    assert out.energy.shape == (b,)
    assert out.residue_energy.shape == (n_r,)
    assert out.physics_latent.shape == (n_r, 152)
    assert out.physics_latent_irreps == "64x0e+16x1o+8x2e"
    assert out.target_scope == "heavy_atom"
    for name in ("atom_force_mean", "residue_force_mean", "physics_latent", "energy"):
        assert bool(torch.isfinite(getattr(out, name)).all()), name


def test_mean_is_residual_plus_conservative(batch):
    out = make_model()(batch)
    assert torch.allclose(
        out.atom_force_mean, out.atom_force_residual + out.atom_force_conservative
    )


def test_force_mean_is_explained_plus_hidden(batch):
    out = make_model()(batch)
    assert torch.allclose(
        out.residue_force_mean,
        out.residue_explained_force + out.residue_hidden_force,
    )


def test_prebuilt_graph_gives_the_same_result(batch):
    model = make_model()
    graph = build_hierarchical_graph(batch, model.config.graph)
    with torch.no_grad():
        a = model(batch)
        b = model(batch, graph)
    assert torch.allclose(a.physics_latent, b.physics_latent, atol=1e-12)


def test_latent_contract_is_reported(batch):
    c = make_model().latent_contract()
    assert c["physics_latent_irreps"] == "64x0e+16x1o+8x2e"
    assert c["physics_latent_dim"] == 152
    assert c["lmax"] == 2
    assert c["num_cycles"] == 2
    assert c["target_scope"] == "heavy_atom"
    assert c["predicts_hidden_force"] is True


# --------------------------------------------------------------------------
# ground-truth forces must never be an input
# --------------------------------------------------------------------------


def test_forward_ignores_ground_truth_forces(batch):
    """The model must read positions and chemistry, never the force label.

    A model that peeked at the label would score beautifully in training and be
    useless at inference, where no force is available.
    """
    model = make_model()
    with torch.no_grad():
        base = model(batch)

        no_labels = dataclasses.replace(
            batch,
            atoms=dataclasses.replace(batch.atoms, forces=None, force_valid=None),
        )
        without = model(no_labels)

        randomised = dataclasses.replace(
            batch,
            atoms=dataclasses.replace(
                batch.atoms, forces=torch.randn_like(batch.atoms.forces) * 100.0
            ),
        )
        scrambled = model(randomised)

    assert torch.equal(base.atom_force_mean, without.atom_force_mean)
    assert torch.equal(base.atom_force_mean, scrambled.atom_force_mean)
    assert torch.equal(base.physics_latent, scrambled.physics_latent)


def test_model_runs_without_any_force_labels(batch):
    no_labels = dataclasses.replace(
        batch, atoms=dataclasses.replace(batch.atoms, forces=None, force_valid=None)
    )
    out = make_model()(no_labels)
    assert bool(torch.isfinite(out.residue_force_mean).all())


# --------------------------------------------------------------------------
# equivariance of the assembled model
# --------------------------------------------------------------------------


def test_model_outputs_are_se3_equivariant(batch):
    model = make_model().eval()
    q = random_rotation_matrix(torch.Generator().manual_seed(1), dtype=torch.float64)
    t = torch.tensor([4.0, -2.0, 7.0], dtype=torch.float64)
    a = model(batch)
    b = model(apply_rigid_transform(batch, q, t))
    for name in ("atom_force_mean", "residue_explained_force", "residue_hidden_force",
                 "residue_force_mean", "residue_torque_mean", "aggregated_atom_force"):
        x, y = getattr(a, name), getattr(b, name)
        err = (y - x @ q.T).abs().max().item() / max(x.abs().max().item(), 1e-12)
        assert err < 1e-6, f"{name}: relative error {err:.2e}"
    assert torch.allclose(a.energy, b.energy, atol=1e-7)
    assert torch.allclose(a.atom_force_logvar, b.atom_force_logvar, atol=1e-8)


def test_conservative_force_is_equivariant(batch):
    model = make_model().eval()
    q = random_rotation_matrix(torch.Generator().manual_seed(2), dtype=torch.float64)
    a = model(batch)
    b = model(apply_rigid_transform(batch, q, torch.zeros(3, dtype=torch.float64)))
    err = (b.atom_force_conservative - a.atom_force_conservative @ q.T).abs().max()
    assert float(err) < 1e-6


# --------------------------------------------------------------------------
# backbone linking (completeness of -grad U)
# --------------------------------------------------------------------------


def test_frame_atom_indices_find_n_ca_c_not_the_cap(batch):
    from force_md.data import residue_constants as rc

    idx, complete = frame_atom_indices(batch)
    assert bool(complete.all())
    names = batch.atoms.atom_name_id
    for slot, name in enumerate(("N", "CA", "C")):
        got = names[idx[:, slot]]
        assert bool((got == rc.atom_name_id(name)).all())
    # CAY exists in this fixture and must not have been chosen as CA
    cay = rc.atom_name_id("CAY")
    assert int((names == cay).sum()) > 0
    assert not bool((names[idx[:, 1]] == cay).any())


def test_linking_makes_backbone_a_view_of_atom_positions(batch):
    linked = link_backbone_to_atom_positions(batch)
    idx, _ = frame_atom_indices(batch)
    assert torch.allclose(linked.backbone.ca_positions, batch.atoms.positions[idx[:, 1]])
    assert torch.allclose(linked.backbone.n_positions, batch.atoms.positions[idx[:, 0]])


def test_conservative_force_gradient_reaches_every_atom(batch):
    """With linked frames the energy gradient is complete, not partial."""
    out = make_model()(batch)
    assert bool(torch.isfinite(out.atom_force_conservative).all())
    assert float(out.atom_force_conservative.detach().abs().sum()) > 0


# --------------------------------------------------------------------------
# ablation hooks: config, not different classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,overrides",
    [
        ("no_plm", {"encoder_kwargs": {"use_plm": False}}),
        ("no_atom_branch", {"encoder_kwargs": {"use_atom_branch": False}}),
        ("no_backbone_branch", {"encoder_kwargs": {"use_backbone_branch": False}}),
        ("no_hidden_residual", {"predict_hidden_force": False}),
        ("no_energy_branch", {"use_energy_branch": False}),
    ],
)
def test_ablation_hooks(batch, name, overrides):
    model = make_model(**overrides)
    assert type(model) is LocalPhysicsModel
    out = model(batch)
    assert bool(torch.isfinite(out.residue_force_mean).all())
    if name == "no_hidden_residual":
        assert out.residue_hidden_force is None
        assert torch.equal(out.residue_force_mean, out.residue_explained_force)
    if name == "no_energy_branch":
        assert out.atom_force_conservative is None
        assert torch.equal(out.atom_force_mean, out.atom_force_residual)
        assert torch.allclose(out.energy, torch.zeros_like(out.energy))


def test_hidden_force_defaults_follow_the_target_scope(batch):
    assert make_model(target_scope="heavy_atom").predict_hidden_force is True
    assert make_model(target_scope="all_atom").predict_hidden_force is False


def test_hidden_force_with_all_atom_target_is_rejected():
    with pytest.raises(ValueError, match="nothing to explain"):
        make_model(target_scope="all_atom", predict_hidden_force=True)


def test_config_changes_depth_without_changing_the_class(batch):
    model = make_model(
        encoder_kwargs={
            "num_cycles": 1,
            "irreps": IrrepsConfig(scalar_channels=16, vector_channels=4,
                                   tensor_channels=2),
        }
    )
    assert type(model) is LocalPhysicsModel
    out = model(batch)
    assert out.physics_latent.shape == (batch.num_residues, 16 + 4 * 3 + 2 * 5)
    assert model.latent_contract()["num_cycles"] == 1


# --------------------------------------------------------------------------
# end-to-end with the loss
# --------------------------------------------------------------------------


def test_end_to_end_loss_and_backward(batch):
    model = make_model()
    model.train()
    out = model(batch)

    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    residual, _ = omitted_atom_residual(allat, heavy)
    frames = frames_from_batch(link_backbone_to_atom_positions(batch))

    total, comp = phase1_loss(
        out, batch, heavy, frames,
        hidden_force_target=residual,
        atom_selection=batch.atoms.is_heavy,
    )
    assert bool(torch.isfinite(total))
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert comp["total"] == pytest.approx(float(total.detach()))


def test_one_optimisation_step_reduces_the_loss(batch):
    """A crude sanity check that the assembled graph is actually trainable."""
    model = make_model()
    model.train()
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    residual, _ = omitted_atom_residual(allat, heavy)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(4):
        opt.zero_grad()
        out = model(batch)
        frames = frames_from_batch(link_backbone_to_atom_positions(batch))
        total, _ = phase1_loss(out, batch, heavy, frames,
                               hidden_force_target=residual,
                               atom_selection=batch.atoms.is_heavy,
                               weights=LossWeights())
        total.backward()
        opt.step()
        losses.append(float(total.detach()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_masked_and_degenerate_residues_do_not_break_the_forward():
    b = synthetic_batch(
        [SyntheticSpec(6, nonstandard_at=(1,), drop_frame_atom_at=(3,),
                       drop_atom_at=(4,))],
        seed=0, include_hydrogens=True, plm_dim=PLM_DIM, dtype=torch.float64,
    )
    out = make_model()(b)
    assert bool(torch.isfinite(out.physics_latent).all())
    assert bool(torch.isfinite(out.atom_force_mean).all())


def test_empty_spatial_graph_still_runs(batch):
    model = make_model(graph=GraphConfig(atom_cutoff=0.01))
    out = model(batch)
    assert bool(torch.isfinite(out.residue_force_mean).all())


def test_single_residue_protein(batch):
    tiny = synthetic_batch([SyntheticSpec(1)], seed=0, include_hydrogens=True,
                           plm_dim=PLM_DIM, dtype=torch.float64)
    out = make_model()(tiny)
    assert out.physics_latent.shape[0] == 1
    assert bool(torch.isfinite(out.physics_latent).all())


def test_batching_does_not_leak(batch):
    model = make_model().eval()
    single = synthetic_batch([SyntheticSpec(6)], seed=0, include_hydrogens=True,
                             plm_dim=PLM_DIM, dtype=torch.float64)
    with torch.no_grad():
        a = model(single)
        b = model(batch)
    assert torch.allclose(a.physics_latent, b.physics_latent[:6], atol=1e-9)


def test_invalid_batch_is_rejected_when_validation_is_on(batch):
    bad = dataclasses.replace(
        batch,
        atoms=dataclasses.replace(
            batch.atoms, atom_to_residue=batch.atoms.atom_to_residue + 100
        ),
    )
    with pytest.raises(ValueError):
        make_model(validate_inputs=True)(bad)


def test_checkpoint_round_trip_preserves_the_output(batch, tmp_path):
    """Reloading a Phase 1 checkpoint must reproduce the same output contract."""
    model = make_model().eval()
    with torch.no_grad():
        before = model(batch)
    path = tmp_path / "phase1.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config}, path)

    payload = torch.load(path, weights_only=False)
    restored = LocalPhysicsModel(payload["config"]).to(torch.float64).eval()
    restored.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        after = restored(batch)

    assert torch.allclose(before.physics_latent, after.physics_latent, atol=1e-12)
    assert torch.allclose(before.atom_force_mean, after.atom_force_mean, atol=1e-12)
    assert before.physics_latent_irreps == after.physics_latent_irreps
    assert restored.latent_contract() == model.latent_contract()
