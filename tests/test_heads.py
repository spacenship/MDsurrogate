"""Physics heads and losses: symmetry, finite differences, identifiability."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from e3nn import o3

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import (
    apply_rigid_transform,
    frames_from_batch,
    random_rotation_matrix,
    to_local_vectors,
)
from force_md.graph import build_hierarchical_graph
from force_md.nn import EncoderConfig, HierarchicalPhysicsEncoder, IrrepsConfig
from force_md.physics import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    AtomicEffectiveForceHead,
    InvariantEnergyHead,
    LossWeights,
    Phase1Output,
    ResiduePhysicsHead,
    ResidueSumProjector,
    TargetNormalizer,
    conservative_force,
    masked_gaussian_nll,
    masked_mse,
    omitted_atom_residual,
    phase1_loss,
)

IRREPS = IrrepsConfig().node_irreps()
PLM_DIM = 32


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(6), SyntheticSpec(4)], seed=0,
                           include_hydrogens=True, plm_dim=PLM_DIM, dtype=torch.float64)


@pytest.fixture
def frames(batch):
    return frames_from_batch(batch)


def make_encoder():
    torch.manual_seed(0)
    return HierarchicalPhysicsEncoder(
        EncoderConfig(plm_dim=PLM_DIM)
    ).to(torch.float64).eval()


def _wigner_vector(q):
    return o3.Irreps("1x1o").D_from_matrix(q.to(torch.float64))


# --------------------------------------------------------------------------
# AtomicEffectiveForceHead
# --------------------------------------------------------------------------


def test_atom_force_head_shapes_and_structural_split(batch):
    torch.manual_seed(0)
    head = AtomicEffectiveForceHead(IRREPS).to(torch.float64)
    feats = torch.randn(batch.num_atoms, IRREPS.dim, dtype=torch.float64)
    out = head(feats)
    assert out.mean.shape == (batch.num_atoms, 3)
    assert out.logvar.shape == (batch.num_atoms, 3)
    assert out.conservative is None
    assert torch.equal(out.mean, out.residual), "no energy branch -> mean is residual"

    cons = torch.randn_like(out.residual)
    out2 = head(feats, conservative=cons)
    assert torch.allclose(out2.mean, out2.residual + cons)
    assert out2.conservative is cons


def test_atom_force_mean_is_equivariant_and_logvar_invariant(batch):
    """The mean is an l=1 vector; the variance is built from scalars only."""
    enc = make_encoder()
    torch.manual_seed(1)
    head = AtomicEffectiveForceHead(IRREPS).to(torch.float64)
    q = random_rotation_matrix(torch.Generator().manual_seed(3), dtype=torch.float64)
    t = torch.tensor([2.0, -1.0, 4.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)

    with torch.no_grad():
        a = head(enc(batch, build_hierarchical_graph(batch)).atom_features)
        b = head(enc(moved, build_hierarchical_graph(moved)).atom_features)
    assert torch.allclose(b.mean, a.mean @ q.T, atol=1e-8)
    assert torch.allclose(b.logvar, a.logvar, atol=1e-9), "uncertainty must be invariant"


def test_isotropic_uncertainty_has_equal_components(batch):
    torch.manual_seed(0)
    head = AtomicEffectiveForceHead(IRREPS, isotropic_uncertainty=True).to(torch.float64)
    out = head(torch.randn(batch.num_atoms, IRREPS.dim, dtype=torch.float64))
    assert torch.allclose(out.logvar[:, 0], out.logvar[:, 1])
    assert torch.allclose(out.logvar[:, 0], out.logvar[:, 2])


def test_logvar_is_clamped(batch):
    torch.manual_seed(0)
    head = AtomicEffectiveForceHead(IRREPS).to(torch.float64)
    feats = torch.randn(batch.num_atoms, IRREPS.dim, dtype=torch.float64) * 1e4
    out = head(feats)
    assert float(out.logvar.detach().min()) >= LOGVAR_MIN
    assert float(out.logvar.detach().max()) <= LOGVAR_MAX


# --------------------------------------------------------------------------
# ResiduePhysicsHead
# --------------------------------------------------------------------------


def test_residue_head_without_hidden_force(batch):
    torch.manual_seed(0)
    head = ResiduePhysicsHead(IRREPS, predict_hidden_force=False).to(torch.float64)
    feats = torch.randn(batch.num_residues, IRREPS.dim, dtype=torch.float64)
    out = head(feats, batch.backbone.ca_positions)
    assert out.hidden_force is None
    assert torch.equal(out.force_mean, out.explained_force)
    assert not hasattr(head, "to_hidden_force")


def test_residue_head_with_hidden_force(batch):
    torch.manual_seed(0)
    head = ResiduePhysicsHead(IRREPS, predict_hidden_force=True).to(torch.float64)
    feats = torch.randn(batch.num_residues, IRREPS.dim, dtype=torch.float64)
    out = head(feats, batch.backbone.ca_positions)
    assert out.hidden_force is not None
    assert torch.allclose(out.force_mean, out.explained_force + out.hidden_force)


def test_residue_force_and_torque_are_equivariant(batch):
    enc = make_encoder()
    torch.manual_seed(2)
    head = ResiduePhysicsHead(IRREPS, predict_hidden_force=True).to(torch.float64)
    q = random_rotation_matrix(torch.Generator().manual_seed(4), dtype=torch.float64)
    t = torch.tensor([-3.0, 6.0, 0.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)

    with torch.no_grad():
        fa = enc(batch, build_hierarchical_graph(batch)).residue_features
        fb = enc(moved, build_hierarchical_graph(moved)).residue_features
        a = head(fa, batch.backbone.ca_positions)
        b = head(fb, moved.backbone.ca_positions)
    for name in ("explained_force", "hidden_force", "force_mean", "torque_mean"):
        x, y = getattr(a, name), getattr(b, name)
        assert torch.allclose(y, x @ q.T, atol=1e-8), name
    assert torch.allclose(b.force_logvar, a.force_logvar, atol=1e-9)
    assert torch.allclose(b.torque_logvar, a.torque_logvar, atol=1e-9)


# --------------------------------------------------------------------------
# InvariantEnergyHead
# --------------------------------------------------------------------------


def test_energy_is_invariant(batch):
    enc = make_encoder()
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)
    q = random_rotation_matrix(torch.Generator().manual_seed(6), dtype=torch.float64)
    t = torch.tensor([9.0, 9.0, -9.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)
    with torch.no_grad():
        ea, _ = head(enc(batch, build_hierarchical_graph(batch)).residue_features,
                     batch.residues.batch_index, batch.num_graphs)
        eb, _ = head(enc(moved, build_hierarchical_graph(moved)).residue_features,
                     moved.residues.batch_index, moved.num_graphs)
    assert torch.allclose(ea, eb, atol=1e-9)


def test_energy_sums_over_residues(batch):
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)
    feats = torch.randn(batch.num_residues, IRREPS.dim, dtype=torch.float64)
    graph_e, res_e = head(feats, batch.residues.batch_index, batch.num_graphs)
    assert graph_e.shape == (batch.num_graphs,)
    assert res_e.shape == (batch.num_residues,)
    for g in range(batch.num_graphs):
        assert torch.allclose(graph_e[g], res_e[batch.residues.batch_index == g].sum())


def test_conservative_force_is_equivariant(batch):
    """-grad U must rotate with the structure."""
    enc = make_encoder()
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)

    def compute(b):
        pos = b.atoms.positions.clone().requires_grad_(True)
        b2 = dataclasses.replace(b, atoms=dataclasses.replace(b.atoms, positions=pos))
        e, _ = head(enc(b2, build_hierarchical_graph(b2)).residue_features,
                    b2.residues.batch_index, b2.num_graphs)
        return conservative_force(e, pos)

    q = random_rotation_matrix(torch.Generator().manual_seed(8), dtype=torch.float64)
    t = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    fa = compute(batch)
    fb = compute(apply_rigid_transform(batch, q, t))
    assert torch.allclose(fb, fa @ q.T, atol=1e-7)


def test_conservative_force_matches_finite_differences(batch):
    """The autograd gradient must agree with a numerical derivative of U."""
    enc = make_encoder()
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)
    graph = build_hierarchical_graph(batch)

    def energy_of(positions):
        b2 = dataclasses.replace(
            batch, atoms=dataclasses.replace(batch.atoms, positions=positions)
        )
        # the SAME fixed graph: the neighbour list is not differentiated
        e, _ = head(enc(b2, graph).residue_features,
                    batch.residues.batch_index, batch.num_graphs)
        return e.sum()

    pos = batch.atoms.positions.clone().requires_grad_(True)
    analytic = conservative_force(energy_of(pos), pos)

    eps = 1e-6
    with torch.no_grad():
        for atom, comp in [(0, 0), (3, 1), (7, 2)]:
            plus = batch.atoms.positions.clone()
            plus[atom, comp] += eps
            minus = batch.atoms.positions.clone()
            minus[atom, comp] -= eps
            numeric = -(energy_of(plus) - energy_of(minus)) / (2 * eps)
            assert abs(float(numeric) - float(analytic[atom, comp])) < 1e-5, (
                f"atom {atom} component {comp}: "
                f"analytic {float(analytic[atom, comp]):.6e} vs numeric {float(numeric):.6e}"
            )


def test_conservative_force_requires_grad_enabled_positions(batch):
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)
    feats = torch.randn(batch.num_residues, IRREPS.dim, dtype=torch.float64)
    e, _ = head(feats, batch.residues.batch_index, batch.num_graphs)
    with pytest.raises(ValueError, match="require grad"):
        conservative_force(e, batch.atoms.positions)


def test_create_graph_allows_second_order_backprop(batch):
    enc = make_encoder()
    torch.manual_seed(0)
    head = InvariantEnergyHead(IRREPS).to(torch.float64)
    pos = batch.atoms.positions.clone().requires_grad_(True)
    b2 = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, positions=pos))
    e, _ = head(enc(b2, build_hierarchical_graph(b2)).residue_features,
                b2.residues.batch_index, b2.num_graphs)
    f = conservative_force(e, pos, create_graph=True)
    f.pow(2).sum().backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def test_masked_nll_ignores_masked_nodes():
    pred = torch.zeros(4, 3, dtype=torch.float64)
    target = torch.zeros(4, 3, dtype=torch.float64)
    target[3] = 1e6  # a huge error, but masked out
    logvar = torch.zeros(4, 3, dtype=torch.float64)
    mask = torch.tensor([True, True, True, False])
    loss = masked_gaussian_nll(pred, target, logvar, mask)
    assert bool(torch.isfinite(loss))
    assert float(loss) == pytest.approx(0.5 * 3 * torch.log(torch.tensor(2 * torch.pi)).item(),
                                        rel=1e-6)


def test_nll_decreases_when_uncertainty_matches_the_error():
    """A well-calibrated variance must beat both over- and under-confidence."""
    err = 2.0
    pred = torch.zeros(1, 3, dtype=torch.float64)
    target = torch.full((1, 3), err, dtype=torch.float64)
    mask = torch.tensor([True])
    losses = {}
    for lv in (-2.0, torch.log(torch.tensor(err**2)).item(), 4.0):
        logvar = torch.full((1, 3), lv, dtype=torch.float64)
        losses[lv] = float(masked_gaussian_nll(pred, target, logvar, mask))
    best = min(losses, key=losses.get)
    assert abs(best - torch.log(torch.tensor(err**2)).item()) < 1e-9


def test_local_frame_nll_is_rotation_invariant(batch, frames):
    pred = torch.randn(batch.num_residues, 3, dtype=torch.float64)
    target = torch.randn(batch.num_residues, 3, dtype=torch.float64)
    logvar = torch.randn(batch.num_residues, 3, dtype=torch.float64) * 0.1
    mask = torch.ones(batch.num_residues, dtype=torch.bool)
    idx = torch.arange(batch.num_residues)

    q = random_rotation_matrix(torch.Generator().manual_seed(11), dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, torch.zeros(3, dtype=torch.float64))
    a = masked_gaussian_nll(pred, target, logvar, mask, frames=frames, index=idx)
    b = masked_gaussian_nll(pred @ q.T, target @ q.T, logvar, mask,
                            frames=frames_from_batch(moved), index=idx)
    assert torch.allclose(a, b, atol=1e-9)


def test_normalizer_fit_produces_order_one_targets():
    """The scale is the RMS of the vector *magnitude*, so for i.i.d. components
    of standard deviation s it is sqrt(3)*s, not s."""
    torch.manual_seed(0)
    forces = torch.randn(4096, 3) * 40.0
    torques = torch.randn(4096, 3) * 300.0
    n = TargetNormalizer.fit(forces, forces[:512], torques)
    root3 = 3.0**0.5
    assert n.atom_force == pytest.approx(root3 * 40.0, rel=0.1)
    assert n.residue_torque == pytest.approx(root3 * 300.0, rel=0.1)
    # the contract that actually matters: normalised targets have unit RMS
    for tensor, scale in ((forces, n.atom_force), (torques, n.residue_torque)):
        rms = float((tensor / scale).pow(2).sum(-1).mean().sqrt())
        assert rms == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------------------
# the full Phase 1 loss
# --------------------------------------------------------------------------


def _make_output(batch, *, hidden: bool, conservative: bool) -> Phase1Output:
    n_a, n_r = batch.num_atoms, batch.num_residues
    g = torch.Generator().manual_seed(0)
    rnd = lambda n: torch.randn(n, 3, generator=g, dtype=torch.float64)  # noqa: E731
    residual = rnd(n_a)
    cons = rnd(n_a) if conservative else None
    explained = rnd(n_r)
    hid = rnd(n_r) if hidden else None
    return Phase1Output(
        atom_force_mean=residual if cons is None else residual + cons,
        atom_force_residual=residual,
        atom_force_logvar=torch.zeros(n_a, 3, dtype=torch.float64),
        atom_force_conservative=cons,
        residue_explained_force=explained,
        residue_hidden_force=hid,
        residue_force_mean=explained if hid is None else explained + hid,
        residue_torque_mean=rnd(n_r),
        residue_torque_origin=batch.backbone.ca_positions,
        residue_force_logvar=torch.zeros(n_r, 3, dtype=torch.float64),
        residue_torque_logvar=torch.zeros(n_r, 3, dtype=torch.float64),
        aggregated_atom_force=rnd(n_r),
        aggregated_atom_torque=rnd(n_r),
        energy=torch.zeros(batch.num_graphs, dtype=torch.float64),
        residue_energy=torch.zeros(n_r, dtype=torch.float64),
        physics_latent=torch.zeros(n_r, IRREPS.dim, dtype=torch.float64),
        physics_latent_irreps=str(IRREPS),
        target_scope="heavy_atom",
    )


def test_phase1_loss_runs_and_reports_components(batch, frames):
    targets = ResidueSumProjector("heavy_atom")(batch)
    out = _make_output(batch, hidden=False, conservative=True)
    total, comp = phase1_loss(out, batch, targets, frames)
    assert bool(torch.isfinite(total))
    assert set(comp) >= {
        "atom_force_nll", "residue_force_nll", "torque_nll", "hidden_force_mse",
        "aggregation_consistency", "conservative_force_mse", "energy_gauge", "total",
    }


def test_hidden_force_without_a_target_is_refused(batch, frames):
    """An unsupervised additive residual absorbs arbitrary mass."""
    targets = ResidueSumProjector("heavy_atom")(batch)
    out = _make_output(batch, hidden=True, conservative=False)
    with pytest.raises(ValueError, match="unidentifiable"):
        phase1_loss(out, batch, targets, frames)


def test_hidden_force_with_a_target_trains(batch, frames):
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    residual, _ = omitted_atom_residual(allat, heavy)
    out = _make_output(batch, hidden=True, conservative=False)
    total, comp = phase1_loss(out, batch, heavy, frames, hidden_force_target=residual)
    assert comp["hidden_force_mse"] > 0
    assert bool(torch.isfinite(total))


def test_aggregation_consistency_excludes_the_hidden_residual(batch, frames):
    """The consistency term must compare `explained` with the atom aggregate.

    If it used `force_mean` (explained + hidden) instead, the two terms would be
    asked to cancel and the residual could absorb anything.
    """
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    residual, _ = omitted_atom_residual(allat, heavy)

    base = _make_output(batch, hidden=True, conservative=False)
    _, c0 = phase1_loss(base, batch, heavy, frames, hidden_force_target=residual)

    # changing ONLY the hidden force must leave aggregation_consistency untouched
    bumped = dataclasses.replace(
        base,
        residue_hidden_force=base.residue_hidden_force + 5.0,
        residue_force_mean=base.residue_explained_force + base.residue_hidden_force + 5.0,
    )
    _, c1 = phase1_loss(bumped, batch, heavy, frames, hidden_force_target=residual)
    assert c1["aggregation_consistency"] == pytest.approx(c0["aggregation_consistency"])
    assert c1["hidden_force_mse"] != pytest.approx(c0["hidden_force_mse"])


def test_aggregation_consistency_is_zero_when_they_agree(batch, frames):
    targets = ResidueSumProjector("heavy_atom")(batch)
    out = _make_output(batch, hidden=False, conservative=False)
    out = dataclasses.replace(out, aggregated_atom_force=out.residue_explained_force)
    _, comp = phase1_loss(out, batch, targets, frames)
    assert comp["aggregation_consistency"] == pytest.approx(0.0, abs=1e-12)


def test_invalid_force_labels_are_masked_out_of_the_loss(batch, frames):
    fv = batch.atoms.force_valid.clone()
    fv[:] = False
    bad = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, force_valid=fv))
    targets = ResidueSumProjector("heavy_atom")(bad)
    assert not bool(targets.valid.any())
    out = _make_output(bad, hidden=False, conservative=False)
    total, comp = phase1_loss(out, bad, targets, frames)
    assert bool(torch.isfinite(total))
    assert comp["residue_force_nll"] == pytest.approx(0.0)
    assert comp["atom_force_nll"] == pytest.approx(0.0)


def test_loss_backward_is_stable(batch, frames):
    targets = ResidueSumProjector("heavy_atom")(batch)
    n_a, n_r = batch.num_atoms, batch.num_residues
    residual = torch.randn(n_a, 3, dtype=torch.float64, requires_grad=True)
    explained = torch.randn(n_r, 3, dtype=torch.float64, requires_grad=True)
    out = _make_output(batch, hidden=False, conservative=False)
    out = dataclasses.replace(out, atom_force_mean=residual, atom_force_residual=residual,
                              residue_explained_force=explained, residue_force_mean=explained)
    total, _ = phase1_loss(out, batch, targets, frames)
    total.backward()
    assert bool(torch.isfinite(residual.grad).all())
    assert bool(torch.isfinite(explained.grad).all())


def test_loss_weights_are_configurable(batch, frames):
    targets = ResidueSumProjector("heavy_atom")(batch)
    out = _make_output(batch, hidden=False, conservative=True)
    a, _ = phase1_loss(out, batch, targets, frames, weights=LossWeights())
    b, _ = phase1_loss(out, batch, targets, frames,
                       weights=LossWeights(conservative_force=0.0))
    assert float(a) != float(b)


def test_normalizer_changes_the_loss_scale(batch, frames):
    targets = ResidueSumProjector("heavy_atom")(batch)
    out = _make_output(batch, hidden=False, conservative=False)
    a, _ = phase1_loss(out, batch, targets, frames)
    b, _ = phase1_loss(out, batch, targets, frames,
                       normalizer=TargetNormalizer(atom_force=50.0, residue_force=50.0,
                                                   residue_torque=200.0))
    assert float(a) != float(b)
