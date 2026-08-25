"""The hierarchical encoder: irreps contracts, SE(3) equivariance, chirality.

Equivariance is checked in float64 against the Wigner-D matrices of the actual
irreps, not by eyeballing a norm. e3nn's Wigner-D carries ~1e-6 error of its own,
and the encoder applies two full cycles on top, so tolerances here are ~1e-5
relative -- tightening them further tests e3nn's internals, not this model.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from e3nn import o3

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import apply_rigid_transform, random_rotation_matrix
from force_md.graph import GraphConfig, build_hierarchical_graph, merge_edge_sets
from force_md.nn import (
    AtomInteractionBlock,
    AtomToResiduePool,
    BackboneToResidue,
    EncoderConfig,
    HierarchicalPhysicsEncoder,
    IrrepsConfig,
    ResidueToAtomBroadcast,
    ResidueToBackboneInjection,
    extract_scalars,
    polynomial_cutoff,
)
from force_md.nn.radial import BesselBasis

PLM_DIM = 32


def make_encoder(**overrides) -> HierarchicalPhysicsEncoder:
    cfg = EncoderConfig(plm_dim=PLM_DIM, **overrides)
    torch.manual_seed(0)
    return HierarchicalPhysicsEncoder(cfg).to(torch.float64).eval()


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(6), SyntheticSpec(4)], seed=0,
                           plm_dim=PLM_DIM, dtype=torch.float64)


@pytest.fixture
def graph(batch):
    return build_hierarchical_graph(batch)


@pytest.fixture
def encoder():
    return make_encoder()


# --------------------------------------------------------------------------
# irreps contract
# --------------------------------------------------------------------------


def test_irreps_config_default_is_the_documented_small_config():
    cfg = IrrepsConfig()
    assert str(cfg.node_irreps()) == "64x0e+16x1o+8x2e"
    assert cfg.node_irreps().dim == 64 + 16 * 3 + 8 * 5 == 152
    assert str(cfg.sh_irreps()) == "1x0e+1x1o+1x2e"
    assert cfg.lmax == 2


def test_encoder_output_shapes_and_irreps(batch, graph, encoder):
    out = encoder(batch, graph)
    d = encoder.irreps.dim
    assert out.atom_features.shape == (batch.num_atoms, d)
    assert out.residue_features.shape == (batch.num_residues, d)
    assert out.backbone_features.shape == (batch.num_residues, d)
    assert out.physics_latent is out.residue_features
    assert str(out.irreps) == "64x0e+16x1o+8x2e"


def test_every_block_preserves_the_node_irreps(batch, graph):
    """Each module must return the same irreps it consumed, so blocks compose."""
    irreps = IrrepsConfig().node_irreps()
    sh = IrrepsConfig().sh_irreps()
    n_atom, n_res = batch.num_atoms, batch.num_residues
    x_atom = torch.randn(n_atom, irreps.dim, dtype=torch.float64)
    x_res = torch.randn(n_res, irreps.dim, dtype=torch.float64)
    scalars = torch.randn(n_res, 64, dtype=torch.float64)

    edges = merge_edge_sets([graph.atom_bonded, graph.atom_spatial], "m")
    from force_md.graph import edge_geometry, edge_spherical_harmonics
    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, edges)
    esh = edge_spherical_harmonics(geom.unit_vector, 2)

    block = AtomInteractionBlock(irreps, sh, num_relation_types=4).to(torch.float64)
    assert block(x_atom, edges, esh, geom.distance).shape == (n_atom, irreps.dim)

    pool = AtomToResiduePool(irreps).to(torch.float64)
    pooled = pool(x_atom, batch.atoms.atom_to_residue, n_res)
    assert pooled.shape == (n_res, irreps.dim)

    inj = ResidueToBackboneInjection(irreps, 64).to(torch.float64)
    assert inj(x_res, pooled, scalars).shape == (n_res, irreps.dim)

    b2r = BackboneToResidue(irreps).to(torch.float64)
    assert b2r(x_res, x_res, batch.backbone.residue_to_backbone).shape == (n_res, irreps.dim)

    r2a = ResidueToAtomBroadcast(irreps).to(torch.float64)
    assert r2a(x_atom, x_res, batch.atoms.atom_to_residue).shape == (n_atom, irreps.dim)


# --------------------------------------------------------------------------
# SE(3) equivariance
# --------------------------------------------------------------------------


def _wigner(irreps: o3.Irreps, q: torch.Tensor) -> torch.Tensor:
    """Wigner-D built on CPU (e3nn caches its generators there) in float64."""
    return irreps.D_from_matrix(q.cpu().to(torch.float64))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_encoder_features_are_se3_equivariant(batch, graph, encoder, seed):
    q = random_rotation_matrix(torch.Generator().manual_seed(seed), dtype=torch.float64)
    t = torch.tensor([3.0, -7.5, 0.25], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)
    # rebuild the graph on the moved coordinates: the topology must come out the
    # same, otherwise we would be comparing different graphs
    moved_graph = build_hierarchical_graph(moved)
    assert torch.equal(graph.atom_spatial.src, moved_graph.atom_spatial.src)

    with torch.no_grad():
        a = encoder(batch, graph)
        b = encoder(moved, moved_graph)

    d = _wigner(encoder.irreps, q)
    for name in ("atom_features", "residue_features", "backbone_features"):
        x, y = getattr(a, name), getattr(b, name)
        err = (y - x @ d.T).abs().max().item()
        scale = x.abs().max().item()
        assert err / scale < 1e-5, f"{name}: relative equivariance error {err/scale:.2e}"


def test_scalar_channels_are_invariant(batch, graph, encoder):
    """The l=0 block must not change at all under a rigid motion."""
    q = random_rotation_matrix(torch.Generator().manual_seed(5), dtype=torch.float64)
    t = torch.tensor([-2.0, 8.0, 1.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)
    with torch.no_grad():
        a = encoder(batch, graph)
        b = encoder(moved, build_hierarchical_graph(moved))
    for name in ("atom_features", "residue_features", "backbone_features"):
        sa = extract_scalars(getattr(a, name), encoder.irreps)
        sb = extract_scalars(getattr(b, name), encoder.irreps)
        assert torch.allclose(sa, sb, atol=1e-8), name


def test_translation_alone_changes_nothing(batch, graph, encoder):
    eye = torch.eye(3, dtype=torch.float64)
    t = torch.tensor([100.0, -250.0, 33.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, eye, t)
    with torch.no_grad():
        a = encoder(batch, graph)
        b = encoder(moved, build_hierarchical_graph(moved))
    assert torch.allclose(a.atom_features, b.atom_features, atol=1e-8)
    assert torch.allclose(a.residue_features, b.residue_features, atol=1e-8)


# --------------------------------------------------------------------------
# chirality: reflection must NOT be a symmetry
# --------------------------------------------------------------------------


def test_encoder_is_chirality_sensitive(batch, graph, encoder):
    """Mirroring must change the features.

    An encoder built only from spherical harmonics of relative positions would
    be accidentally E(3)-equivariant and could not distinguish an L-protein from
    its D mirror image. Chirality enters through the residue-frame local
    coordinates in AtomEmbedding; this test is what proves that path is live.
    """
    m = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    mirrored = apply_rigid_transform(batch, m, torch.zeros(3, dtype=torch.float64))
    with torch.no_grad():
        a = encoder(batch, graph)
        b = encoder(mirrored, build_hierarchical_graph(mirrored))
    d = _wigner(encoder.irreps, m)
    # the scalar block would be invariant for an E(3)-equivariant model
    sa = extract_scalars(a.atom_features, encoder.irreps)
    sb = extract_scalars(b.atom_features, encoder.irreps)
    assert not torch.allclose(sa, sb, atol=1e-4), "encoder cannot see chirality"
    assert not torch.allclose(b.atom_features, a.atom_features @ d.T, atol=1e-4)


# --------------------------------------------------------------------------
# masks, empty relations, degenerate inputs
# --------------------------------------------------------------------------


def test_empty_spatial_relation_is_a_no_op_not_a_nan(batch):
    """A cutoff that admits no edge must still produce finite output."""
    graph = build_hierarchical_graph(batch, GraphConfig(atom_cutoff=0.01))
    assert graph.atom_spatial.num_edges == 0
    enc = make_encoder()
    out = enc(batch, graph)
    assert bool(torch.isfinite(out.atom_features).all())
    assert bool(torch.isfinite(out.residue_features).all())


def test_single_residue_protein_has_no_backbone_edges(batch):
    tiny = synthetic_batch([SyntheticSpec(1)], seed=0, plm_dim=PLM_DIM, dtype=torch.float64)
    graph = build_hierarchical_graph(tiny)
    assert graph.backbone_sequence.num_edges == 0
    assert graph.backbone_spatial.num_edges == 0
    out = make_encoder()(tiny, graph)
    assert bool(torch.isfinite(out.residue_features).all())


def test_masked_residue_does_not_produce_nan():
    b = synthetic_batch([SyntheticSpec(6, nonstandard_at=(2,), drop_frame_atom_at=(4,))],
                        seed=0, plm_dim=PLM_DIM, dtype=torch.float64)
    out = make_encoder()(b, build_hierarchical_graph(b))
    assert bool(torch.isfinite(out.atom_features).all())
    assert bool(torch.isfinite(out.residue_features).all())


def test_batching_does_not_leak_between_graphs(batch, graph):
    """Protein 0's features must be identical alone and in a batch."""
    enc = make_encoder()
    single = synthetic_batch([SyntheticSpec(6)], seed=0, plm_dim=PLM_DIM,
                             dtype=torch.float64)
    with torch.no_grad():
        a = enc(single, build_hierarchical_graph(single))
        b = enc(batch, graph)
    assert torch.allclose(a.residue_features, b.residue_features[:6], atol=1e-9)
    assert torch.allclose(a.atom_features, b.atom_features[: single.num_atoms], atol=1e-9)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


def test_forward_and_backward_are_finite(batch, graph):
    enc = make_encoder()
    out = enc(batch, graph)
    out.residue_features.pow(2).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_gradient_flows_to_positions(batch, graph):
    enc = make_encoder()
    pos = batch.atoms.positions.clone().requires_grad_(True)
    b2 = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, positions=pos))
    enc(b2, graph).residue_features.pow(2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())
    assert float(pos.grad.abs().sum()) > 0


# --------------------------------------------------------------------------
# ablation hooks (Checkpoint 8 needs these to be config, not code changes)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag", ["use_plm", "use_temperature", "use_atom_branch",
             "use_backbone_branch", "use_body_order_3"]
)
def test_ablation_flags_build_and_run(batch, graph, flag):
    enc = make_encoder(**{flag: False})
    out = enc(batch, graph)
    assert out.residue_features.shape == (batch.num_residues, enc.irreps.dim)
    assert bool(torch.isfinite(out.residue_features).all())


def test_no_plm_ablation_ignores_the_embedding(batch, graph):
    enc = make_encoder(use_plm=False)
    with torch.no_grad():
        a = enc(batch, graph)
        perturbed = dataclasses.replace(
            batch,
            residues=dataclasses.replace(
                batch.residues,
                plm_embedding=torch.randn_like(batch.residues.plm_embedding),
            ),
        )
        b = enc(perturbed, graph)
    assert torch.allclose(a.residue_features, b.residue_features)


def test_body_order_3_changes_the_function(batch, graph):
    """The TensorSquare path must actually contribute."""
    with torch.no_grad():
        a = make_encoder(use_body_order_3=True)(batch, graph)
        b = make_encoder(use_body_order_3=False)(batch, graph)
    assert not torch.allclose(a.residue_features, b.residue_features)


def test_deeper_config_is_the_same_class(batch, graph):
    """Phase 2 grows depth/width without swapping classes."""
    big = make_encoder(num_cycles=3, irreps=IrrepsConfig(scalar_channels=32,
                                                         vector_channels=8,
                                                         tensor_channels=4))
    assert type(big) is HierarchicalPhysicsEncoder
    out = big(batch, graph)
    assert out.irreps.dim == 32 + 8 * 3 + 4 * 5


# --------------------------------------------------------------------------
# radial basis
# --------------------------------------------------------------------------


def test_cutoff_envelope_vanishes_at_the_boundary():
    r = torch.tensor([0.0, 2.5, 4.999, 5.0, 6.0], dtype=torch.float64)
    env = polynomial_cutoff(r, 5.0)
    assert abs(float(env[0]) - 1.0) < 1e-12
    assert float(env[3]) == 0.0
    assert float(env[4]) == 0.0
    assert 0.0 < float(env[2]) < 1e-3


def test_bessel_basis_is_finite_including_at_zero_distance():
    basis = BesselBasis(5.0, 8).to(torch.float64)
    r = torch.tensor([0.0, 1e-9, 1.0, 4.9, 5.0, 7.0], dtype=torch.float64,
                     requires_grad=True)
    out = basis(r)
    assert out.shape == (6, 8)
    assert bool(torch.isfinite(out).all())
    assert torch.allclose(out[4], torch.zeros(8, dtype=torch.float64))
    out.sum().backward()
    assert bool(torch.isfinite(r.grad).all())


def test_message_is_continuous_across_the_cutoff():
    """An atom crossing r_cut must not make the message jump."""
    basis = BesselBasis(5.0, 8).to(torch.float64)
    inside = basis(torch.tensor([4.9999], dtype=torch.float64))
    assert float(inside.abs().max()) < 1e-8
