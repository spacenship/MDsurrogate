"""Hierarchical topology: relations, cardinality, leakage, determinism, features."""

from __future__ import annotations

import pytest
import torch

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import (
    apply_rigid_transform,
    frames_from_batch,
    random_rotation_matrix,
    to_local_vectors,
)
from force_md.graph import (
    GraphConfig,
    build_hierarchical_graph,
    build_knn_edges,
    build_sequence_edges,
    edge_geometry,
    edge_spherical_harmonics,
)


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(9), SyntheticSpec(5)], seed=0, dtype=torch.float64)


@pytest.fixture
def graph(batch):
    return build_hierarchical_graph(batch)


def test_graph_validates(batch, graph):
    graph.validate(batch)
    counts = graph.edge_counts()
    assert set(counts) == {
        "backbone__sequence__backbone", "backbone__spatial__backbone",
        "backbone__owns__residue", "residue__contains__atom",
        "atom__bonded__atom", "atom__spatial__atom",
    }
    assert all(v > 0 for v in counts.values())


# --------------------------------------------------------------------------
# sequence edges and chain breaks
# --------------------------------------------------------------------------


def test_sequence_offsets_are_pm1_and_pm2(batch, graph):
    e = graph.backbone_sequence
    offsets = (e.dst - e.src).tolist()
    assert set(offsets) == {-2, -1, 1, 2}
    # edge_type encodes the offset as offset + max_offset
    assert torch.equal(e.edge_type, (e.dst - e.src) + 2)


def test_sequence_edges_do_not_cross_a_chain_break():
    batch = synthetic_batch([SyntheticSpec(10, num_chains=2)], seed=0, dtype=torch.float64)
    e = build_hierarchical_graph(batch).backbone_sequence
    chain = batch.residues.chain_index
    assert bool((chain[e.src] == chain[e.dst]).all()), "sequence edge bridged two chains"
    # the pair straddling the boundary must be absent
    first_of_2 = int((chain == 1).nonzero()[0])
    bridging = ((e.src == first_of_2 - 1) & (e.dst == first_of_2)).any()
    assert not bool(bridging)


def test_sequence_edges_respect_a_residue_numbering_gap():
    """A numbering gap means residues are missing, so there is no peptide bond."""
    batch = synthetic_batch([SyntheticSpec(6)], seed=0, dtype=torch.float64)
    resid = batch.residues.resid_original.clone()
    resid[3:] += 5  # gap between residue 2 and 3
    e = build_sequence_edges(batch.backbone.batch_index, batch.residues.chain_index,
                             resid, max_offset=2, require_contiguous_resid=True)
    pairs = set(zip(e.src.tolist(), e.dst.tolist()))
    assert (2, 3) not in pairs and (3, 2) not in pairs
    assert (0, 1) in pairs and (3, 4) in pairs

    loose = build_sequence_edges(batch.backbone.batch_index, batch.residues.chain_index,
                                 resid, max_offset=2, require_contiguous_resid=False)
    assert (2, 3) in set(zip(loose.src.tolist(), loose.dst.tolist()))


def test_sequence_edges_do_not_cross_graphs(batch, graph):
    e = graph.backbone_sequence
    g = batch.residues.batch_index
    assert bool((g[e.src] == g[e.dst]).all())


# --------------------------------------------------------------------------
# spatial edges
# --------------------------------------------------------------------------


def test_knn_has_no_self_edges_and_respects_k(batch):
    e = build_knn_edges(batch.backbone.ca_positions, batch.backbone.batch_index, k=4)
    assert not bool((e.src == e.dst).any())
    counts = torch.bincount(e.dst, minlength=batch.num_residues)
    assert int(counts.max()) <= 4


def test_knn_k_is_capped_by_graph_size():
    """A 3-residue graph cannot have 16 distinct neighbours; it must not fake them."""
    batch = synthetic_batch([SyntheticSpec(3)], seed=0, dtype=torch.float64)
    e = build_knn_edges(batch.backbone.ca_positions, batch.backbone.batch_index, k=16)
    counts = torch.bincount(e.dst, minlength=3)
    assert counts.tolist() == [2, 2, 2]
    assert not bool((e.src == e.dst).any())


def test_knn_cutoff_drops_far_neighbours(batch):
    near = build_knn_edges(batch.backbone.ca_positions, batch.backbone.batch_index,
                           k=16, cutoff=6.0)
    far = build_knn_edges(batch.backbone.ca_positions, batch.backbone.batch_index, k=16)
    assert near.num_edges < far.num_edges
    geom = edge_geometry(batch.backbone.ca_positions, batch.backbone.ca_positions, near)
    assert float(geom.distance.max()) <= 6.0 + 1e-9


def test_atom_spatial_respects_cutoff(batch):
    cfg = GraphConfig(atom_cutoff=4.0)
    g = build_hierarchical_graph(batch, cfg)
    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, g.atom_spatial)
    assert float(geom.distance.max()) <= 4.0 + 1e-9
    assert not bool((g.atom_spatial.src == g.atom_spatial.dst).any())


def test_atom_spatial_intra_inter_types(batch, graph):
    e = graph.atom_spatial
    a2r = batch.atoms.atom_to_residue
    same = a2r[e.src] == a2r[e.dst]
    assert torch.equal(e.edge_type, (~same).to(torch.int64))
    assert int((e.edge_type == 0).sum()) > 0 and int((e.edge_type == 1).sum()) > 0


def test_intra_residue_only_mode(batch):
    g = build_hierarchical_graph(batch, GraphConfig(atom_include_inter_residue=False))
    assert bool((g.atom_spatial.edge_type == 0).all())
    a2r = batch.atoms.atom_to_residue
    assert bool((a2r[g.atom_spatial.src] == a2r[g.atom_spatial.dst]).all())


def test_larger_cutoff_is_a_superset(batch):
    small = build_hierarchical_graph(batch, GraphConfig(atom_cutoff=4.0)).atom_spatial
    large = build_hierarchical_graph(batch, GraphConfig(atom_cutoff=6.0)).atom_spatial
    s = set(zip(small.src.tolist(), small.dst.tolist()))
    l = set(zip(large.src.tolist(), large.dst.tolist()))
    assert s < l


# --------------------------------------------------------------------------
# bonds
# --------------------------------------------------------------------------


def test_covalent_bonds_are_symmetric_and_short(batch, graph):
    e = graph.atom_bonded
    pairs = set(zip(e.src.tolist(), e.dst.tolist()))
    assert all((d, s) in pairs for s, d in pairs), "bond list must be symmetric"
    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, e)
    assert float(geom.distance.max()) < 2.5


def test_bonds_include_the_backbone_and_the_peptide_bond(batch, graph):
    """N-CA and CA-C within a residue, and C(i)-N(i+1) across residues."""
    e = graph.atom_bonded
    pairs = set(zip(e.src.tolist(), e.dst.tolist()))
    a2r = batch.atoms.atom_to_residue
    from force_md.data import residue_constants as rc
    name = lambda i: rc.ATOM_NAMES[int(batch.atoms.atom_name_id[i])]  # noqa: E731

    idx0 = (a2r == 0).nonzero(as_tuple=True)[0]
    n0 = next(int(i) for i in idx0 if name(i) == "N")
    ca0 = next(int(i) for i in idx0 if name(i) == "CA")
    c0 = next(int(i) for i in idx0 if name(i) == "C")
    assert (n0, ca0) in pairs and (ca0, c0) in pairs

    idx1 = (a2r == 1).nonzero(as_tuple=True)[0]
    n1 = next(int(i) for i in idx1 if name(i) == "N")
    assert (c0, n1) in pairs, "peptide bond missing"
    inter = e.edge_type[((e.src == c0) & (e.dst == n1)).nonzero()[0]]
    assert int(inter) == 1, "peptide bond must be typed inter-residue"


def test_explicit_bonds_override_the_heuristic(batch):
    explicit = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)  # 2 bonds
    g = build_hierarchical_graph(batch, explicit_bonds=explicit)
    assert g.atom_bonded.num_edges == 4  # symmetrised
    assert set(zip(g.atom_bonded.src.tolist(), g.atom_bonded.dst.tolist())) == {
        (0, 1), (1, 0), (1, 2), (2, 1)
    }


# --------------------------------------------------------------------------
# vertical relations
# --------------------------------------------------------------------------


def test_vertical_cardinality_is_exact(batch, graph):
    assert graph.residue_contains_atom.num_edges == batch.num_atoms
    assert graph.backbone_owns_residue.num_edges == batch.num_residues
    # every atom appears exactly once as a child
    counts = torch.bincount(graph.residue_contains_atom.dst, minlength=batch.num_atoms)
    assert bool((counts == 1).all())


def test_reverse_relations_are_consistent(graph):
    fwd = graph.residue_contains_atom
    rev = graph.atom_belongs_to_residue
    assert torch.equal(fwd.src, rev.dst) and torch.equal(fwd.dst, rev.src)
    assert rev.relation == "atom__belongs_to__residue"
    assert graph.residue_owned_by_backbone.relation == "residue__owned_by__backbone"


def test_containment_matches_atom_to_residue(batch, graph):
    e = graph.residue_contains_atom
    assert torch.equal(e.src[e.dst], batch.atoms.atom_to_residue)


# --------------------------------------------------------------------------
# leakage, determinism, duplication
# --------------------------------------------------------------------------


def test_no_relation_crosses_a_graph_boundary(batch, graph):
    graph.validate(batch)  # does the check for all six
    for name, e in graph.relations().items():
        if e.num_edges == 0:
            continue
        gs = batch.atoms.batch_index if name.startswith("atom") else batch.residues.batch_index
        gd = batch.atoms.batch_index if name.endswith("atom") else batch.residues.batch_index
        assert bool((gs[e.src] == gd[e.dst]).all()), name


def test_batching_does_not_change_the_first_graph():
    single = synthetic_batch([SyntheticSpec(6)], seed=0, dtype=torch.float64)
    pair = synthetic_batch([SyntheticSpec(6), SyntheticSpec(11)], seed=0, dtype=torch.float64)
    g1 = build_hierarchical_graph(single)
    g2 = build_hierarchical_graph(pair)
    n_res, n_atom = single.num_residues, single.num_atoms
    e1, e2 = g1.backbone_sequence, g2.backbone_sequence
    keep = e2.dst < n_res
    assert torch.equal(e1.src, e2.src[keep]) and torch.equal(e1.dst, e2.dst[keep])
    a1, a2 = g1.atom_spatial, g2.atom_spatial
    keep_a = a2.dst < n_atom
    assert torch.equal(a1.src, a2.src[keep_a])


def test_edge_order_is_deterministic(batch):
    a = build_hierarchical_graph(batch)
    b = build_hierarchical_graph(batch)
    for name, e in a.relations().items():
        f = b.relations()[name]
        assert torch.equal(e.src, f.src) and torch.equal(e.dst, f.dst), name
        assert torch.equal(e.edge_type, f.edge_type), name


def test_edges_are_sorted_lexicographically(batch, graph):
    for name, e in graph.relations().items():
        if e.num_edges < 2 or name.endswith("__residue") or name.endswith("__atom"):
            pass
        key = e.dst * (int(e.src.max()) + 1) + e.src if e.num_edges else None
        if key is not None and name in ("backbone__sequence__backbone",
                                        "backbone__spatial__backbone",
                                        "atom__bonded__atom", "atom__spatial__atom"):
            assert bool((key[1:] >= key[:-1]).all()), f"{name} not sorted"


def test_sequence_and_spatial_relations_are_not_deduplicated(batch, graph):
    """A pair may be both a sequence and a spatial neighbour; both must survive."""
    seq = set(zip(graph.backbone_sequence.src.tolist(), graph.backbone_sequence.dst.tolist()))
    spa = set(zip(graph.backbone_spatial.src.tolist(), graph.backbone_spatial.dst.tolist()))
    assert seq & spa, "expected overlap between sequence and spatial neighbours"


def test_bonded_and_spatial_relations_overlap(batch, graph):
    bonded = set(zip(graph.atom_bonded.src.tolist(), graph.atom_bonded.dst.tolist()))
    spatial = set(zip(graph.atom_spatial.src.tolist(), graph.atom_spatial.dst.tolist()))
    assert bonded <= spatial, "bonded pairs are within the spatial cutoff too"
    assert len(spatial) > len(bonded)


# --------------------------------------------------------------------------
# edge features and symmetry
# --------------------------------------------------------------------------


def test_distance_is_invariant_and_vector_is_equivariant(batch, graph):
    g = torch.Generator().manual_seed(0)
    q = random_rotation_matrix(g, dtype=torch.float64)
    t = torch.tensor([2.0, -3.0, 9.0], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)

    a = edge_geometry(batch.atoms.positions, batch.atoms.positions, graph.atom_spatial)
    b = edge_geometry(moved.atoms.positions, moved.atoms.positions, graph.atom_spatial)
    assert torch.allclose(a.distance, b.distance, atol=1e-10)
    assert torch.allclose(b.vector, a.vector @ q.T, atol=1e-10)


def test_local_relative_coordinates_are_invariant(batch, graph):
    """The relative vector expressed in the destination residue frame is the
    invariant geometric input the encoder actually consumes."""
    g = torch.Generator().manual_seed(1)
    q = random_rotation_matrix(g, dtype=torch.float64)
    t = torch.tensor([-4.0, 1.0, 0.5], dtype=torch.float64)
    moved = apply_rigid_transform(batch, q, t)
    dst_res = batch.atoms.atom_to_residue[graph.atom_spatial.dst]

    v0 = edge_geometry(batch.atoms.positions, batch.atoms.positions, graph.atom_spatial).vector
    v1 = edge_geometry(moved.atoms.positions, moved.atoms.positions, graph.atom_spatial).vector
    l0 = to_local_vectors(v0, frames_from_batch(batch), dst_res)
    l1 = to_local_vectors(v1, frames_from_batch(moved), dst_res)
    assert torch.allclose(l0, l1, atol=1e-9)


def test_spherical_harmonics_shape_and_l0_invariance(batch, graph):
    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, graph.atom_spatial)
    sh = edge_spherical_harmonics(geom.unit_vector, lmax=2)
    assert sh.shape == (graph.atom_spatial.num_edges, 9)  # 1 + 3 + 5

    g = torch.Generator().manual_seed(2)
    q = random_rotation_matrix(g, dtype=torch.float64)
    sh_rot = edge_spherical_harmonics(geom.unit_vector @ q.T, lmax=2)
    assert torch.allclose(sh[:, :1], sh_rot[:, :1], atol=1e-10), "l=0 must be invariant"


def test_spherical_harmonics_transform_with_wigner_d(batch, graph):
    """l=1 and l=2 blocks must rotate by their Wigner-D matrices."""
    from e3nn import o3

    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, graph.atom_spatial)
    sh = edge_spherical_harmonics(geom.unit_vector, lmax=2)

    g = torch.Generator().manual_seed(3)
    q = random_rotation_matrix(g, dtype=torch.float64)
    sh_rot = edge_spherical_harmonics(geom.unit_vector @ q.T, lmax=2)

    irreps = o3.Irreps.spherical_harmonics(2)
    # Build D on CPU: e3nn caches its Wigner generators there and mixing devices
    # raises. Its internals are float32, so the achievable accuracy is ~1e-5
    # even in float64 -- tightening this tolerance makes the test flaky, it does
    # not make the model more equivariant.
    d = irreps.D_from_matrix(q)
    assert torch.allclose(sh_rot, sh @ d.T, atol=1e-5)


def test_zero_length_edge_gives_finite_direction_and_gradient():
    pos = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    from force_md.graph import EdgeSet
    e = EdgeSet(torch.tensor([0]), torch.tensor([1]), "test",
                torch.tensor([0]), 1)
    geom = edge_geometry(pos, pos, e)
    assert bool(torch.isfinite(geom.unit_vector).all())
    geom.distance.sum().backward()
    assert bool(torch.isfinite(pos.grad).all())


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------


def test_single_residue_graph_builds():
    batch = synthetic_batch([SyntheticSpec(1)], seed=0, dtype=torch.float64)
    g = build_hierarchical_graph(batch)
    g.validate(batch)
    assert g.backbone_sequence.num_edges == 0
    assert g.backbone_spatial.num_edges == 0
    assert g.residue_contains_atom.num_edges == batch.num_atoms


def test_tiny_cutoff_yields_empty_spatial_relation(batch):
    g = build_hierarchical_graph(batch, GraphConfig(atom_cutoff=0.01))
    assert g.atom_spatial.num_edges == 0
    g.validate(batch)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_matches_between_cpu_and_cuda(batch):
    cpu = build_hierarchical_graph(batch)
    gpu = build_hierarchical_graph(batch.to("cuda:0"))
    for name, e in cpu.relations().items():
        f = gpu.relations()[name]
        assert torch.equal(e.src, f.src.cpu()), name
        assert torch.equal(e.dst, f.dst.cpu()), name


def test_neighbour_list_is_translation_invariant():
    """The same structure at a different place in the box must give the same graph.

    torch.cdist defaults to ``||a||^2 + ||b||^2 - 2 a.b`` above 25 rows, which
    loses precision in proportion to the *coordinates* rather than the distances.
    With mdCATH's absolute box positions that biases distances downward and flips
    edges at the cutoff: measured +1488/-360 out of 120,326 atom edges under a
    1000 A translation before this was pinned.
    """
    from force_md.graph.edges import build_knn_edges, build_radius_edges

    g = torch.Generator().manual_seed(0)
    n = 400
    x = torch.rand(n, 3, generator=g) * 40.0
    batch_index = torch.zeros(n, dtype=torch.int64)
    atom_to_residue = torch.arange(n) // 8

    def edge_set(e):
        return set(zip(e.src.tolist(), e.dst.tolist()))

    for shift in (100.0, 1000.0, 5000.0):
        moved = x + shift

        knn_a = edge_set(build_knn_edges(x, batch_index, k=16, cutoff=13.0))
        knn_b = edge_set(build_knn_edges(moved, batch_index, k=16, cutoff=13.0))
        assert knn_a == knn_b, (
            f"kNN: translating by {shift} A changed the edge set by "
            f"+{len(knn_b - knn_a)}/-{len(knn_a - knn_b)} of {len(knn_a)}"
        )

        rad_a = edge_set(build_radius_edges(x, atom_to_residue, batch_index, cutoff=5.0))
        rad_b = edge_set(
            build_radius_edges(moved, atom_to_residue, batch_index, cutoff=5.0)
        )
        assert rad_a == rad_b, (
            f"radius: translating by {shift} A changed the edge set by "
            f"+{len(rad_b - rad_a)}/-{len(rad_a - rad_b)} of {len(rad_a)}"
        )
