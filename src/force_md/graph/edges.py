"""Typed edge sets and their geometric features.

Every relation is built as an explicit :class:`EdgeSet` with a stable name and a
per-edge sub-type. Two rules hold everywhere:

**Relations are never merged or de-duplicated.** A residue pair that is both a
sequence neighbour and a spatial neighbour appears in *both* relations. Merging
them would destroy the distinction the encoder conditions on, and silently
dropping the "duplicate" would make the sequence relation depend on geometry.

**No edge may cross a graph boundary.** Batch leakage is checked by
construction (every builder filters on ``batch_index``) and by tests, because a
leaked edge is invisible in the loss but corrupts every prediction in the batch.

Edge direction is ``src -> dst``: messages flow from ``src`` into ``dst``, and
the displacement vector is ``x[dst] - x[src]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "EdgeSet",
    "EdgeGeometry",
    "COVALENT_RADII",
    "build_sequence_edges",
    "build_knn_edges",
    # NOTE ON DISTANCES. Both neighbour searches pass
    # ``compute_mode="donot_use_mm_for_euclid_dist"``. torch.cdist otherwise
    # switches to ``||a||^2 + ||b||^2 - 2 a.b`` above 25 rows, which is
    # catastrophically unstable when the coordinates are large compared with the
    # distances being measured -- exactly this dataset, where absolute box
    # positions reach ~460 A and the cutoffs are 5 and 13 A. Measured: translating
    # a batch by 1000 A changed the atom neighbour list by +1488/-360 edges out of
    # 120,326, asymmetrically, because the squared form biases distances downward.
    # The same structure at a different place in the box must produce the same
    # graph; the direct form costs more memory and delivers that.
    "build_vertical_edges",
    "build_covalent_bonds",
    "build_radius_edges",
    "merge_edge_sets",
    "edge_geometry",
    "edge_spherical_harmonics",
]

#: Covalent radii in Angstrom (Cordero et al. 2008) for the elements mdCATH
#: contains. Used only by the distance-based bond fallback; when a PSF bond list
#: is available it is authoritative and this heuristic is not used.
COVALENT_RADII: dict[int, float] = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}


@dataclass
class EdgeSet:
    """One typed relation, ``src -> dst``.

    Args:
        src: ``[E]`` int64 source node index.
        dst: ``[E]`` int64 destination node index.
        relation: stable name, e.g. ``"backbone__sequence__backbone"``.
        edge_type: ``[E]`` int64 sub-type whose meaning depends on ``relation``
            (sequence offset bucket, intra/inter-residue flag, ...). Preserved
            rather than collapsed, so the encoder can condition on it.
        num_types: number of distinct sub-types, for embedding table sizing.
    """

    src: Tensor
    dst: Tensor
    relation: str
    edge_type: Tensor
    num_types: int

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @property
    def device(self) -> torch.device:
        return self.src.device

    def reverse(self, relation: Optional[str] = None) -> "EdgeSet":
        """Swap ``src``/``dst``, keeping sub-types."""
        return EdgeSet(
            src=self.dst, dst=self.src,
            relation=relation or f"{self.relation}__reverse",
            edge_type=self.edge_type, num_types=self.num_types,
        )

    def to(self, device) -> "EdgeSet":
        return EdgeSet(self.src.to(device), self.dst.to(device), self.relation,
                       self.edge_type.to(device), self.num_types)

    def validate(self, *, num_src_nodes: int, num_dst_nodes: int) -> None:
        if self.src.shape != self.dst.shape or self.src.shape != self.edge_type.shape:
            raise ValueError(f"{self.relation}: src/dst/edge_type shapes disagree")
        if self.src.dtype != torch.int64 or self.dst.dtype != torch.int64:
            raise ValueError(f"{self.relation}: indices must be int64")
        if self.num_edges:
            if int(self.src.max()) >= num_src_nodes or int(self.src.min()) < 0:
                raise ValueError(f"{self.relation}: src index out of range")
            if int(self.dst.max()) >= num_dst_nodes or int(self.dst.min()) < 0:
                raise ValueError(f"{self.relation}: dst index out of range")
            if int(self.edge_type.max()) >= self.num_types:
                raise ValueError(f"{self.relation}: edge_type exceeds num_types")


def merge_edge_sets(sets: list[EdgeSet], relation: str) -> EdgeSet:
    """Concatenate relations into one message pass, keeping their identity.

    Sub-types are offset so that every input relation occupies its own range of
    ``edge_type``. This is *not* the de-duplicating merge the module docstring
    forbids: no edge is dropped and no two relations become indistinguishable --
    the encoder can still condition on which relation an edge came from. It only
    avoids running one scatter per relation over the same node set.
    """
    src_l, dst_l, typ_l = [], [], []
    offset = 0
    for e in sets:
        src_l.append(e.src)
        dst_l.append(e.dst)
        typ_l.append(e.edge_type + offset)
        offset += e.num_types
    src = torch.cat(src_l) if src_l else torch.zeros(0, dtype=torch.int64)
    dst = torch.cat(dst_l) if dst_l else torch.zeros(0, dtype=torch.int64)
    typ = torch.cat(typ_l) if typ_l else torch.zeros(0, dtype=torch.int64)
    src, dst, typ = _sort_edges(src, dst, typ)
    return EdgeSet(src, dst, relation, typ, offset)


def _sort_edges(src: Tensor, dst: Tensor, edge_type: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Lexicographic (dst, src) order, so edge order never depends on the
    kernel, the device or the construction path."""
    if src.numel() == 0:
        return src, dst, edge_type
    key = dst.to(torch.int64) * (int(src.max()) + 1) + src.to(torch.int64)
    order = torch.argsort(key, stable=True)
    return src[order], dst[order], edge_type[order]


# --------------------------------------------------------------------------
# backbone-level relations
# --------------------------------------------------------------------------


def build_sequence_edges(
    batch_index: Tensor,
    chain_index: Tensor,
    resid_original: Tensor,
    *,
    max_offset: int = 2,
    require_contiguous_resid: bool = True,
) -> EdgeSet:
    """Sequence edges between backbone nodes at offsets ``+-1 .. +-max_offset``.

    An edge is created only when both endpoints are in the same graph *and* the
    same chain. With ``require_contiguous_resid`` the residue numbering must also
    advance by exactly the offset: a numbering gap means residues are missing
    from the structure, so the two residues are not actually bonded neighbours
    and a sequence edge would assert a peptide bond that does not exist.

    Args:
        batch_index: ``[N_res]`` graph id.
        chain_index: ``[N_res]`` chain id within the graph.
        resid_original: ``[N_res]`` source-file residue numbering.

    Returns:
        :class:`EdgeSet` with ``edge_type = offset + max_offset`` in
        ``[0, 2*max_offset]`` (the zero offset is unused but kept so that the
        bucket index is a simple shift).
    """
    n = int(batch_index.shape[0])
    device = batch_index.device
    src_l, dst_l, typ_l = [], [], []
    idx = torch.arange(n, device=device)
    for offset in range(-max_offset, max_offset + 1):
        if offset == 0:
            continue
        if offset > 0:
            a, b = idx[:-offset], idx[offset:]
        else:
            a, b = idx[-offset:], idx[:offset]
        if a.numel() == 0:
            continue
        ok = (batch_index[a] == batch_index[b]) & (chain_index[a] == chain_index[b])
        if require_contiguous_resid:
            ok = ok & ((resid_original[b] - resid_original[a]) == offset)
        src_l.append(a[ok])
        dst_l.append(b[ok])
        typ_l.append(torch.full((int(ok.sum()),), offset + max_offset,
                                dtype=torch.int64, device=device))
    if src_l:
        src, dst, typ = torch.cat(src_l), torch.cat(dst_l), torch.cat(typ_l)
    else:
        src = dst = torch.zeros(0, dtype=torch.int64, device=device)
        typ = torch.zeros(0, dtype=torch.int64, device=device)
    src, dst, typ = _sort_edges(src, dst, typ)
    return EdgeSet(src, dst, "backbone__sequence__backbone", typ, 2 * max_offset + 1)


def build_knn_edges(
    positions: Tensor,
    batch_index: Tensor,
    *,
    k: int,
    cutoff: Optional[float] = None,
    relation: str = "backbone__spatial__backbone",
) -> EdgeSet:
    """k-nearest-neighbour edges within each graph, optionally capped by radius.

    Self-edges are excluded. If a graph has fewer than ``k+1`` nodes, every
    other node is connected -- ``k`` is an upper bound, not a guarantee, so a
    short peptide does not silently gain duplicated neighbours.

    Args:
        positions: ``[N, 3]``; for the backbone level these are the CA positions.
        k: neighbours per node.
        cutoff: optional maximum distance; neighbours beyond it are dropped.
    """
    device = positions.device
    src_l, dst_l = [], []
    for g in torch.unique(batch_index):
        sel = (batch_index == g).nonzero(as_tuple=True)[0]
        x = positions[sel]
        m = x.shape[0]
        if m < 2:
            continue
        d = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
        d.fill_diagonal_(float("inf"))
        kk = min(k, m - 1)
        dist, nb = torch.topk(d, kk, dim=1, largest=False)
        dst = sel.unsqueeze(1).expand(-1, kk)
        src = sel[nb]
        keep = torch.ones_like(dist, dtype=torch.bool) if cutoff is None else dist <= cutoff
        src_l.append(src[keep])
        dst_l.append(dst[keep])
    if src_l:
        src, dst = torch.cat(src_l), torch.cat(dst_l)
    else:
        src = dst = torch.zeros(0, dtype=torch.int64, device=device)
    typ = torch.zeros_like(src)
    src, dst, typ = _sort_edges(src, dst, typ)
    return EdgeSet(src, dst, relation, typ, 1)


# --------------------------------------------------------------------------
# vertical relations
# --------------------------------------------------------------------------


def build_vertical_edges(
    child_to_parent: Tensor, relation: str
) -> EdgeSet:
    """Containment edges ``parent -> child`` from a child-to-parent index.

    Cardinality is exact by construction: every child has exactly one parent, so
    ``num_edges == len(child_to_parent)``.
    """
    child = torch.arange(child_to_parent.shape[0], device=child_to_parent.device)
    typ = torch.zeros_like(child)
    return EdgeSet(child_to_parent, child, relation, typ, 1)


# --------------------------------------------------------------------------
# atom-level relations
# --------------------------------------------------------------------------


def build_covalent_bonds(
    positions: Tensor,
    atomic_number: Tensor,
    atom_to_residue: Tensor,
    batch_index: Tensor,
    *,
    tolerance: float = 1.3,
    max_radius: float = 2.5,
    chunk_size: int = 2048,
) -> EdgeSet:
    """Distance-based covalent bonds, as a fallback when no PSF list is given.

    A pair is bonded when ``d < tolerance * (r_cov[z_i] + r_cov[z_j])``. This is
    a heuristic; mdCATH ships an authoritative PSF bond list and the adapter
    should prefer it (see :func:`force_md.data.psf.parse_psf_bonds`). The
    heuristic exists so synthetic fixtures and PSF-less inputs still build.

    Returns:
        Bidirectional :class:`EdgeSet` with ``edge_type`` 0 = intra-residue,
        1 = inter-residue (i.e. the peptide bond and disulfides).
    """
    radii = torch.zeros(int(max(COVALENT_RADII)) + 1, dtype=positions.dtype,
                        device=positions.device)
    for z, r in COVALENT_RADII.items():
        radii[z] = r
    r_atom = radii[atomic_number.clamp(max=len(radii) - 1)]

    src, dst = _pair_candidates(
        positions, batch_index, cutoff=max_radius, chunk_size=chunk_size
    )
    if src.numel():
        d = torch.linalg.norm(positions[dst] - positions[src], dim=-1)
        keep = d < tolerance * (r_atom[src] + r_atom[dst])
        src, dst = src[keep], dst[keep]
    typ = (atom_to_residue[src] != atom_to_residue[dst]).to(torch.int64)
    src, dst, typ = _sort_edges(src, dst, typ)
    return EdgeSet(src, dst, "atom__bonded__atom", typ, 2)


def build_radius_edges(
    positions: Tensor,
    atom_to_residue: Tensor,
    batch_index: Tensor,
    *,
    cutoff: float,
    include_inter_residue: bool = True,
    chunk_size: int = 2048,
) -> EdgeSet:
    """Spatial atom edges within ``cutoff``.

    ``edge_type`` 0 = intra-residue, 1 = inter-residue. Bonded pairs are *not*
    removed: ``atom__bonded__atom`` and ``atom__spatial__atom`` are separate
    relations and the encoder conditions on both.

    The cutoff default is set from a real-data audit; see ``GraphConfig``.
    """
    src, dst = _pair_candidates(positions, batch_index, cutoff=cutoff,
                               chunk_size=chunk_size)
    typ = (atom_to_residue[src] != atom_to_residue[dst]).to(torch.int64)
    if not include_inter_residue:
        keep = typ == 0
        src, dst, typ = src[keep], dst[keep], typ[keep]
    src, dst, typ = _sort_edges(src, dst, typ)
    return EdgeSet(src, dst, "atom__spatial__atom", typ, 2)


def _pair_candidates(
    positions: Tensor, batch_index: Tensor, *, cutoff: float, chunk_size: int
) -> tuple[Tensor, Tensor]:
    """All ordered pairs within ``cutoff`` and within one graph, self excluded.

    Distances are computed in chunks of destination atoms so peak memory is
    ``chunk_size x N_graph`` rather than ``N x N``: a 7.2k-atom domain would
    otherwise need a 200 MB matrix per protein, which does not survive batching.
    A cell list would be asymptotically better and is the place to optimise if
    this becomes the bottleneck.
    """
    device = positions.device
    src_l, dst_l = [], []
    for g in torch.unique(batch_index):
        sel = (batch_index == g).nonzero(as_tuple=True)[0]
        x = positions[sel]
        m = x.shape[0]
        for start in range(0, m, chunk_size):
            stop = min(start + chunk_size, m)
            d = torch.cdist(x[start:stop], x,
                            compute_mode="donot_use_mm_for_euclid_dist")
            rows = torch.arange(start, stop, device=device)
            d[torch.arange(stop - start, device=device), rows] = float("inf")
            hit = (d <= cutoff).nonzero(as_tuple=False)
            if hit.numel() == 0:
                continue
            dst_l.append(sel[rows[hit[:, 0]]])
            src_l.append(sel[hit[:, 1]])
    if not src_l:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty
    return torch.cat(src_l), torch.cat(dst_l)


# --------------------------------------------------------------------------
# edge features
# --------------------------------------------------------------------------


@dataclass
class EdgeGeometry:
    """Geometric features of one relation.

    Args:
        vector: ``[E, 3]`` global displacement ``x[dst] - x[src]``. Equivariant.
        distance: ``[E]`` length. Invariant.
        unit_vector: ``[E, 3]`` normalised displacement. Equivariant.
    """

    vector: Tensor
    distance: Tensor
    unit_vector: Tensor


def edge_geometry(
    src_positions: Tensor,
    dst_positions: Tensor,
    edges: EdgeSet,
    *,
    eps: float = 1e-8,
) -> EdgeGeometry:
    """Displacement, length and direction of every edge.

    ``eps`` clamps the denominator before the division so a zero-length edge
    yields a finite (arbitrary) direction and a finite gradient, rather than a
    NaN that propagates through the whole backward pass.
    """
    vec = dst_positions[edges.dst] - src_positions[edges.src]
    dist = torch.linalg.norm(vec, dim=-1)
    unit = vec / dist.clamp(min=eps).unsqueeze(-1)
    return EdgeGeometry(vector=vec, distance=dist, unit_vector=unit)


def edge_spherical_harmonics(vectors: Tensor, lmax: int = 2) -> Tensor:
    """Real spherical harmonics ``Y_l(v)`` for ``l = 0..lmax``.

    Returns:
        ``[E, (lmax+1)^2]``, i.e. ``[E, 9]`` at ``lmax=2``, in e3nn's
        ``component`` normalisation and irreps order ``1x0e + 1x1o + 1x2e``.

    Under a proper rotation ``Q`` these transform with the Wigner-D matrices of
    their degree; the ``l=0`` block is invariant. Build the Wigner-D on CPU and
    move it, since e3nn caches its generators on CPU and mixing devices raises.
    """
    from e3nn import o3

    irreps = o3.Irreps.spherical_harmonics(lmax)
    return o3.spherical_harmonics(irreps, vectors, normalize=True,
                                  normalization="component")
