"""Backbone torsions and sequence adjacency.

``phi`` and ``psi`` are reported because Ca RMSD and residue-frame rotation can
both look reasonable while the backbone itself is not a protein: they measure
where residues are and how they are oriented, not whether the chain that connects
them is physical. A torsion error is the cheapest check that stays sensitive to
that.

The sign convention is IUPAC and is the same one ``tests/conftest.py`` fixes with
a self-test; ``test_dihedral_matches_the_reference_convention`` pins this
implementation to that one. Getting the sign backwards mirrors every torsion,
which for a chiral molecule is not a cosmetic difference -- it would report a
left-handed helix as right-handed.

Adjacency follows the same rule as ``graph.edges.build_sequence_edges``: two
residues are sequence neighbours only when they share a graph and a chain **and**
their source-file residue numbers differ by exactly one. A numbering gap means
residues are missing from the structure, so the peptide bond that ``psi`` is
defined across does not exist and the torsion is not computed rather than being
computed across a hole.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "dihedral_angle",
    "sequence_neighbours",
    "backbone_torsions",
    "wrap_to_pi",
]


def wrap_to_pi(angle: Tensor) -> Tensor:
    """Wrap radians into ``[-pi, pi)``.

    Required for any difference of angles: ``179 deg`` and ``-179 deg`` are two
    degrees apart, and an unwrapped subtraction reports 358.

    The half-open end is at ``+pi``: an angle of exactly ``pi`` comes back as
    ``-pi``. For the use here -- the *magnitude* of an angular error -- the sign of
    a half turn is not meaningful and both give the same answer; the convention is
    stated only so that no caller depends on the other one.
    """
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def dihedral_angle(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor,
                   *, eps: float = 1e-8) -> Tensor:
    """Signed dihedral ``p0-p1-p2-p3`` in radians, ``[N]``, IUPAC convention.

    Uses ``atan2`` of the projected components rather than ``arccos`` of a
    normalised dot product, for the same reason as everywhere else in this
    project: ``arccos`` is ill conditioned exactly where the angle is small.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp(min=eps)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2(
        (torch.linalg.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1)
    )


def sequence_neighbours(
    batch_index: Tensor,
    chain_index: Tensor,
    resid_original: Tensor,
    *,
    require_contiguous_resid: bool = True,
) -> tuple[Tensor, Tensor]:
    """Previous/next residue index along the chain, ``-1`` where there is none.

    Args:
        batch_index / chain_index / resid_original: ``[N_res]``, as carried by
            :class:`force_md.data.contracts.ResidueSemanticBatch`. ``resid_original``
            is the source-file numbering, **not** a 0-based index.

    Returns:
        ``(previous [N_res], next [N_res])`` int64.
    """
    n = int(batch_index.shape[0])
    device = batch_index.device
    previous = torch.full((n,), -1, dtype=torch.int64, device=device)
    following = torch.full((n,), -1, dtype=torch.int64, device=device)
    if n < 2:
        return previous, following

    idx = torch.arange(n, device=device)
    a, b = idx[:-1], idx[1:]
    linked = (batch_index[a] == batch_index[b]) & (chain_index[a] == chain_index[b])
    if require_contiguous_resid:
        linked = linked & ((resid_original[b] - resid_original[a]) == 1)
    previous[b[linked]] = a[linked]
    following[a[linked]] = b[linked]
    return previous, following


def backbone_torsions(
    n_positions: Tensor,
    ca_positions: Tensor,
    c_positions: Tensor,
    previous: Tensor,
    following: Tensor,
    *,
    valid: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``phi`` and ``psi`` per residue, in radians.

    ``phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)``,
    ``psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})``.

    Args:
        n_positions / ca_positions / c_positions: ``[N_res, 3]``.
        previous / following: from :func:`sequence_neighbours`.
        valid: optional ``[N_res]`` bool; a residue with a degenerate or missing
            backbone invalidates the torsions of both itself and its neighbour.

    Returns:
        ``(phi, psi, phi_valid, psi_valid)``. Angles at invalid positions are 0
        and must be masked, not read.
    """
    n_res = ca_positions.shape[0]
    if valid is None:
        valid = torch.ones(n_res, dtype=torch.bool, device=ca_positions.device)

    has_prev = previous >= 0
    has_next = following >= 0
    prev_safe = previous.clamp(min=0)
    next_safe = following.clamp(min=0)

    phi = dihedral_angle(
        c_positions[prev_safe], n_positions, ca_positions, c_positions
    )
    psi = dihedral_angle(
        n_positions, ca_positions, c_positions, n_positions[next_safe]
    )
    phi_valid = has_prev & valid & valid[prev_safe]
    psi_valid = has_next & valid & valid[next_safe]
    zero = torch.zeros((), dtype=phi.dtype, device=phi.device)
    return (
        torch.where(phi_valid, phi, zero),
        torch.where(psi_valid, psi, zero),
        phi_valid,
        psi_valid,
    )
