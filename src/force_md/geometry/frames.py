"""Residue frames and residue-local coordinates.

Convention (fixed here once, asserted by tests, reused by every later module)::

    r_i = CA_i                                   # origin
    e1  = normalize(C_i - CA_i)
    e2  = normalize((N_i - CA_i) - (N_i - CA_i, e1) e1)
    e3  = e1 x e2
    R_i = [e1 | e2 | e3]                         # COLUMNS are the local axes,
                                                 # expressed in global coordinates
    y_ia = R_i^T (x_ia - r_i)                    # global point -> local point
    x_ia = R_i y_ia + r_i                        # local point  -> global point

Because the columns of ``R_i`` are the local axes in global coordinates, ``R_i``
maps *local* vectors to *global* ones and ``R_i^T`` does the reverse. Getting
this backwards transposes every equivariant feature, so it is asserted directly
in :func:`test_rotation_columns_are_the_local_axes`.

Symmetry: this construction is equivariant under proper rotations and
translations -- SE(3) -- and deliberately **not** under reflection. ``e3`` is a
cross product, so a mirrored structure does not produce the mirrored frame. That
asymmetry is what keeps protein chirality a modelled property rather than a
symmetry the network is free to average away.

Degenerate frames (missing or collinear N/CA/C) are reported through ``valid``
and replaced by the identity rotation. Denominators are clamped *before*
division rather than patched afterwards with ``torch.where``: a NaN produced in
an unselected branch still poisons the backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data.contracts import HierarchicalProteinBatch

__all__ = [
    "ResidueFrames",
    "build_residue_frames",
    "frames_from_batch",
    "to_local_points",
    "to_global_points",
    "to_local_vectors",
    "to_global_vectors",
    "atom_local_coordinates",
    "frame_atom_indices",
    "link_backbone_to_atom_positions",
    "random_rotation_matrix",
    "apply_rigid_transform",
]

#: Below this length (in the batch's length unit) an axis is treated as
#: degenerate. 1e-3 A is far smaller than any real bond fluctuation and far
#: larger than float32 noise on a ~1.5 A bond.
DEGENERATE_EPS = 1e-3


@dataclass
class ResidueFrames:
    """Per-residue rigid frame.

    Args:
        rotation: ``[N_res, 3, 3]``. Columns are the local axes in global
            coordinates, so ``rotation @ y`` is a global vector.
        origin: ``[N_res, 3]``, the CA position in global coordinates.
        valid: ``[N_res]`` bool. False where N/CA/C were missing or collinear;
            ``rotation`` is the identity there and must not enter a loss.
    """

    rotation: Tensor
    origin: Tensor
    valid: Tensor

    @property
    def num_residues(self) -> int:
        return int(self.origin.shape[0])

    def to(self, device) -> "ResidueFrames":
        return ResidueFrames(
            self.rotation.to(device), self.origin.to(device), self.valid.to(device)
        )


def build_residue_frames(
    n_positions: Tensor,
    ca_positions: Tensor,
    c_positions: Tensor,
    *,
    eps: float = DEGENERATE_EPS,
    prior_valid: Optional[Tensor] = None,
) -> ResidueFrames:
    """Build right-handed residue frames from backbone N/CA/C.

    Args:
        n_positions / ca_positions / c_positions: ``[N_res, 3]`` global positions.
        eps: degeneracy threshold on the two constructed axis lengths.
        prior_valid: optional ``[N_res]`` bool AND-ed into the result, for
            residues already known to be unusable (e.g. missing atoms).

    Returns:
        :class:`ResidueFrames` with ``R^T R = I`` and ``det(R) = +1`` on valid
        rows and the identity on invalid ones.

    The operation is differentiable with respect to all three inputs, including
    at degenerate configurations, where the gradient is finite (though the frame
    itself is meaningless and masked out).
    """
    v1 = c_positions - ca_positions
    len1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
    e1 = v1 / len1.clamp(min=eps)

    u = n_positions - ca_positions
    v2 = u - (u * e1).sum(-1, keepdim=True) * e1
    len2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
    e2 = v2 / len2.clamp(min=eps)

    e3 = torch.linalg.cross(e1, e2, dim=-1)

    # Columns are the local axes -> stack along the last dim.
    rotation = torch.stack([e1, e2, e3], dim=-1)

    finite = (
        torch.isfinite(n_positions).all(-1)
        & torch.isfinite(ca_positions).all(-1)
        & torch.isfinite(c_positions).all(-1)
    )
    valid = (len1.squeeze(-1) > eps) & (len2.squeeze(-1) > eps) & finite
    if prior_valid is not None:
        valid = valid & prior_valid

    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(rotation)
    rotation = torch.where(valid[:, None, None], rotation, eye)
    origin = torch.where(valid[:, None], ca_positions, torch.zeros_like(ca_positions))
    return ResidueFrames(rotation=rotation, origin=origin, valid=valid)


def frames_from_batch(
    batch: HierarchicalProteinBatch, *, eps: float = DEGENERATE_EPS
) -> ResidueFrames:
    """Build frames from a batch's backbone level, honouring ``frame_valid``."""
    bb = batch.backbone
    return build_residue_frames(
        bb.n_positions, bb.ca_positions, bb.c_positions,
        eps=eps, prior_valid=bb.frame_valid,
    )


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------


def to_local_points(points: Tensor, frames: ResidueFrames, index: Tensor) -> Tensor:
    """Global points -> residue-local points: ``y = R^T (x - r)``.

    Args:
        points: ``[N, 3]`` global positions.
        frames: frames to map into.
        index: ``[N]`` int64, which residue frame each point belongs to.

    Returns:
        ``[N, 3]`` local coordinates, invariant under a global rigid motion.
    """
    r = frames.origin[index]
    rot = frames.rotation[index]
    return torch.einsum("nji,nj->ni", rot, points - r)


def to_global_points(local: Tensor, frames: ResidueFrames, index: Tensor) -> Tensor:
    """Residue-local points -> global points: ``x = R y + r``."""
    r = frames.origin[index]
    rot = frames.rotation[index]
    return torch.einsum("nij,nj->ni", rot, local) + r


def to_local_vectors(vectors: Tensor, frames: ResidueFrames, index: Tensor) -> Tensor:
    """Global vectors -> local: ``R^T v``. No translation (forces, velocities)."""
    return torch.einsum("nji,nj->ni", frames.rotation[index], vectors)


def to_global_vectors(vectors: Tensor, frames: ResidueFrames, index: Tensor) -> Tensor:
    """Local vectors -> global: ``R v``. No translation."""
    return torch.einsum("nij,nj->ni", frames.rotation[index], vectors)


def atom_local_coordinates(
    batch: HierarchicalProteinBatch, frames: Optional[ResidueFrames] = None
) -> tuple[Tensor, ResidueFrames]:
    """Residue-local coordinates ``y_ia`` of every atom.

    Returns:
        ``([N_atom, 3], frames)``. Atoms of a residue with a degenerate frame get
        coordinates relative to the identity frame; use ``frames.valid`` (lifted
        through ``atom_to_residue``) to mask them.
    """
    if frames is None:
        frames = frames_from_batch(batch)
    y = to_local_points(batch.atoms.positions, frames, batch.atoms.atom_to_residue)
    return y, frames


# --------------------------------------------------------------------------
# rigid-motion helpers (shared by every equivariance test downstream)
# --------------------------------------------------------------------------


def frame_atom_indices(batch: HierarchicalProteinBatch) -> tuple[Tensor, Tensor]:
    """Locate each residue's N, CA and C atom inside the flat atom array.

    Matching is by **atom name**, never by element: CHARMM's N-terminal cap
    contributes a carbon called ``CAY``, and an element-based search would pick
    it up as the alpha carbon and silently build the frame from the wrong atom.

    Returns:
        ``(indices [N_res, 3], complete [N_res])`` where the columns are N, CA, C
        and ``complete`` is False for any residue missing one of them (its index
        row is then ``-1`` and must not be used).
    """
    from ..data.residue_constants import atom_name_id

    device = batch.atoms.positions.device
    n_res = batch.num_residues
    indices = torch.full((n_res, 3), -1, dtype=torch.int64, device=device)
    names = batch.atoms.atom_name_id
    for slot, name in enumerate(("N", "CA", "C")):
        hit = (names == atom_name_id(name)).nonzero(as_tuple=True)[0]
        indices[batch.atoms.atom_to_residue[hit], slot] = hit
    return indices, (indices >= 0).all(dim=-1)


def link_backbone_to_atom_positions(
    batch: HierarchicalProteinBatch,
) -> HierarchicalProteinBatch:
    """Make backbone N/CA/C positions gathers of ``atoms.positions``.

    Why this matters for the energy branch: ``-dU/dx`` is differentiated with
    respect to the atom position tensor. If the backbone frame positions are a
    *separate* tensor, the part of the geometry dependence that flows through the
    frames bypasses that gradient and the conservative force is incomplete. After
    this call there is one tensor of record, so the gradient is complete.

    Residues whose N/CA/C cannot all be found keep their existing positions and
    are marked invalid in ``frame_valid``.
    """
    import dataclasses

    indices, complete = frame_atom_indices(batch)
    if not bool(complete.any()):
        return batch

    positions = batch.atoms.positions
    safe = indices.clamp(min=0)
    gathered = [positions[safe[:, slot]] for slot in range(3)]
    keep = complete.unsqueeze(-1)
    backbone = dataclasses.replace(
        batch.backbone,
        n_positions=torch.where(keep, gathered[0], batch.backbone.n_positions),
        ca_positions=torch.where(keep, gathered[1], batch.backbone.ca_positions),
        c_positions=torch.where(keep, gathered[2], batch.backbone.c_positions),
        frame_valid=batch.backbone.frame_valid & complete,
    )
    return dataclasses.replace(batch, backbone=backbone)


def random_rotation_matrix(
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Uniform random **proper** rotation (``det = +1``) via QR.

    Reflections are excluded on purpose: an improper transform is not a symmetry
    of this model and must never appear in an equivariance test as if it were.
    """
    a = torch.empty((3, 3), dtype=torch.float64)
    a.normal_(generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype=dtype, device=device)


def apply_rigid_transform(
    batch: HierarchicalProteinBatch, rotation: Tensor, translation: Tensor
) -> HierarchicalProteinBatch:
    """Apply ``x -> R x + t`` to every position, and ``f -> R f`` to forces.

    Returns a new batch; the input is untouched. Scalars, indices and masks are
    shared, not copied, because they are invariant by construction.
    """
    import dataclasses

    def rot_points(x: Tensor) -> Tensor:
        return x @ rotation.transpose(-1, -2).to(x.dtype) + translation.to(x.dtype)

    def rot_vectors(v: Tensor) -> Tensor:
        return v @ rotation.transpose(-1, -2).to(v.dtype)

    atoms = dataclasses.replace(
        batch.atoms,
        positions=rot_points(batch.atoms.positions),
        forces=None if batch.atoms.forces is None else rot_vectors(batch.atoms.forces),
    )
    backbone = dataclasses.replace(
        batch.backbone,
        n_positions=rot_points(batch.backbone.n_positions),
        ca_positions=rot_points(batch.backbone.ca_positions),
        c_positions=rot_points(batch.backbone.c_positions),
    )
    return dataclasses.replace(batch, atoms=atoms, backbone=backbone)
