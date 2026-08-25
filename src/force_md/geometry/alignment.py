"""Proper-rotation Kabsch alignment over a ragged batch.

**What this is for, and what it must never be used for.** mdCATH stores a protein
diffusing and tumbling in a box. Between two frames 1 ns apart, the largest part
of every atom's displacement is that global rigid motion, which carries no
information about the conformational change being modelled: measured on real
pairs here, the mean unaligned ``|CA(t+4ns) - CA(t)|`` reaches ~18 A while the
same quantity after alignment is ~1-4 A. So the **target** and the **metrics** are
defined on the aligned future structure.

The model's **input** is never aligned. Aligning inputs would leak the future
into the conditioning path (the alignment is computed *from* the future) and
would also hand the network a canonical orientation that inference cannot
reproduce. Phase 1.5 keeps the encoder SE(3)-equivariant and removes global
motion only where it is a definition of the target or a metric, which is the
distinction :mod:`force_md.transition.targets` is built around.

**Proper rotations only.** The reflection-including solution has a smaller
residual for some inputs and is always wrong here: a mirrored protein is a
different molecule, not the same one seen differently. The determinant
correction below is what forbids it, and
``test_kabsch_never_returns_a_reflection`` feeds it a mirrored structure to check
that it is not quietly optimised away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..nn.irreps import scatter_sum

__all__ = ["RigidAlignment", "kabsch_rotation", "align_to_reference"]

#: Fewest correspondences that determine a rotation. With two points the
#: rotation about their common axis is free and the SVD returns an arbitrary one;
#: those graphs are reported through ``valid`` and get the identity.
MIN_CORRESPONDENCES = 3


@dataclass
class RigidAlignment:
    """The proper rigid motion taking a mobile structure onto a reference.

    Applied as ``x -> rotation @ (x - mobile_centroid) + reference_centroid``.

    Args:
        rotation: ``[B, 3, 3]``, ``det = +1``.
        mobile_centroid / reference_centroid: ``[B, 3]``, over the atoms actually
            used for the fit (not over all atoms).
        valid: ``[B]`` bool, False where too few correspondences were available;
            ``rotation`` is the identity there.
        num_used: ``[B]`` int64, correspondences per graph.
    """

    rotation: Tensor
    mobile_centroid: Tensor
    reference_centroid: Tensor
    valid: Tensor
    num_used: Tensor

    def apply(self, points: Tensor, batch_index: Tensor) -> Tensor:
        """Map ``[N, 3]`` mobile-frame points into the reference frame."""
        centred = points - self.mobile_centroid[batch_index]
        rotated = torch.einsum("nij,nj->ni", self.rotation[batch_index], centred)
        return rotated + self.reference_centroid[batch_index]

    def apply_rotation(self, vectors: Tensor, batch_index: Tensor) -> Tensor:
        """Rotate ``[N, 3]`` vectors (no translation) into the reference frame."""
        return torch.einsum("nij,nj->ni", self.rotation[batch_index], vectors)

    def apply_frames(self, frames: Tensor, batch_index: Tensor) -> Tensor:
        """Rotate ``[N, 3, 3]`` residue frames: ``R -> A R``.

        Equivalent to rebuilding the frames from the aligned N/CA/C positions,
        because the frame construction is equivariant -- asserted by
        ``test_aligning_positions_and_aligning_frames_agree`` rather than assumed.
        """
        return self.rotation[batch_index] @ frames

    def to(self, device) -> "RigidAlignment":
        import dataclasses

        return dataclasses.replace(
            self,
            rotation=self.rotation.to(device),
            mobile_centroid=self.mobile_centroid.to(device),
            reference_centroid=self.reference_centroid.to(device),
            valid=self.valid.to(device),
            num_used=self.num_used.to(device),
        )


def kabsch_rotation(
    mobile: Tensor,
    reference: Tensor,
    batch_index: Tensor,
    num_graphs: int,
    *,
    weights: Optional[Tensor] = None,
) -> RigidAlignment:
    """Least-squares **proper** rotation taking ``mobile`` onto ``reference``.

    Args:
        mobile / reference: ``[N, 3]`` corresponding points, row ``i`` of one
            matching row ``i`` of the other. Correspondence is by row and is not
            searched for; a caller that mixes up the residue ordering gets a
            meaningless rotation, which is why
            :func:`force_md.transition.targets.build_transition_target` validates
            the ordering before calling this.
        batch_index: ``[N]`` int64 graph id.
        num_graphs: number of graphs.
        weights: optional ``[N]`` non-negative weights; a 0/1 mask is the usual
            case (exclude residues with a degenerate frame). Rows with weight 0
            contribute to neither centroid nor covariance.

    Returns:
        :class:`RigidAlignment`.

    The SVD runs in float64 regardless of the input dtype. The covariance of a
    250-residue protein at 100 A scale has entries ~1e6, and a float32 SVD of that
    loses enough precision to move the rotation by a noticeable fraction of a
    degree -- which then shows up as a fake conformational change in every
    residue.
    """
    if mobile.shape != reference.shape:
        raise ValueError(
            f"mobile {tuple(mobile.shape)} and reference {tuple(reference.shape)} "
            "must have the same shape; Kabsch pairs points by row"
        )
    n = mobile.shape[0]
    if weights is None:
        weights = mobile.new_ones(n)
    if weights.shape[0] != n:
        raise ValueError(f"weights has {weights.shape[0]} rows for {n} points")

    dtype = torch.float64
    device = mobile.device
    w = weights.to(dtype).unsqueeze(-1)
    m = mobile.to(dtype)
    r = reference.to(dtype)

    count = scatter_sum(w, batch_index, num_graphs)
    denom = count.clamp(min=1.0)
    mobile_centroid = scatter_sum(m * w, batch_index, num_graphs) / denom
    reference_centroid = scatter_sum(r * w, batch_index, num_graphs) / denom

    mc = (m - mobile_centroid[batch_index]) * w
    rc = r - reference_centroid[batch_index]
    outer = (mc.unsqueeze(-1) * rc.unsqueeze(-2)).reshape(n, 9)
    covariance = scatter_sum(outer, batch_index, num_graphs).reshape(num_graphs, 3, 3)

    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    # The determinant correction is the whole point: without it this is the
    # orthogonal Procrustes solution and may return a reflection.
    sign = torch.sign(torch.linalg.det(v @ u.transpose(-1, -2)))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    correction = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )
    rotation = v @ correction @ u.transpose(-1, -2)

    num_used = scatter_sum((weights > 0).to(torch.int64).unsqueeze(-1),
                           batch_index, num_graphs).squeeze(-1)
    valid = num_used >= MIN_CORRESPONDENCES
    eye = torch.eye(3, dtype=dtype, device=device).expand_as(rotation)
    rotation = torch.where(valid[:, None, None], rotation, eye)

    out_dtype = mobile.dtype
    return RigidAlignment(
        rotation=rotation.to(out_dtype),
        mobile_centroid=mobile_centroid.to(out_dtype),
        reference_centroid=reference_centroid.to(out_dtype),
        valid=valid,
        num_used=num_used,
    )


def align_to_reference(
    mobile: Tensor,
    reference: Tensor,
    batch_index: Tensor,
    num_graphs: int,
    *,
    weights: Optional[Tensor] = None,
) -> tuple[Tensor, RigidAlignment]:
    """Fit and apply in one call. Returns ``(aligned_mobile, alignment)``."""
    alignment = kabsch_rotation(
        mobile, reference, batch_index, num_graphs, weights=weights
    )
    return alignment.apply(mobile, batch_index), alignment
