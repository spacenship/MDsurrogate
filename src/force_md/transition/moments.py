"""Force moments and residue shape, in the residue's own frame.

**Why the net force is not enough.** The residue force ``F_i = sum_a f_ia`` is the
correct observable for translation, but it is a *sum*, and Newton's third law
cancels most of what is inside a residue: Phase 1 measured mean atomic ``|f| =
38.4`` against a mean residue ``|F| = 51.8``, where uncorrelated atoms would give
``sqrt(8) x 38 ~ 108``. The cancellation is real, and what it cancels -- how the
residue is being squeezed, sheared and twisted -- is exactly the part that a
conformational transition depends on.

The first moment of the force about the residue origin keeps that information::

    M_i = sum_a y_ia (x) f_local_ia            [3, 3] per residue

and splits into three physically distinct pieces:

===================  ============================================================
trace(M)             isotropic compression / expansion -- one scalar
antisymmetric part   the torque, ``vee((M - M^T)/2) = -tau_i / 2``
symmetric traceless  directional stress and shear -- five components
===================  ============================================================

The antisymmetric part is the torque and is reported as such rather than as a
second, differently-scaled copy of it: two names for one quantity in a feature
vector is a way of quietly doubling its weight.

**Everything here is an SE(3) invariant**, because ``y_ia = R_i^T (x_ia - r_i)``
and ``f_local_ia = R_i^T f_ia`` are. Under a global motion ``x -> Qx + t``,
``f -> Qf``, ``R -> QR``, both are unchanged, so every moment built from them is
too. That is what makes it safe to feed them to an ordinary MLP.

**Shape** is measured from the same local coordinates: the gyration tensor
``G_i = sum_a y_ia (x) y_ia / n_i`` gives size (its trace) and anisotropy (its
traceless part). No van der Waals radius table is introduced -- the repository has
no sourced one, and inventing constants to make a feature look physical is worse
than measuring the coordinates that are actually there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..nn.irreps import scatter_sum

__all__ = ["ForceMoments", "ResidueShape", "force_moments", "residue_shape",
           "SYMMETRIC_TRACELESS_DIM", "MOMENT_FEATURE_DIM"]

#: Independent components of a symmetric traceless 3x3 tensor.
SYMMETRIC_TRACELESS_DIM = 5

#: Width of :meth:`ForceMoments.as_features`.
MOMENT_FEATURE_DIM = 3 + 3 + 1 + 1 + 1 + SYMMETRIC_TRACELESS_DIM


def _symmetric_traceless_components(tensor: Tensor) -> Tensor:
    """The five independent components of a symmetric traceless ``[N, 3, 3]``.

    Ordered ``(xy, xz, yz, xx - yy, (2 zz - xx - yy)/sqrt(3))`` -- the same
    combinations the ``l = 2`` spherical components carry, normalised so that no
    single entry dominates the others by construction.
    """
    xx, yy, zz = tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2]
    return torch.stack(
        [
            tensor[:, 0, 1],
            tensor[:, 0, 2],
            tensor[:, 1, 2],
            xx - yy,
            (2.0 * zz - xx - yy) / 3.0**0.5,
        ],
        dim=-1,
    )


@dataclass
class ForceMoments:
    """Moments of the atomic force field inside each residue, local frame.

    Args:
        net_force: ``[N_res, 3]`` ``sum_a f_local_ia``. Zero for a residue whose
            internal forces cancel exactly.
        torque: ``[N_res, 3]`` ``sum_a y_ia x f_local_ia``.
        isotropic: ``[N_res, 1]`` ``trace(M) / 3`` -- compression when negative,
            expansion when positive.
        symmetric_traceless: ``[N_res, 5]`` directional stress and shear.
        net_force_norm / torque_norm: ``[N_res, 1]`` magnitudes, kept explicitly
            because an MLP reaching them from components has to learn a square
            root that the feature can simply carry.
        num_atoms: ``[N_res]`` contributing atoms.
    """

    net_force: Tensor
    torque: Tensor
    isotropic: Tensor
    symmetric_traceless: Tensor
    net_force_norm: Tensor
    torque_norm: Tensor
    num_atoms: Tensor

    def as_features(self) -> Tensor:
        """``[N_res, MOMENT_FEATURE_DIM]`` invariant scalars, ready for an MLP."""
        return torch.cat(
            [
                self.net_force,
                self.torque,
                self.net_force_norm,
                self.torque_norm,
                self.isotropic,
                self.symmetric_traceless,
            ],
            dim=-1,
        )


@dataclass
class ResidueShape:
    """Size and anisotropy of a residue's atom cloud, local frame.

    Args:
        radius_of_gyration: ``[N_res, 1]`` ``sqrt(trace(G))``.
        anisotropy: ``[N_res, 5]`` traceless part of the gyration tensor.
        mean_radius / max_radius: ``[N_res, 1]`` first moment and extent of
            ``|y_ia|``.
        num_atoms: ``[N_res]``.
    """

    radius_of_gyration: Tensor
    anisotropy: Tensor
    mean_radius: Tensor
    max_radius: Tensor
    num_atoms: Tensor

    def as_features(self) -> Tensor:
        return torch.cat(
            [self.radius_of_gyration, self.mean_radius, self.max_radius, self.anisotropy],
            dim=-1,
        )

    @staticmethod
    def feature_dim() -> int:
        return 3 + SYMMETRIC_TRACELESS_DIM


def force_moments(
    local_coordinates: Tensor,
    local_forces: Tensor,
    atom_to_residue: Tensor,
    num_residues: int,
    *,
    weights: Optional[Tensor] = None,
) -> ForceMoments:
    """Zeroth and first moments of the local force field, per residue.

    Args:
        local_coordinates: ``[N_atom, 3]`` ``y_ia``.
        local_forces: ``[N_atom, 3]`` ``f_local_ia``.
        atom_to_residue: ``[N_atom]`` parent residue.
        num_residues: number of residue rows.
        weights: optional ``[N_atom]`` non-negative weights, used to down-weight
            atoms whose predicted force is uncertain. Weighting the *contribution*
            rather than masking keeps the map continuous in the uncertainty.

    Aggregation is a segment sum over the flattened-ragged layout -- no residue is
    ever padded to a fixed atom count.
    """
    if weights is None:
        weighted_forces = local_forces
    else:
        weighted_forces = local_forces * weights.unsqueeze(-1)

    net_force = scatter_sum(weighted_forces, atom_to_residue, num_residues)
    torque = scatter_sum(
        torch.linalg.cross(local_coordinates, weighted_forces, dim=-1),
        atom_to_residue,
        num_residues,
    )
    outer = (local_coordinates.unsqueeze(-1) * weighted_forces.unsqueeze(-2)).reshape(-1, 9)
    moment = scatter_sum(outer, atom_to_residue, num_residues).reshape(num_residues, 3, 3)

    trace = moment.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0
    eye = torch.eye(3, dtype=moment.dtype, device=moment.device)
    symmetric = 0.5 * (moment + moment.transpose(-1, -2)) - trace.unsqueeze(-1) * eye

    counts = scatter_sum(
        torch.ones_like(local_coordinates[:, :1]), atom_to_residue, num_residues
    ).squeeze(-1)
    return ForceMoments(
        net_force=net_force,
        torque=torque,
        isotropic=trace,
        symmetric_traceless=_symmetric_traceless_components(symmetric),
        net_force_norm=net_force.norm(dim=-1, keepdim=True),
        torque_norm=torque.norm(dim=-1, keepdim=True),
        num_atoms=counts,
    )


def residue_shape(
    local_coordinates: Tensor,
    atom_to_residue: Tensor,
    num_residues: int,
) -> ResidueShape:
    """Gyration tensor and radial extent of each residue's atom cloud."""
    ones = torch.ones_like(local_coordinates[:, :1])
    counts = scatter_sum(ones, atom_to_residue, num_residues).clamp(min=1.0)

    outer = (local_coordinates.unsqueeze(-1) * local_coordinates.unsqueeze(-2)).reshape(-1, 9)
    gyration = (
        scatter_sum(outer, atom_to_residue, num_residues).reshape(num_residues, 3, 3)
        / counts.unsqueeze(-1)
    )
    trace = gyration.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0
    eye = torch.eye(3, dtype=gyration.dtype, device=gyration.device)
    traceless = gyration - trace.unsqueeze(-1) * eye

    radius = local_coordinates.norm(dim=-1, keepdim=True)
    mean_radius = scatter_sum(radius, atom_to_residue, num_residues) / counts
    # scatter_reduce rather than index_reduce: the latter is still flagged beta and
    # warns on every call, which would put a UserWarning in every training log.
    max_radius = torch.zeros_like(mean_radius).scatter_reduce(
        0, atom_to_residue.unsqueeze(-1), radius, "amax", include_self=False
    )
    return ResidueShape(
        radius_of_gyration=(3.0 * trace).clamp(min=0.0).sqrt(),
        anisotropy=_symmetric_traceless_components(traceless),
        mean_radius=mean_radius,
        max_radius=max_radius,
        num_atoms=counts.squeeze(-1),
    )
