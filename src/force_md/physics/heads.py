"""Physics output heads.

Two structural rules run through all of them:

**The conservative part and the residual part are separate tensors.** The full
effective force on a coarse-grained protein atom is *not* the gradient of a
potential: momentum and solvent have been integrated out, so the system is open
and generally non-Markovian. The energy branch therefore produces one clearly
labelled ``conservative`` component, the direct vector head produces a
``residual``, and their sum is the predicted mean. Nothing in the code claims
the two are the same object.

**Means are equivariant, uncertainties are invariant.** A predicted force is an
``l=1`` output in the global frame and rotates with the structure. A predicted
variance is built from ``l=0`` scalars and is expressed in the **residue-local
frame**, where it is rotation-invariant -- a diagonal covariance quoted in the
global frame would silently change meaning when the protein tumbles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..data.contracts import HierarchicalProteinBatch
from ..geometry.frames import ResidueFrames, to_local_vectors
from ..nn.blocks import extract_scalars, scalar_dim

__all__ = [
    "AtomicForceOutput",
    "ResiduePhysicsOutput",
    "AtomicEffectiveForceHead",
    "ResiduePhysicsHead",
]

#: Predicted log-variances are clamped to this range. exp(-8) ~ 3e-4 and
#: exp(8) ~ 3e3 in normalised units, which spans any plausible confidence while
#: keeping 1/sigma^2 finite in the NLL.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0


def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
    )


@dataclass
class AtomicForceOutput:
    """Per-atom force prediction.

    Args:
        mean: ``[N_atom, 3]`` global frame. ``residual + conservative``.
        residual: ``[N_atom, 3]`` the non-conservative part, always present.
        conservative: ``[N_atom, 3]`` ``-grad U``, or None when the energy
            branch is disabled or its gradient was not requested.
        logvar: ``[N_atom, 3]`` diagonal log-variance in the **residue-local**
            frame, of the normalised error.
    """

    mean: Tensor
    residual: Tensor
    conservative: Optional[Tensor]
    logvar: Tensor


@dataclass
class ResiduePhysicsOutput:
    """Per-residue force and torque prediction.

    Args:
        explained_force: ``[N_res, 3]`` residue-level prediction of the force
            carried by the *represented* atoms. Tied to the aggregation of the
            predicted atom forces by the consistency loss.
        hidden_force: ``[N_res, 3]`` or None. The omitted-atom (hydrogen)
            residual. Present only when the target scope makes it identifiable.
        force_mean: ``explained_force`` plus ``hidden_force`` when present.
        torque_mean: ``[N_res, 3]`` about ``torque_origin``.
        torque_origin: ``[N_res, 3]``, the CA positions.
        force_logvar / torque_logvar: ``[N_res, 3]`` local-frame diagonal.
    """

    explained_force: Tensor
    hidden_force: Optional[Tensor]
    force_mean: Tensor
    torque_mean: Tensor
    torque_origin: Tensor
    force_logvar: Tensor
    torque_logvar: Tensor


class AtomicEffectiveForceHead(nn.Module):
    """Atom-level effective force mean and heteroscedastic uncertainty.

    Args:
        irreps_node: irreps of the encoder's atom features.
        isotropic_uncertainty: predict one variance per atom instead of three.

    Shape:
        ``[N_atom, D] -> AtomicForceOutput``.
    """

    def __init__(self, irreps_node: o3.Irreps, *, isotropic_uncertainty: bool = False):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.isotropic = isotropic_uncertainty
        # An l=1 output is a genuine vector: it rotates with the structure.
        self.to_force = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))
        self.to_logvar = _mlp(scalar_dim(self.irreps_node), 1 if isotropic_uncertainty else 3)

    def forward(
        self,
        atom_features: Tensor,
        *,
        conservative: Optional[Tensor] = None,
    ) -> AtomicForceOutput:
        residual = self.to_force(atom_features)
        logvar = self.to_logvar(extract_scalars(atom_features, self.irreps_node))
        if self.isotropic:
            logvar = logvar.expand(-1, 3)
        logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
        mean = residual if conservative is None else residual + conservative
        return AtomicForceOutput(
            mean=mean, residual=residual, conservative=conservative, logvar=logvar
        )


class ResiduePhysicsHead(nn.Module):
    """Residue-level force, torque and their uncertainties.

    Args:
        irreps_node: irreps of the encoder's residue features.
        predict_hidden_force: emit the omitted-atom residual. This must be set
            from the *target scope*, not chosen freely: with only a heavy-atom
            target the residual is not identifiable and would absorb arbitrary
            mass. mdCATH stores hydrogens, so both scopes exist and the residual
            is a real target -- measured at 0.58-0.71 of the heavy-only residue
            force across 12 audited domains.

    Shape:
        ``[N_res, D] -> ResiduePhysicsOutput``.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        *,
        predict_hidden_force: bool = False,
        isotropic_uncertainty: bool = False,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.predict_hidden_force = predict_hidden_force
        self.isotropic = isotropic_uncertainty

        self.to_explained_force = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))
        self.to_torque = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))
        if predict_hidden_force:
            self.to_hidden_force = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))

        n_out = 1 if isotropic_uncertainty else 3
        n_scalar = scalar_dim(self.irreps_node)
        self.to_force_logvar = _mlp(n_scalar, n_out)
        self.to_torque_logvar = _mlp(n_scalar, n_out)

    def forward(
        self, residue_features: Tensor, torque_origin: Tensor
    ) -> ResiduePhysicsOutput:
        explained = self.to_explained_force(residue_features)
        torque = self.to_torque(residue_features)
        hidden = (
            self.to_hidden_force(residue_features) if self.predict_hidden_force else None
        )
        force_mean = explained if hidden is None else explained + hidden

        scalars = extract_scalars(residue_features, self.irreps_node)
        f_logvar = self.to_force_logvar(scalars)
        t_logvar = self.to_torque_logvar(scalars)
        if self.isotropic:
            f_logvar, t_logvar = f_logvar.expand(-1, 3), t_logvar.expand(-1, 3)

        return ResiduePhysicsOutput(
            explained_force=explained,
            hidden_force=hidden,
            force_mean=force_mean,
            torque_mean=torque,
            torque_origin=torque_origin,
            force_logvar=f_logvar.clamp(LOGVAR_MIN, LOGVAR_MAX),
            torque_logvar=t_logvar.clamp(LOGVAR_MIN, LOGVAR_MAX),
        )


def to_local_frame_error(
    error: Tensor, frames: ResidueFrames, index: Tensor
) -> Tensor:
    """Express a global-frame error vector in the residue-local frame.

    The uncertainty heads predict a diagonal covariance in the local frame, so
    the error must be rotated into that frame before the NLL is evaluated.
    Skipping this makes a diagonal covariance mean something different for every
    orientation of the same residue.
    """
    return to_local_vectors(error, frames, index)
