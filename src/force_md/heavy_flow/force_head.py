"""Direct atom-force distribution head for Stage 3.

The head is a narrow boundary over :class:`PhysicsState`: the mean is a
channel-mixing map of polar ``1o`` atom features and the isotropic log-variance
is a scalar ``0e`` map.  It accepts no coordinates, target, or teacher input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .physics_types import PhysicsState

__all__ = ["ForceDistribution", "ForceHeadOutput", "AtomForceDistribution", "ForceHead", "AtomForceHead"]


@dataclass(frozen=True)
class ForceDistribution:
    """Dense per-atom output: mean ``[B,N,3]`` and logvar ``[B,N,1]``."""

    force_mean: Tensor
    force_logvar: Tensor
    state: Optional[PhysicsState] = None

    @property
    def mean(self) -> Tensor:
        return self.force_mean

    @property
    def logvar(self) -> Tensor:
        return self.force_logvar

    def validate(self) -> None:
        if not isinstance(self.force_mean, Tensor) or not isinstance(self.force_logvar, Tensor):
            raise ValueError("force distribution fields must be tensors")
        if self.force_mean.ndim != 3 or self.force_mean.shape[-1] != 3:
            raise ValueError("force_mean must have shape [B,N,3]")
        if self.force_logvar.ndim != 3 or self.force_logvar.shape[:2] != self.force_mean.shape[:2]:
            raise ValueError("force_logvar must have shape [B,N,1] or [B,N,3]")
        if self.force_logvar.shape[-1] not in (1, 3):
            raise ValueError("force_logvar must have one isotropic or three component channels")


ForceHeadOutput = ForceDistribution
AtomForceDistribution = ForceDistribution


class ForceHead(nn.Module):
    """Read a PhysicsState into an atom-wise mean and isotropic variance."""

    def __init__(
        self,
        scalar_dim: int,
        *,
        vector_channels: int = 1,
        predict_force_mean: bool = True,
        predict_isotropic_force_variance: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ):
        super().__init__()
        if scalar_dim < 1 or vector_channels < 1:
            raise ValueError("scalar_dim and vector_channels must be positive")
        if not predict_force_mean or not predict_isotropic_force_variance:
            raise ValueError("Stage 3 requires force mean and isotropic variance outputs")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.predict_force_mean = bool(predict_force_mean)
        self.predict_isotropic_force_variance = bool(predict_isotropic_force_variance)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        # Polar vector channels are mixed into one polar output.  There is no
        # Cartesian bias or scalar-to-vector shortcut.
        self.vector_weight = nn.Parameter(torch.empty(self.vector_channels))
        nn.init.normal_(self.vector_weight, mean=0.0, std=self.vector_channels ** -0.5)
        self.logvar = nn.Linear(self.scalar_dim, 1)

    def _mean(self, vector: Tensor, *, dense: bool) -> Tensor:
        if dense:
            if vector.ndim == 3 and vector.shape[-1] == 3:
                return vector * self.vector_weight[0]
            if vector.ndim == 4 and vector.shape[2:] == (self.vector_channels, 3):
                return torch.einsum("bnvc,v->bnc", vector, self.vector_weight)
            raise ValueError("dense atom_vector must be [B,N,3] or [B,N,V,3]")
        if vector.ndim == 2 and vector.shape[-1] == 3:
            if self.vector_channels != 1:
                raise ValueError("a packed [M,3] vector requires vector_channels=1")
            return vector * self.vector_weight[0]
        if vector.ndim == 3 and vector.shape[1:] == (self.vector_channels, 3):
            return torch.einsum("mvc,v->mc", vector, self.vector_weight)
        raise ValueError("packed atom_vector must be [M,3] or [M,V,3]")

    def _dense_from_packed(self, value: Tensor, state: PhysicsState) -> Tensor:
        if state.atom_batch is None or state.atom_local is None:
            raise ValueError("packed PhysicsState requires atom_batch/atom_local for dense outputs")
        if state.batch_size is None or state.max_atoms is None:
            raise ValueError("packed PhysicsState requires batch_size/max_atoms for dense outputs")
        dense = value.new_zeros((state.batch_size, state.max_atoms) + tuple(value.shape[1:]))
        dense[state.atom_batch, state.atom_local] = value
        return dense

    def forward(self, state: PhysicsState) -> ForceDistribution:
        """Return dense ``[B,N,3]`` mean and ``[B,N,1]`` log-variance."""
        if not isinstance(state, PhysicsState):
            raise TypeError("ForceHead.forward accepts only PhysicsState")
        state.validate()
        dense = state.atom_scalar.ndim == 3
        mean = self._mean(state.atom_vector, dense=dense)
        if state.atom_scalar.shape[-1] != self.scalar_dim:
            raise ValueError("atom_scalar shape disagrees with ForceHead scalar_dim")
        logvar = self.logvar(state.atom_scalar)
        logvar = logvar.float().clamp(self.logvar_min, self.logvar_max).to(state.atom_scalar.dtype)
        if not dense:
            mean = self._dense_from_packed(mean, state)
            logvar = self._dense_from_packed(logvar, state)
        result = ForceDistribution(mean, logvar, state)
        result.validate()
        return result


AtomForceHead = ForceHead
