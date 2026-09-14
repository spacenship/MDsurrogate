"""Stage 3 atom/pair physics contracts.

The Stage 3 boundary is deliberately small and explicit::

    C_atom -> PhysicsPredictor -> PhysicsState -> ForceHead

``PhysicsState`` stores packed valid atoms because that is the layout produced
by the Stage 1/2 graph.  The optional ``atom_batch``/``atom_local`` mapping is
what lets the force head return the public dense ``[B, N_atom, ...]`` shape
without ever reading the original context or a target tensor.

``edge_scalar`` is a learned interaction representation.  It is *not* a
claim that a net atomic force has a unique pairwise-force decomposition; the
pair channels are only a useful latent for downstream physics and diagnostics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor

__all__ = ["PhysicsState"]


def _same_shape_prefix(left: Tensor, right: Tensor, name: str) -> None:
    if left.ndim != right.ndim:
        raise ValueError(f"{name} rank disagrees with atom_scalar")


@dataclass(frozen=True)
class PhysicsState:
    """Atom/pair latent emitted by :class:`PhysicsPredictor`.

    Packed predictor layout:

    * ``atom_scalar``: ``[M, S]`` (``0e`` channels)
    * ``atom_vector``: ``[M, V, 3]`` (``1o`` channels)
    * ``atom_axial``: ``[M, A, 3]`` (``1e`` channels)
    * ``edge_scalar``: ``[E, P]`` (invariant pair latent)

    A single-vector ``[M, 3]`` and dense ``[B, N, 3]`` form are accepted by
    the public force head as a convenience for small hand-written fixtures.
    The predictor always emits the explicit channel form and mapping metadata.

    ``force_mean`` and ``force_logvar`` are optional enrichments added by a
    force head.  They are not inputs to the predictor and are never used to
    create the latent channels.
    """

    atom_scalar: Tensor
    atom_vector: Tensor
    atom_axial: Tensor
    edge_scalar: Tensor
    force_mean: Optional[Tensor] = None
    force_logvar: Optional[Tensor] = None
    atom_batch: Optional[Tensor] = None
    atom_local: Optional[Tensor] = None
    atom_mask: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_kind: Optional[Tensor] = None
    bond_type: Optional[Tensor] = None
    edge_mask: Optional[Tensor] = None
    batch_size: Optional[int] = None
    max_atoms: Optional[int] = None

    @property
    def num_atoms(self) -> int:
        if self.atom_scalar.ndim == 2:
            return int(self.atom_scalar.shape[0])
        return int(self.atom_scalar.shape[0] * self.atom_scalar.shape[1])

    @property
    def vector_channels(self) -> int:
        if self.atom_vector.ndim == 2:
            return 1
        if self.atom_vector.ndim == 3 and self.atom_scalar.ndim == 2:
            return int(self.atom_vector.shape[1])
        return 1

    @property
    def axial_channels(self) -> int:
        if self.atom_axial.ndim == 2:
            return 1
        if self.atom_axial.ndim == 3 and self.atom_scalar.ndim == 2:
            return int(self.atom_axial.shape[1])
        return 1

    def validate(self) -> None:
        tensors = (self.atom_scalar, self.atom_vector, self.atom_axial, self.edge_scalar)
        if not all(isinstance(value, Tensor) for value in tensors):
            raise ValueError("PhysicsState latent fields must be torch.Tensor values")
        if self.atom_scalar.ndim not in (2, 3):
            raise ValueError("atom_scalar must be [M,S] or [B,N,S]")
        if self.edge_scalar.ndim != 2:
            raise ValueError("edge_scalar must be [E,P]")
        if self.atom_vector.shape[-1] != 3 or self.atom_axial.shape[-1] != 3:
            raise ValueError("atom_vector and atom_axial must end in Cartesian width 3")
        if self.atom_scalar.ndim == 2:
            m = self.atom_scalar.shape[0]
            if self.atom_vector.ndim not in (2, 3) or self.atom_vector.shape[0] != m:
                raise ValueError("packed atom_vector does not align with atom_scalar")
            if self.atom_axial.ndim not in (2, 3) or self.atom_axial.shape[0] != m:
                raise ValueError("packed atom_axial does not align with atom_scalar")
        else:
            if self.atom_vector.ndim != 3 or self.atom_vector.shape[:2] != self.atom_scalar.shape[:2]:
                raise ValueError("dense atom_vector must be [B,N,3]")
            if self.atom_axial.ndim != 3 or self.atom_axial.shape[:2] != self.atom_scalar.shape[:2]:
                raise ValueError("dense atom_axial must be [B,N,3]")
        if self.atom_batch is not None or self.atom_local is not None:
            if self.atom_batch is None or self.atom_local is None:
                raise ValueError("atom_batch and atom_local must be supplied together")
            if self.atom_scalar.ndim != 2:
                raise ValueError("packed mapping metadata is only valid for packed latents")
            if self.atom_batch.shape != self.atom_local.shape or self.atom_batch.ndim != 1:
                raise ValueError("atom_batch and atom_local must be [M]")
            if self.atom_batch.shape[0] != self.atom_scalar.shape[0]:
                raise ValueError("atom mapping length disagrees with atom latents")
        if self.atom_mask is not None:
            if self.atom_mask.ndim != 2 or self.batch_size is None or self.max_atoms is None:
                raise ValueError("atom_mask requires batch_size and max_atoms")
            if self.atom_mask.shape != (self.batch_size, self.max_atoms):
                raise ValueError("atom_mask shape disagrees with batch_size/max_atoms")
        if self.force_mean is not None and self.force_logvar is None:
            raise ValueError("force_mean and force_logvar must be supplied together")
        if self.force_logvar is not None and self.force_mean is None:
            raise ValueError("force_logvar and force_mean must be supplied together")
        if self.force_mean is not None:
            if self.force_mean.shape[-1] != 3 or self.force_logvar.shape[-1] not in (1, 3):
                raise ValueError("force outputs must end in 3 and 1/3 channels")
            if self.force_mean.shape[:-1] != self.force_logvar.shape[:-1]:
                raise ValueError("force_mean and force_logvar leading shapes disagree")
        if self.edge_index is not None:
            if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
                raise ValueError("edge_index must be [2,E]")
            if self.edge_index.shape[1] != self.edge_scalar.shape[0]:
                raise ValueError("edge_index and edge_scalar disagree on E")

        for name in ("edge_kind", "bond_type", "edge_mask"):
            value = getattr(self, name)
            if value is not None and value.shape != (self.edge_scalar.shape[0],):
                raise ValueError(f"{name} must align with edge_scalar")
        if self.edge_mask is not None and self.edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be boolean")
        if self.edge_index is not None and self.edge_index.numel():
            if self.edge_index.min() < 0 or self.edge_index.max() >= self.num_atoms:
                raise ValueError("physics edge endpoint outside atom mapping")
            if self.atom_batch is not None and not torch.equal(self.atom_batch[self.edge_index[0]], self.atom_batch[self.edge_index[1]]):
                raise ValueError("physics edges cross batch/trajectory boundaries")

    def with_force(self, force_mean: Tensor, force_logvar: Tensor) -> "PhysicsState":
        """Return this latent enriched with a force distribution."""
        result = replace(self, force_mean=force_mean, force_logvar=force_logvar)
        result.validate()
        return result

    def to(self, device: torch.device | str) -> "PhysicsState":
        values = {
            field.name: (getattr(self, field.name).to(device)
                         if isinstance(getattr(self, field.name), Tensor)
                         else getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        result = PhysicsState(**values)
        result.validate()
        return result

