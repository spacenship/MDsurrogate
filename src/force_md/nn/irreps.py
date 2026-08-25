"""Irreps bookkeeping and gated nonlinearities.

Symmetry contract for the whole encoder:

* Features live in the **global frame** at every level. Sums, scatters and
  tensor products of global-frame irreps are equivariant without any frame
  bookkeeping, so there is exactly one place where frames appear -- building the
  invariant input scalars, and expressing output uncertainty.
* Irreps use ``o3`` (i.e. O(3)) labels because that is e3nn's vocabulary, but the
  requirement is **SE(3)**: proper rotations and translations. Reflection is not
  a symmetry of a chiral molecule.
* Chirality enters through the residue frame. Spherical harmonics of relative
  positions alone would leave the network accidentally E(3)-equivariant, so the
  atom scalars include ``y_ia = R_i^T (x_ia - r_i)`` as three invariant numbers.
  ``R_i``'s third axis is a cross product, so ``y_ia`` distinguishes a structure
  from its mirror image. This is asserted in
  ``test_encoder_is_chirality_sensitive``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from e3nn import nn as enn
from e3nn import o3
from torch import Tensor, nn

__all__ = ["IrrepsConfig", "GatedLinear", "scatter_sum", "scatter_mean"]


@dataclass(frozen=True)
class IrrepsConfig:
    """Width of every equivariant feature in the model.

    Phase 1 small config keeps ``lmax=2`` and small multiplicities. Phase 2 grows
    the channel counts and the number of blocks; it does not change the class or
    the maximum degree, so a Phase 1 checkpoint stays loadable into the same
    module family.
    """

    lmax: int = 2
    scalar_channels: int = 64
    vector_channels: int = 16
    tensor_channels: int = 8

    def node_irreps(self) -> o3.Irreps:
        """``64x0e + 16x1o + 8x2e`` at the defaults."""
        parts = [f"{self.scalar_channels}x0e"]
        if self.lmax >= 1:
            parts.append(f"{self.vector_channels}x1o")
        if self.lmax >= 2:
            parts.append(f"{self.tensor_channels}x2e")
        return o3.Irreps("+".join(parts))

    def sh_irreps(self) -> o3.Irreps:
        """Spherical-harmonic irreps of an edge direction: ``1x0e+1x1o+1x2e``."""
        return o3.Irreps.spherical_harmonics(self.lmax)


class GatedLinear(nn.Module):
    """Equivariant linear map followed by a gated nonlinearity.

    A nonlinearity cannot be applied elementwise to an ``l>0`` feature without
    breaking equivariance. The gate instead multiplies each non-scalar channel by
    a scalar function of the scalar channels -- a scalar times an ``l`` feature is
    still an ``l`` feature. This is one of the two sanctioned routes for scalar
    information (PLM, temperature, radial) to reach the equivariant channels.

    Args:
        irreps_in: input irreps.
        irreps_out: desired output irreps.
        act_scalars / act_gates: activations for the scalar and gate channels.

    Shape:
        ``[N, irreps_in.dim] -> [N, irreps_out.dim]``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        act_scalars: Callable[[Tensor], Tensor] = torch.nn.functional.silu,
        act_gates: Callable[[Tensor], Tensor] = torch.sigmoid,
    ):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
        num_gates = sum(mul for mul, _ in gated)
        gates = o3.Irreps(f"{num_gates}x0e") if num_gates else o3.Irreps("")

        self.gate = enn.Gate(
            scalars, [act_scalars] * len(scalars),
            gates, [act_gates] * len(gates),
            gated,
        )
        self.linear = o3.Linear(irreps_in, self.gate.irreps_in)
        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(self.gate.irreps_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.gate(self.linear(x))


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``.

    A pure-PyTorch ``index_add`` rather than ``torch_scatter``: the core
    hierarchy must not depend on an optional compiled package. Empty buckets
    produce exact zeros, which is why an empty relation is a safe no-op rather
    than a NaN.
    """
    out = src.new_zeros((dim_size,) + src.shape[1:])
    if src.shape[0] == 0:
        return out
    return out.index_add(0, index, src)


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean over each bucket; empty buckets stay exactly zero."""
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(src[:, :1]), index, dim_size)
    return total / count.clamp(min=1.0)
