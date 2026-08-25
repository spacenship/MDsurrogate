"""Radial basis and cutoff envelope.

The radial part of a message is the only place a distance enters, and it is an
invariant scalar. It is expanded in Bessel functions and multiplied by a smooth
envelope that reaches exactly zero at the cutoff.

The envelope matters more than it looks: without it, an atom pair crossing the
cutoff makes the energy jump discontinuously, and any force read off that energy
is wrong near the boundary. With it, the message and its derivative both go to
zero at ``r_cut``, so rebuilding the neighbour list does not change the output.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["BesselBasis", "polynomial_cutoff", "RadialMLP"]


def polynomial_cutoff(r: Tensor, r_cut: float, p: int = 6) -> Tensor:
    """Smooth envelope that is 1 at ``r=0`` and 0 with zero derivative at ``r_cut``.

    The degree-``p`` polynomial of Klicpera et al. (2020); ``p=6`` makes the
    value and the first two derivatives vanish at the cutoff.
    """
    x = (r / r_cut).clamp(max=1.0)
    out = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
        + p * (p + 2.0) * x ** (p + 1)
        - (p * (p + 1.0) / 2.0) * x ** (p + 2)
    )
    return out * (r < r_cut)


class BesselBasis(nn.Module):
    """Radial Bessel basis ``sqrt(2/rc) sin(n pi r / rc) / r``, times the envelope.

    Args:
        r_cut: cutoff in the batch's length unit (Angstrom).
        num_basis: number of basis functions.
        trainable: let the frequencies adapt.

    Shape:
        ``[E] -> [E, num_basis]``.

    ``r`` is clamped away from zero before the division: coincident atoms would
    otherwise produce inf in the forward pass and NaN in the backward pass, and a
    zero-length edge does occur (a duplicated position in a malformed frame).
    """

    def __init__(self, r_cut: float, num_basis: int = 8, trainable: bool = False):
        super().__init__()
        self.r_cut = float(r_cut)
        self.num_basis = int(num_basis)
        freqs = torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)
        self.prefactor = math.sqrt(2.0 / self.r_cut)

    def forward(self, r: Tensor) -> Tensor:
        r_safe = r.clamp(min=1e-4).unsqueeze(-1)
        basis = self.prefactor * torch.sin(self.frequencies * r_safe / self.r_cut) / r_safe
        return basis * polynomial_cutoff(r, self.r_cut).unsqueeze(-1)


class RadialMLP(nn.Module):
    """Maps invariant edge scalars to tensor-product path weights.

    This is the second sanctioned route for scalar information into the
    equivariant channels: the weights of every tensor-product path are a learned
    function of (radial basis, relation sub-type, endpoint scalars). The
    equivariance of the block does not depend on what this MLP computes, only on
    the fact that its output is an invariant scalar per path.

    Shape:
        ``[E, in_dim] -> [E, num_weights]``.
    """

    def __init__(self, in_dim: int, num_weights: int, hidden: int = 64,
                 num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [num_weights]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        # Small init: at the start the messages should be a mild perturbation of
        # the residual stream rather than dominating it.
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
