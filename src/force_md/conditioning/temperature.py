"""Temperature conditioning.

mdCATH samples five discrete temperatures (320/348/379/413/450 K), but the
encoding here is continuous. Two reasons: the model should interpolate rather
than memorise five one-hot buckets, and Phase 2 adds physical-lag conditioning
on the same footing, where a continuous encoding is unavoidable.

Temperature enters the network only as an invariant (``l=0``) scalar. It
modulates equivariant channels through gates and tensor-product weights, never
by being concatenated onto a vector or ``l=2`` feature.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data.units import BOLTZMANN_KCAL_PER_MOL_PER_K

__all__ = ["temperature_features", "TEMPERATURE_FEATURE_DIM"]

#: Range spanning mdCATH's 320-450 K with margin, used to normalise.
_T_MIN = 300.0
_T_MAX = 470.0

TEMPERATURE_FEATURE_DIM = 10  # num_basis (8) + normalised T + beta


def temperature_features(
    temperature_kelvin: Tensor, num_basis: int = 8
) -> Tensor:
    """Featurise temperature as invariant scalars.

    Args:
        temperature_kelvin: ``[B]`` temperatures in kelvin.
        num_basis: Gaussian radial basis functions spread over ``[300, 470] K``.

    Returns:
        ``[B, num_basis + 2]``: the RBF expansion, the linearly normalised
        temperature, and ``beta = 1/kT`` rescaled to O(1). ``beta`` is included
        because Boltzmann weights depend on ``1/T``, not ``T``, so a purely
        linear encoding would force the network to learn a reciprocal.
    """
    t = temperature_kelvin.reshape(-1).to(torch.get_default_dtype())
    centres = torch.linspace(_T_MIN, _T_MAX, num_basis, dtype=t.dtype, device=t.device)
    width = (_T_MAX - _T_MIN) / max(num_basis - 1, 1)
    rbf = torch.exp(-(((t[:, None] - centres[None, :]) / width) ** 2))

    normalised = ((t - _T_MIN) / (_T_MAX - _T_MIN)).unsqueeze(-1)
    kt = BOLTZMANN_KCAL_PER_MOL_PER_K * t
    beta = (1.0 / kt).unsqueeze(-1) * BOLTZMANN_KCAL_PER_MOL_PER_K * _T_MIN  # ~1 at T_MIN
    return torch.cat([rbf, normalised, beta], dim=-1)
