"""Physical-lag conditioning for the transition probe.

The probe answers "where is this residue in ``D`` picoseconds", and ``D`` is an
input, not a property of the weights: one model is trained on 1 ns and 4 ns
transitions together and has to tell them apart. This module is how ``D`` reaches
it.

It follows :mod:`force_md.conditioning.temperature` deliberately -- continuous
basis functions, not a two-way one-hot. The lags in use are 1000 and 4000 ps and
a one-hot would fit them perfectly, which is the problem: adding 8 ns later would
change the tensor contract, retrain every arm and invalidate the comparison. A
continuous encoding of a quantity that spans a decade is also better placed in
**log** time, because 1 -> 4 ns is the same kind of step as 4 -> 16 ns, and
diffusive displacement grows like a power of the elapsed time rather than
linearly in it.

Like temperature, the lag is an invariant (``l = 0``) scalar and enters only
through scalar channels.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["lag_features", "LAG_FEATURE_DIM"]

#: Range the basis spans, in ps: 0.5 ns to 20 ns. Wider than the 1-4 ns actually
#: used so that a later 8 or 16 ns experiment is a config change, not a new
#: feature width and a fresh set of weights.
_LAG_MIN_PS = 500.0
_LAG_MAX_PS = 20000.0

LAG_FEATURE_DIM = 10  # num_basis (8) + normalised log lag + sqrt(lag) scale


def lag_features(lag_ps: Tensor, num_basis: int = 8) -> Tensor:
    """Featurise the physical lag as invariant scalars.

    Args:
        lag_ps: ``[B]`` lags in picoseconds.
        num_basis: Gaussian basis functions, evenly spaced in ``log10`` over
            ``[500, 20000] ps``.

    Returns:
        ``[B, num_basis + 2]``: the log-spaced RBF expansion, the normalised
        ``log10`` lag, and ``sqrt(lag / 1 ns)``. The square root is included for
        the same reason ``temperature_features`` includes ``beta``: the quantity
        the network needs is not the input itself. A freely diffusing coordinate
        moves like ``sqrt(t)``, so handing over ``sqrt`` saves the network from
        learning it, while the RBFs stay free to represent the departures from
        diffusion that are the interesting part.
    """
    t = lag_ps.reshape(-1).to(torch.get_default_dtype())
    if bool((t <= 0).any()):
        raise ValueError(
            f"lag_ps must be positive; got a minimum of {float(t.min())}. A "
            "non-positive lag is not a transition."
        )
    log_t = torch.log10(t)
    centres = torch.linspace(
        math.log10(_LAG_MIN_PS), math.log10(_LAG_MAX_PS), num_basis,
        dtype=t.dtype, device=t.device,
    )
    width = (centres[-1] - centres[0]) / max(num_basis - 1, 1)
    rbf = torch.exp(-(((log_t[:, None] - centres[None, :]) / width) ** 2))

    span = math.log10(_LAG_MAX_PS) - math.log10(_LAG_MIN_PS)
    normalised = ((log_t - math.log10(_LAG_MIN_PS)) / span).unsqueeze(-1)
    diffusive = torch.sqrt(t / 1000.0).unsqueeze(-1)
    return torch.cat([rbf, normalised, diffusive], dim=-1)
