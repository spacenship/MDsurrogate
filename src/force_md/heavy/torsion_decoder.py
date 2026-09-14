"""H1a -- the Delta-chi head, and its circular losses.

The head predicts, per residue and per chi slot, how far that torsion turns
between ``t`` and ``t + lag``. The coarse transition checkpoint is frozen; this
is the only thing that trains.

**Why a mixture and not a regression.** A side-chain torsion is not unimodally
distributed. Over 1-4 ns a chi either stays in its rotamer well or hops to
another one, so the conditional distribution is multimodal with most of its mass
near zero and isolated lumps near +-120 degrees. A single circular mean --
sin/cos regression or one von Mises -- can only answer "somewhere between", and
the value it lands on is a point no real side chain occupies. The default is
therefore a **fixed-component circular mixture**: the component count and the
concentration initialisation are config, fixed before validation is looked at, as
the brief requires.

The three cheaper parameterisations are implemented behind the same interface so
the choice is a config change rather than a rewrite, and so the report can say
which was used.

**Symmetry is handled in the loss, not by editing the target.** For a
pi-periodic torsion (ASP chi2, GLU chi3, PHE chi2, TYR chi2) the two states half
a turn apart are the same structure. The loss folds the residual modulo pi
instead of the target being pre-wrapped, so a perfect prediction that happens to
name the other equivalent state scores zero rather than 180 degrees of error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from ..geometry.torsions import wrap_to_pi
from .chemistry import MAX_CHI

__all__ = [
    "TorsionHeadConfig",
    "SidechainTorsionHead",
    "circular_mixture_nll",
    "von_mises_nll",
    "circular_mae",
    "chi_state_accuracy_3bin",
    "CHI_STATE_EDGES",
]

#: Boundaries of the three chi states, in radians, **fixed here before any
#: validation number was seen** (brief §3). The names are the standard rotamer
#: labels but the metric they feed is called ``chi_state_accuracy_3bin`` and
#: never "rotamer recovery": recovering a rotamer means matching a library
#: entry for the whole side chain, and this repository has no library.
CHI_STATE_EDGES: tuple[float, float, float] = (
    -math.pi / 3.0,      # gauche- / trans boundary at -60 deg
    math.pi / 3.0,       # trans / gauche+ boundary at +60 deg
    math.pi,
)


@dataclass
class TorsionHeadConfig:
    """Everything about the head that is a choice, fixed in config.

    Args:
        parameterisation: ``mixture`` (default), ``state_plus_residual``,
            ``von_mises`` or ``sincos``, in the brief's priority order.
        num_components: mixture components. Fixed before validation.
        hidden: width of the trunk.
        concentration_init: initial log-concentration of every component. A
            small value starts the mixture broad, which matters because a sharp
            random init assigns near-zero likelihood to the truth and the
            gradient vanishes.
        max_chi: chi slots per residue.
        predict_uncertainty: emit a per-torsion concentration as well.
    """

    parameterisation: str = "mixture"
    num_components: int = 3
    hidden: int = 128
    concentration_init: float = 0.5
    max_chi: int = MAX_CHI
    predict_uncertainty: bool = True

    def __post_init__(self) -> None:
        allowed = {"mixture", "state_plus_residual", "von_mises", "sincos"}
        if self.parameterisation not in allowed:
            raise ValueError(
                f"unknown parameterisation {self.parameterisation!r}; "
                f"expected one of {sorted(allowed)}"
            )


class SidechainTorsionHead(nn.Module):
    """Per-residue, per-chi-slot Delta-chi distribution.

    Input is an invariant per-residue feature vector; chi is an internal
    coordinate, so a rotation of the whole structure must leave the prediction
    unchanged. Feeding an equivariant vector channel in here would break that,
    which is why the interface takes scalars only and
    ``test_torsion_head_is_invariant_under_global_rotation`` checks it.
    """

    def __init__(self, in_features: int, config: TorsionHeadConfig | None = None):
        super().__init__()
        self.config = config or TorsionHeadConfig()
        slots = self.config.max_chi
        self.trunk = nn.Sequential(
            nn.Linear(in_features, self.config.hidden),
            nn.SiLU(),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.SiLU(),
        )
        per_slot = self._outputs_per_slot()
        self.out = nn.Linear(self.config.hidden, slots * per_slot)
        # Zero-init the last layer so an untrained head predicts "no change",
        # which is the identity baseline. Starting anywhere else would make the
        # first evaluation of an untrained arm look like a broken model rather
        # than an uninformative one.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._per_slot = per_slot
        if self.config.parameterisation == "state_plus_residual":
            # Zero logits are a three-way tie and ``argmax`` breaks it towards
            # index 0, which is the gauche-minus bin centred at -120 degrees --
            # so an untrained head would predict a 120-degree turn on every
            # torsion. Bias the *trans* bin, whose centre is 0, so this
            # parameterisation starts where the other three do.
            with torch.no_grad():
                bias = self.out.bias.view(slots, per_slot)
                bias[:, 1] = 1.0
        self._state_centres = torch.tensor(
            [-2.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0]
        )

    def _outputs_per_slot(self) -> int:
        mode = self.config.parameterisation
        if mode == "mixture":
            # weight, mean cos, mean sin, log concentration -- per component
            return 4 * self.config.num_components
        if mode == "state_plus_residual":
            return 3 + 2 + (1 if self.config.predict_uncertainty else 0)
        if mode == "von_mises":
            return 2 + 1
        return 2  # sincos

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """``[N_res, D] -> {name: [N_res, max_chi, ...]}``."""
        hidden = self.trunk(features)
        raw = self.out(hidden).view(
            features.shape[0], self.config.max_chi, self._per_slot
        )
        mode = self.config.parameterisation
        if mode == "mixture":
            k = self.config.num_components
            weight_logit = raw[..., :k]
            cos = raw[..., k : 2 * k]
            sin = raw[..., 2 * k : 3 * k]
            log_kappa = raw[..., 3 * k : 4 * k] + self.config.concentration_init
            return {
                "log_weight": torch.log_softmax(weight_logit, dim=-1),
                "mean": torch.atan2(sin, cos),
                "log_concentration": log_kappa,
            }
        if mode == "von_mises":
            return {
                "mean": torch.atan2(raw[..., 1], raw[..., 0]),
                "log_concentration": raw[..., 2] + self.config.concentration_init,
            }
        if mode == "state_plus_residual":
            out = {
                "state_logit": raw[..., :3],
                "residual": torch.atan2(raw[..., 4], raw[..., 3]),
            }
            if self.config.predict_uncertainty:
                out["log_concentration"] = raw[..., 5] + self.config.concentration_init
            return out
        return {"mean": torch.atan2(raw[..., 1], raw[..., 0])}

    def point_estimate(self, output: dict[str, Tensor]) -> Tensor:
        """The single Delta-chi to apply at inference, ``[N_res, max_chi]``.

        For a mixture this is the **mode** -- the mean of the highest-weight
        component -- not the circular mean of the whole mixture. The circular
        mean of a bimodal distribution sits between the two modes, which is a
        torsion value no side chain adopts; taking the dominant mode gives a
        structure that at least exists.
        """
        if "log_weight" in output:
            best = output["log_weight"].argmax(dim=-1, keepdim=True)
            return output["mean"].gather(-1, best).squeeze(-1)
        if "state_logit" in output:
            state = output["state_logit"].argmax(dim=-1)
            centre = torch.tensor(
                [-2.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0],
                dtype=output["residual"].dtype, device=output["residual"].device,
            )[state]
            return wrap_to_pi(centre + output["residual"])
        return output["mean"]


# --------------------------------------------------------------------------
# losses and metrics
# --------------------------------------------------------------------------


def _fold(residual: Tensor, periodicity: Tensor) -> Tensor:
    """Fold a circular residual into the torsion's own symmetry period."""
    period = 2.0 * math.pi / periodicity.to(residual.dtype)
    return residual - period * torch.round(residual / period)


def circular_mae(
    predicted: Tensor, target: Tensor, mask: Tensor, periodicity: Tensor
) -> Tensor:
    """Symmetry-aware mean absolute circular error, in **radians**.

    ``179`` and ``-179`` degrees are two degrees apart, and for a pi-periodic
    torsion ``0`` and ``180`` are zero apart.
    """
    if not bool(mask.any()):
        return predicted.new_tensor(float("nan"))
    residual = _fold(wrap_to_pi(predicted - target), periodicity)
    return residual[mask].abs().mean()


def von_mises_nll(
    mean: Tensor,
    log_concentration: Tensor,
    target: Tensor,
    mask: Tensor,
    periodicity: Tensor,
) -> Tensor:
    """Negative log likelihood of a von Mises, up to the log-Bessel term.

    ``log I_0(kappa)`` is computed with :func:`torch.special.i0e` in log space so
    that a confident prediction (large kappa) does not overflow -- ``i0`` itself
    overflows float32 around kappa = 90, which a well-trained head reaches.
    """
    kappa = log_concentration.exp().clamp(max=1e4)
    residual = _fold(wrap_to_pi(mean - target), periodicity)
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    nll = -(kappa * torch.cos(residual) - log_i0 - math.log(2 * math.pi))
    return nll[mask].mean() if bool(mask.any()) else nll.new_tensor(float("nan"))


def circular_mixture_nll(
    log_weight: Tensor,
    mean: Tensor,
    log_concentration: Tensor,
    target: Tensor,
    mask: Tensor,
    periodicity: Tensor,
) -> Tensor:
    """NLL of a mixture of von Mises components, ``[N_res, max_chi, K]`` inputs.

    The whole reason for the mixture: a torsion that either stays put or hops
    120 degrees is badly served by any single mode, and a model trained on the
    mean of the two learns to predict a value that never occurs.
    """
    kappa = log_concentration.exp().clamp(max=1e4)
    residual = _fold(
        wrap_to_pi(mean - target.unsqueeze(-1)), periodicity.unsqueeze(-1)
    )
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    component = kappa * torch.cos(residual) - log_i0 - math.log(2 * math.pi)
    log_likelihood = torch.logsumexp(log_weight + component, dim=-1)
    return (
        -log_likelihood[mask].mean()
        if bool(mask.any())
        else log_likelihood.new_tensor(float("nan"))
    )


def chi_state_accuracy_3bin(
    predicted: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    """Fraction of torsions whose predicted **absolute** chi lands in the right bin.

    Not rotamer recovery, and deliberately not called that: a rotamer is a
    library entry for a whole side chain, and this repository has no library. The
    bins are the three sp3 states with boundaries fixed in
    :data:`CHI_STATE_EDGES` before any validation number existed.
    """
    if not bool(mask.any()):
        return predicted.new_tensor(float("nan"))
    edges = torch.tensor(
        CHI_STATE_EDGES[:2], dtype=predicted.dtype, device=predicted.device
    )
    predicted_state = torch.bucketize(wrap_to_pi(predicted), edges)
    target_state = torch.bucketize(wrap_to_pi(target), edges)
    return (predicted_state[mask] == target_state[mask]).to(predicted.dtype).mean()
