"""Stage 3 force likelihoods and train-split normalization.

Ground-truth force tensors are intentionally referenced only in this module
(and metrics/audit code).  The predictor and force head are target-free.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import Tensor

from .force_head import ForceDistribution
from .types import HeavyFlowCondition, HeavyFlowSample, HeavyFlowTargets

__all__ = [
    "ForceLossConfig", "ForceNormalizer", "heteroscedastic_force_nll",
    "masked_heteroscedastic_gaussian_nll", "force_nll", "masked_force_nll",
    "force_loss", "loss_with_diagnostics", "variance_inflation_diagnostic",
]


@dataclass(frozen=True)
class ForceLossConfig:
    logvar_min: float = -10.0
    logvar_max: float = 10.0
    epsilon: float = 1e-6
    include_constant: bool = False

    def __post_init__(self) -> None:
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class ForceNormalizer:
    """One scalar force RMS fitted on the training split only."""

    scale: float = 1.0
    count: int = 0
    force_unit: str = "kcal/mol/angstrom"
    fit_source: str = "train"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.scale)) or self.scale <= 0:
            raise ValueError("normalizer scale must be finite and positive")
        if self.count < 0:
            raise ValueError("normalizer count cannot be negative")

    @classmethod
    def fit(cls, force: Tensor, mask: Optional[Tensor] = None, *, force_unit: str = "kcal/mol/angstrom") -> "ForceNormalizer":
        if not isinstance(force, Tensor) or force.shape[-1] != 3:
            raise ValueError("force must be a tensor ending in width 3")
        values = force.reshape(-1, 3)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=force.device).reshape(-1)
            if mask.numel() != values.shape[0]:
                raise ValueError("normalizer mask must cover the force leading dimensions")
            values = values[mask]
        values = values[torch.isfinite(values).all(dim=-1)]
        if values.numel() == 0:
            raise ValueError("cannot fit force normalizer with no finite valid atoms")
        rms = values.float().square().mean().sqrt().clamp_min(1e-8)
        return cls(float(rms), int(values.shape[0]), force_unit, "train")

    @classmethod
    def fit_from_samples(cls, samples: Iterable[HeavyFlowSample], *, force_unit: Optional[str] = None) -> "ForceNormalizer":
        # Keep this pass streaming.  A 500-domain chunk can contain tens of
        # thousands of frames, so retaining every force tensor just to fit one
        # scalar normalizer would defeat chunk rotation's memory boundary.
        sum_squares = 0.0
        component_count = 0
        atom_count = 0
        unit = force_unit
        for sample in samples:
            sample.validate()
            target = sample.targets
            values = target.force_current.detach().reshape(-1, 3)
            mask = target.force_mask if target.force_mask is not None else sample.condition.atom_mask
            values = values[torch.as_tensor(mask, dtype=torch.bool, device=values.device).reshape(-1)]
            values = values[torch.isfinite(values).all(dim=-1)]
            if values.numel():
                sum_squares += float(values.float().square().sum())
                component_count += int(values.numel())
                atom_count += int(values.shape[0])
            if unit is None and sample.condition.provenance is not None:
                unit = sample.condition.provenance.force_unit
        if atom_count == 0 or component_count == 0:
            raise ValueError("cannot fit force normalizer from an empty sample iterable")
        rms = math.sqrt(max(sum_squares / component_count, 1e-16))
        return cls(rms, atom_count, unit or "kcal/mol/angstrom", "train")

    @classmethod
    def fit_from_force_arrays(
        cls,
        samples: Iterable[tuple[Any, Any]],
        *,
        force_unit: str = "kcal/mol/angstrom",
    ) -> "ForceNormalizer":
        """Fit from ``(force, force_mask)`` without constructing full samples.

        The mdCATH normalizer path uses this method so that coordinates,
        residue semantics, and PLM embeddings are not read merely to compute a
        scalar force RMS.  ``force`` is ``[N,3]`` and ``force_mask`` is
        ``[N]``; masked or non-finite rows do not contribute.
        """
        sum_squares = 0.0
        component_count = 0
        atom_count = 0
        for force, mask in samples:
            values = torch.as_tensor(force)
            if values.ndim != 2 or values.shape[-1] != 3:
                raise ValueError(
                    "force arrays must have shape [N,3], "
                    f"got {tuple(values.shape)}"
                )
            valid = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
            if valid.shape != (values.shape[0],):
                raise ValueError(
                    "force masks must have shape [N], "
                    f"got {tuple(valid.shape)} for {tuple(values.shape)}"
                )
            values = values[valid]
            values = values[torch.isfinite(values).all(dim=-1)]
            if values.numel():
                values = values.float()
                sum_squares += float(values.square().sum())
                component_count += int(values.numel())
                atom_count += int(values.shape[0])
        if atom_count == 0 or component_count == 0:
            raise ValueError("cannot fit force normalizer from empty force arrays")
        rms = math.sqrt(max(sum_squares / component_count, 1e-16))
        return cls(rms, atom_count, force_unit, "train")

    def transform(self, force: Tensor) -> Tensor:
        return force / force.new_tensor(self.scale)

    def inverse_transform(self, force: Tensor) -> Tensor:
        return force * force.new_tensor(self.scale)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def state_dict(self) -> dict[str, Any]:
        return self.as_dict()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForceNormalizer":
        return cls(**{key: value[key] for key in ("scale", "count", "force_unit", "fit_source", "epsilon") if key in value})

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "ForceNormalizer":
        return cls.from_dict(value)


def _check_force_shapes(force_mean: Tensor, force_logvar: Tensor, target: Tensor) -> None:
    if force_mean.shape != target.shape or force_mean.ndim != 3 or force_mean.shape[-1] != 3:
        raise ValueError("force_mean and target must both have shape [B,N,3]")
    if force_logvar.ndim != 3 or force_logvar.shape[:2] != target.shape[:2] or force_logvar.shape[-1] not in (1, 3):
        raise ValueError("force_logvar must have shape [B,N,1] or [B,N,3]")


def _safe_atom_batch(values: Tensor, mask: Tensor) -> Tensor:
    # Protein-level averaging prevents a long protein from dominating the
    # objective.  Empty proteins contribute a finite zero.
    numerator = (values * mask.to(values.dtype)).sum(dim=-1)
    denominator = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
    per_protein = numerator / denominator
    active = mask.any(dim=-1)
    if active.any():
        return per_protein[active].mean()
    return values.sum() * 0.0


def heteroscedastic_force_nll(
    force_mean: Tensor,
    force_logvar: Tensor,
    target_force: Tensor,
    mask: Optional[Tensor] = None,
    *,
    normalizer: Optional[ForceNormalizer] = None,
    logvar_min: float = -10.0,
    logvar_max: float = 10.0,
    epsilon: float = 1e-6,
    include_constant: bool = False,
) -> Tensor:
    """Masked isotropic/per-component Gaussian NLL with atom then protein mean."""
    _check_force_shapes(force_mean, force_logvar, target_force)
    if mask is None:
        mask = torch.ones(target_force.shape[:2], dtype=torch.bool, device=target_force.device)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=target_force.device)
        if mask.shape != target_force.shape[:2]:
            raise ValueError("force mask must have shape [B,N]")
    if logvar_min >= logvar_max or epsilon <= 0:
        raise ValueError("invalid variance bounds or epsilon")
    # Keep exponentiation and quadratic arithmetic in fp32 under AMP.  The
    # returned scalar remains differentiable with respect to model outputs.
    arithmetic_dtype = torch.float32 if force_mean.dtype in (torch.float16, torch.bfloat16) else force_mean.dtype
    mean = force_mean.to(arithmetic_dtype)
    target = target_force.to(arithmetic_dtype)
    logvar = force_logvar.to(arithmetic_dtype).clamp(logvar_min, logvar_max)
    if normalizer is not None:
        scale = mean.new_tensor(normalizer.scale)
        mean, target = mean / scale, target / scale
        logvar = logvar - 2.0 * torch.log(scale)
        logvar = logvar.clamp(logvar_min, logvar_max)
    variance = torch.exp(logvar).clamp_min(epsilon)
    squared = (target - mean).square()
    if logvar.shape[-1] == 1:
        per_atom = squared.sum(dim=-1) / (2.0 * variance[..., 0]) + 1.5 * logvar[..., 0]
        if include_constant:
            per_atom = per_atom + 1.5 * math.log(2.0 * math.pi)
    else:
        per_atom = (squared / (2.0 * variance)).sum(dim=-1) + 0.5 * logvar.sum(dim=-1)
        if include_constant:
            per_atom = per_atom + 1.5 * math.log(2.0 * math.pi)
    return _safe_atom_batch(per_atom, mask)


masked_heteroscedastic_gaussian_nll = heteroscedastic_force_nll
force_nll = heteroscedastic_force_nll
masked_force_nll = heteroscedastic_force_nll


def _prediction_fields(prediction: ForceDistribution | Any) -> tuple[Tensor, Tensor]:
    if isinstance(prediction, ForceDistribution):
        return prediction.force_mean, prediction.force_logvar
    for mean_name, logvar_name in (("force_mean", "force_logvar"), ("mean", "logvar")):
        if hasattr(prediction, mean_name) and hasattr(prediction, logvar_name):
            return getattr(prediction, mean_name), getattr(prediction, logvar_name)
    raise TypeError("prediction must expose force_mean/force_logvar")


def force_loss(
    prediction: ForceDistribution | Any,
    targets: HeavyFlowTargets,
    condition: HeavyFlowCondition,
    *,
    normalizer: Optional[ForceNormalizer] = None,
    config: Optional[ForceLossConfig] = None,
) -> Tensor:
    """Compute the only Stage 3 training loss that consumes GT force."""
    condition.validate()
    targets.validate(condition)
    mean, logvar = _prediction_fields(prediction)
    mask = targets.force_mask if targets.force_mask is not None else condition.atom_mask
    return heteroscedastic_force_nll(
        mean, logvar, targets.force_current, mask,
        normalizer=normalizer,
        **dataclasses.asdict(config or ForceLossConfig()),
    )


def loss_with_diagnostics(
    prediction: ForceDistribution | Any,
    targets: HeavyFlowTargets,
    condition: HeavyFlowCondition,
    *,
    normalizer: Optional[ForceNormalizer] = None,
    config: Optional[ForceLossConfig] = None,
) -> dict[str, Tensor]:
    mean, logvar = _prediction_fields(prediction)
    mask = targets.force_mask if targets.force_mask is not None else condition.atom_mask
    objective = force_loss(prediction, targets, condition, normalizer=normalizer, config=config)
    values = logvar.detach().float()[mask]
    return {
        "loss": objective,
        "mean_logvar": values.mean() if values.numel() else logvar.sum() * 0.0,
        "std_logvar": values.std(unbiased=False) if values.numel() else logvar.sum() * 0.0,
        "mean_abs_force": mean.detach().float()[mask].abs().mean() if bool(mask.any()) else mean.sum() * 0.0,
    }


def variance_inflation_diagnostic(
    force_mean: Tensor,
    force_logvar: Tensor,
    target_force: Tensor,
    mask: Optional[Tensor] = None,
    *,
    normalizer: Optional[ForceNormalizer] = None,
) -> dict[str, float]:
    """Report whether a lower NLL is coming mostly from variance inflation."""
    base = heteroscedastic_force_nll(force_mean, force_logvar, target_force, mask, normalizer=normalizer)
    shifted = heteroscedastic_force_nll(force_mean, force_logvar + 1.0, target_force, mask, normalizer=normalizer)
    logvar_values = force_logvar.detach().float()
    if mask is not None:
        logvar_values = logvar_values[mask]
    return {
        "nll": float(base.detach()),
        "nll_logvar_plus_one": float(shifted.detach()),
        "mean_logvar": float(logvar_values.mean()) if logvar_values.numel() else 0.0,
        "variance_inflation_worsens_nll": float(shifted > base),
    }
