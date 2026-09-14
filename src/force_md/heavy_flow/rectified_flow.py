"""Conditional rectified-flow path construction and loss.

The future structure is used only to construct a training target.  It is never
accepted by :class:`FlowDecoder` or by the inference sampler.  For internal
conformational transitions, the future is first proper-rotation Kabsch aligned
to the current coordinates; the model still receives the original current
orientation at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..geometry.alignment import RigidAlignment, kabsch_rotation

__all__ = [
    "FlowPath",
    "align_future_to_current",
    "sample_flow_base",
    "build_flow_path",
    "endpoint_from_velocity",
    "rectified_flow_loss",
    "rf_loss",
]


@dataclass(frozen=True)
class FlowPath:
    """One rectified-flow training path.

    All coordinates are dense ``[B,N,3]``.  ``mask`` is the intersection of
    current and future atom validity.  ``alignment`` records the target-only
    Kabsch operation for auditability and is not a model input.
    """

    x0: Tensor
    x1: Tensor
    x_s: Tensor
    target_velocity: Tensor
    flow_time: Tensor
    noise: Tensor
    sigma: Tensor
    mask: Tensor
    alignment: Optional[RigidAlignment] = None

    def validate(self) -> None:
        fields = (self.x0, self.x1, self.x_s, self.target_velocity, self.noise)
        if not all(isinstance(value, Tensor) and value.ndim == 3 and value.shape[-1] == 3 for value in fields):
            raise ValueError("flow path coordinates must all be [B,N,3]")
        if any(value.shape != self.x0.shape for value in fields):
            raise ValueError("flow path coordinate shapes disagree")
        if self.flow_time.shape != (self.x0.shape[0],):
            raise ValueError("flow_time must be [B]")
        if self.sigma.shape != (self.x0.shape[0],):
            raise ValueError("sigma must be [B]")
        if self.mask.shape != self.x0.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("flow mask must be [B,N] bool")


def _as_mask(current: Tensor, mask: Optional[Tensor]) -> Tensor:
    if current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B,N,3]")
    if mask is None:
        return torch.ones(current.shape[:2], dtype=torch.bool, device=current.device)
    if mask.shape != current.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be [B,N] bool")
    return mask.to(device=current.device)


def align_future_to_current(
    current: Tensor,
    future: Tensor,
    *,
    atom_mask: Optional[Tensor] = None,
    future_mask: Optional[Tensor] = None,
) -> tuple[Tensor, RigidAlignment]:
    """Return ``X_1 = Align(X_future, X_current)`` using a proper rotation."""
    if current.shape != future.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("current and future must have identical [B,N,3] shape")
    mask = _as_mask(current, atom_mask) & _as_mask(future, future_mask)
    batch, atoms = current.shape[:2]
    batch_index = torch.arange(batch, device=current.device, dtype=torch.int64)[:, None].expand(batch, atoms).reshape(-1)
    alignment = kabsch_rotation(
        future.reshape(-1, 3), current.reshape(-1, 3), batch_index, batch,
        weights=mask.reshape(-1).to(current.dtype),
    )
    aligned = alignment.apply(future.reshape(-1, 3), batch_index).reshape_as(future)
    # There is no target on masked rows.  Keeping those rows at current makes
    # every downstream tensor finite and the explicit mask still excludes them.
    aligned = torch.where(mask[..., None], aligned, current)
    return aligned, alignment


def _center_noise(noise: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(noise.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (noise * weights[..., None]).sum(dim=1, keepdim=True) / count[..., None]
    return (noise - center) * weights[..., None]


def sample_flow_base(
    current: Tensor,
    *,
    atom_mask: Optional[Tensor] = None,
    lag: Optional[Tensor] = None,
    sigma_scale: float = 0.1,
    remove_center_of_mass: bool = True,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ``X_0 = X_t + sigma(Delta) epsilon``.

    The default ``sigma(Delta) = sigma_scale * sqrt(max(Delta, 1e-6))`` is an
    explicit bounded base-distribution choice, not a learned variance.  When
    requested, protein-wide centre-of-mass translation is removed from noise.
    """
    mask = _as_mask(current, atom_mask)
    batch = current.shape[0]
    if lag is None:
        lag = current.new_ones((batch,))
    if lag.shape != (batch,) or bool((lag <= 0).any()):
        raise ValueError("lag must be positive with shape [B]")
    if sigma_scale < 0:
        raise ValueError("sigma_scale must be non-negative")
    noise = torch.randn(current.shape, dtype=current.dtype, device=current.device, generator=generator)
    noise = noise * mask[..., None].to(noise.dtype)
    if remove_center_of_mass:
        noise = _center_noise(noise, mask)
    sigma = sigma_scale * lag.clamp_min(1e-6).sqrt()
    x0 = current + sigma[:, None, None] * noise
    x0 = torch.where(mask[..., None], x0, current)
    return x0, noise, sigma


def build_flow_path(
    current: Tensor,
    future: Tensor,
    *,
    atom_mask: Optional[Tensor] = None,
    future_mask: Optional[Tensor] = None,
    lag: Optional[Tensor] = None,
    flow_time: Optional[Tensor] = None,
    noise: Optional[Tensor] = None,
    sigma_scale: float = 0.1,
    remove_center_of_mass_noise: bool = True,
    generator: Optional[torch.Generator] = None,
) -> FlowPath:
    """Construct ``X_s=(1-s)X_0+sX_1`` and ``u_s=X_1-X_0``."""
    if current.shape != future.shape:
        raise ValueError("current and future must have identical shapes")
    mask = _as_mask(current, atom_mask) & _as_mask(future, future_mask)
    x1, alignment = align_future_to_current(
        current, future, atom_mask=mask, future_mask=mask
    )
    if lag is None:
        lag = current.new_ones((current.shape[0],))
    if noise is None:
        x0, sampled_noise, sigma = sample_flow_base(
            current,
            atom_mask=mask,
            lag=lag,
            sigma_scale=sigma_scale,
            remove_center_of_mass=remove_center_of_mass_noise,
            generator=generator,
        )
    else:
        if noise.shape != current.shape:
            raise ValueError("noise must have shape [B,N,3]")
        sampled_noise = noise * mask[..., None].to(noise.dtype)
        if remove_center_of_mass_noise:
            sampled_noise = _center_noise(sampled_noise, mask)
        sigma = sigma_scale * lag.clamp_min(1e-6).sqrt()
        x0 = torch.where(mask[..., None], current + sigma[:, None, None] * sampled_noise, current)
    if flow_time is None:
        flow_time = torch.rand((current.shape[0],), dtype=current.dtype, device=current.device, generator=generator)
    elif flow_time.ndim == 0:
        flow_time = flow_time.expand(current.shape[0]).to(device=current.device, dtype=current.dtype)
    elif flow_time.shape != (current.shape[0],):
        raise ValueError("flow_time must be scalar or [B]")
    flow_time = flow_time.to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
    x1 = torch.where(mask[..., None], x1, current)
    x_s = (1.0 - flow_time[:, None, None]) * x0 + flow_time[:, None, None] * x1
    target_velocity = (x1 - x0) * mask[..., None].to(current.dtype)
    path = FlowPath(x0, x1, x_s, target_velocity, flow_time, sampled_noise, sigma, mask, alignment)
    path.validate()
    return path


def endpoint_from_velocity(x_s: Tensor, flow_time: Tensor, velocity: Tensor) -> Tensor:
    """Rectified-flow endpoint estimate ``X_hat_1=X_s+(1-s)v_theta``."""
    if x_s.shape != velocity.shape or x_s.ndim != 3 or x_s.shape[-1] != 3:
        raise ValueError("x_s and velocity must have identical [B,N,3] shape")
    if flow_time.shape != (x_s.shape[0],):
        raise ValueError("flow_time must be [B]")
    return x_s + (1.0 - flow_time[:, None, None]) * velocity


def rectified_flow_loss(
    flow_velocity: Tensor,
    target_velocity: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Masked per-atom RF MSE, averaged within each protein then over proteins."""
    if flow_velocity.shape != target_velocity.shape or flow_velocity.ndim != 3 or flow_velocity.shape[-1] != 3:
        raise ValueError("flow_velocity and target_velocity must be [B,N,3]")
    if atom_mask.shape != flow_velocity.shape[:2] or atom_mask.dtype != torch.bool:
        raise ValueError("atom_mask must be [B,N] bool")
    squared = (flow_velocity - target_velocity).float().square().sum(dim=-1)
    mask = atom_mask.to(squared.dtype)
    per_graph = (squared * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    active = mask.sum(dim=1) > 0
    # Keep an autograd-connected zero for the all-masked fixture.
    return (per_graph * active.to(per_graph.dtype)).sum() / active.to(per_graph.dtype).sum().clamp_min(1.0)


rf_loss = rectified_flow_loss
