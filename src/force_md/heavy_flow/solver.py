"""Small explicit ODE solvers for the heavy-atom flow sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

__all__ = ["FlowSolverConfig", "integrate_flow", "euler_step", "heun_step"]


@dataclass(frozen=True)
class FlowSolverConfig:
    solver: str = "heun"
    steps: int = 20
    spatial_recompute_every: int = 1

    def __post_init__(self) -> None:
        if self.solver.lower() not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if self.steps < 1 or self.spatial_recompute_every < 1:
            raise ValueError("steps and spatial_recompute_every must be positive")


VelocityFn = Callable[[Tensor, Tensor, bool], Tensor]


def euler_step(x: Tensor, time: Tensor, dt: Tensor, velocity: Tensor) -> Tensor:
    return x + dt * velocity


def heun_step(x: Tensor, time: Tensor, dt: Tensor, velocity: Tensor, next_velocity: Tensor) -> Tensor:
    del time
    return x + 0.5 * dt * (velocity + next_velocity)


def integrate_flow(
    x0: Tensor,
    velocity_fn: VelocityFn,
    *,
    solver: str = "heun",
    steps: int = 20,
    spatial_recompute_every: int = 1,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate ``dx/ds=v(x,s)`` from ``s=0`` to ``s=1``.

    ``velocity_fn(x, time, recompute_spatial)`` receives a boolean refresh
    signal.  Heun evaluates the predictor at the next time using the same
    refreshed graph edge set as the current integration interval; the decoder
    still recomputes radial geometry from the predictor coordinates.
    """
    if x0.ndim != 3 or x0.shape[-1] != 3:
        raise ValueError("x0 must have shape [B,N,3]")
    config = FlowSolverConfig(solver=solver, steps=steps, spatial_recompute_every=spatial_recompute_every)
    x = x0
    trajectory = [x]
    for step in range(config.steps):
        t0 = x0.new_tensor(float(step) / config.steps)
        t1 = x0.new_tensor(float(step + 1) / config.steps)
        dt = t1 - t0
        refresh = step % config.spatial_recompute_every == 0
        v0 = velocity_fn(x, t0.expand(x.shape[0]), refresh)
        if v0.shape != x.shape or not torch.isfinite(v0).all():
            raise FloatingPointError("flow velocity is non-finite or has an invalid shape")
        if config.solver.lower() == "euler":
            x = euler_step(x, t0, dt, v0)
        else:
            predictor = euler_step(x, t0, dt, v0)
            v1 = velocity_fn(predictor, t1.expand(x.shape[0]), refresh)
            if v1.shape != x.shape or not torch.isfinite(v1).all():
                raise FloatingPointError("Heun predictor velocity is non-finite or has an invalid shape")
            x = heun_step(x, t1, dt, v0, v1)
        if not torch.isfinite(x).all():
            raise FloatingPointError("flow integration produced non-finite coordinates")
        trajectory.append(x)
    result = x
    if return_trajectory:
        return result, torch.stack(trajectory, dim=0)
    return result
