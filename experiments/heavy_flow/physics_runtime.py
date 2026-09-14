"""Bounded Stage 3 input pipeline and DDP loss/update utilities."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from force_md.heavy_flow import collate_heavy_flow, force_loss


def rank_world() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def rank_zero_call(function):
    """Propagate rank-zero I/O failures, rather than leaving peers waiting."""
    rank, world = rank_world()
    result = [None, None]
    if rank == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    return result[0]


def capture_rng() -> list[dict[str, Any]]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    _, world = rank_world()
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    return states


def restore_rng(states: list[dict[str, Any]]) -> None:
    rank, world = rank_world()
    if len(states) != world:
        raise ValueError("exact RNG resume requires the saved world size")
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def batch_indices(size: int, batch_size: int, rank: int, world: int, step: int) -> list[int]:
    """Exact epoch coverage without DistributedSampler's duplicate padding."""
    width = batch_size * world
    epoch_steps = (size + width - 1) // width
    start = (step % epoch_steps) * width + rank * batch_size
    return list(range(start, min(start + batch_size, size)))


def batches(dataset, *, batch_size: int, start_step: int, steps: int, prefetch: int):
    """One HDF5 reader thread; bounded CPU batches, no forked HDF5/CUDA state.

    Stage 3 frame construction is deterministic and does not consume RNG.
    Always join before the caller closes the reader, including on exceptions.
    """
    rank, world = rank_world()

    def read(step):
        indices = batch_indices(len(dataset), batch_size, rank, world, step)
        sample = collate_heavy_flow([dataset[i] for i in (indices or [0])])
        mask = sample.targets.force_mask
        if mask is None:
            mask = sample.condition.atom_mask
        active = int(mask.any(dim=-1).sum()) if indices else 0
        return sample, active

    if not prefetch:
        for step in range(start_step, steps):
            yield read(step)
        return
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage3-input")
    pending = deque()
    iterator = iter(range(start_step, steps))
    try:
        for _ in range(prefetch):
            step = next(iterator, None)
            if step is not None:
                pending.append(executor.submit(read, step))
        while pending:
            batch = pending.popleft().result()
            step = next(iterator, None)
            if step is not None:
                pending.append(executor.submit(read, step))
            yield batch
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class PhysicsObjective(torch.nn.Module):
    """Return only the scalar loss to DDP's unused-parameter traversal.

    Traverse actual loss dependencies, including force-message pair latents
    and axial invariants used by variance. Empty graphs and irrep selection
    can still leave individual parameters unused.
    """
    def __init__(self, model, normalizer, loss_config):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.loss_config = loss_config

    def forward(self, sample):
        output = self.model(sample.condition)
        return force_loss(output.force_distribution, sample.targets, sample.condition,
                          normalizer=self.normalizer, config=self.loss_config)


def train_stream(model, dataset, optimizer, *, normalizer, steps, batch_size,
                 loss_config, step_offset, progress_callback, start_step=0,
                 prefetch=1):
    if steps < 1 or batch_size < 1 or not 0 <= start_step < steps or prefetch < 0:
        raise ValueError("invalid steps, batch size, start step or prefetch depth")
    if not len(dataset) or step_offset < 0:
        raise ValueError("empty dataset or negative step offset")
    rank, world = rank_world()
    device = next(model.parameters()).device
    model.train()
    objective_model = PhysicsObjective(model, normalizer, loss_config)
    if world > 1:
        objective_model = DistributedDataParallel(
            objective_model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    losses = []
    stream = batches(dataset, batch_size=batch_size, start_step=start_step,
                     steps=steps, prefetch=prefetch)
    try:
        for local_step, (sample, active) in enumerate(stream, start=start_step):
            sample = sample.to(device)
            optimizer.zero_grad(set_to_none=True)
            objective = objective_model(sample)
            count = torch.tensor(float(active), device=device)
            finite = torch.isfinite(objective).to(torch.int32)
            if world > 1:
                dist.all_reduce(count)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError(f"non-finite Stage 3 loss at step {step_offset + local_step}")
            # DDP averages gradients over ranks. Weight by active proteins,
            # not by atom count or rank count; empty tail ranks contribute 0.
            weighted = objective * (world * active / count.clamp_min(1))
            weighted.backward()
            grad_finite = torch.ones((), dtype=torch.int32, device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_finite.mul_(torch.isfinite(parameter.grad).all().to(torch.int32))
            if world > 1:
                dist.all_reduce(grad_finite, op=dist.ReduceOp.MIN)
            if not bool(grad_finite):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite Stage 3 gradient at step {step_offset + local_step}; optimizer not updated")
            optimizer.step()
            mean_loss = objective.detach() * active / count.clamp_min(1)
            if world > 1:
                dist.all_reduce(mean_loss)
            losses.append(float(mean_loss))
            # Release large autograd outputs before checkpoint I/O.
            del weighted, objective, sample
            if progress_callback is not None:
                progress_callback(local_step + 1, losses[-1])
    finally:
        stream.close()
    return losses
