"""Phase 1.5 training loop for the transition probe.

Deliberately thinner than :mod:`force_md.training.phase1_module`, because the
probe is a small deterministic model and the experiment it serves is a
comparison, not a performance run. What it does keep from Phase 1 are the four
things that were expensive to learn there:

**Every arm must see the identical experiment.** Same manifest, same seed, same
batch order, same optimiser and schedule, same step budget. The only difference
permitted is which conditioner the probe was built with. :meth:`TransitionTrainer.provenance`
records the manifest hash, the split, the Phase 1 checkpoint hash and the arm's
parameter counts into every checkpoint and every results row, so "were these two
numbers produced under the same conditions" is answerable after the fact rather
than by trust.

**Validation is reported twice.** ``micro`` weights residues, ``domain_macro``
weights domains. A gain that appears in only one of them is not a gain, and Phase
1's habit of quoting a single averaged number is what this avoids.

**Everything is quoted against the identity baseline.** "Nothing moves" is a
strong predictor at 1-4 ns, so a Ca RMSD without its baseline is unreadable.

**A non-finite step costs one batch, not the run.** The backward always runs --
DDP installs its hooks during forward and expects exactly one backward per
forward -- and only the optimiser step is withheld.

Ground-truth forces and the future frame enter here and nowhere else: the target
is built in the trainer, the conditioner bundle is built separately, and the
probe's forward signature cannot accept either.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from ..data.adapters.lag_pairs import LagPairBatch, LagPairManifest
from ..transition.conditioners import ConditionerConfig
from ..transition.losses import TransitionLossWeights, transition_loss
from ..transition.metrics import (
    MetricConfig,
    aggregate_metric_records,
    metric_records,
)
from ..transition.phase1_features import FrozenPhase1Extractor
from ..transition.probe import TransitionProbe, TransitionProbeConfig
from ..transition.targets import build_transition_target, identity_prediction
from .phase1_module import set_seed

__all__ = ["TransitionTrainConfig", "TransitionTrainer"]


@dataclass(frozen=True)
class TransitionTrainConfig:
    """Training hyper-parameters, identical across the arms of one ablation.

    Args:
        learning_rate: 1e-3 by default. A single-pair sweep found 3e-3 fits
            fastest (loss 1.26 -> 0.003 in 600 steps against 0.019 at 1e-3), and
            it is still not the default: this repository's Phase 1 config records
            "3e-3 diverged. Twice." over a long run, and a short sweep measures
            which rate descends fastest, not which survives. 3e-3 also bounced at
            step 50 in that same sweep.
        warmup_steps / lr_schedule / min_lr_factor: the Phase 1 shape.
        max_steps: step-bounded, not epoch-bounded; trajectory counts vary.
        grad_clip: max global gradient norm.
        eval_every / eval_batches: validation cadence and cap.
        checkpoint_every: 0 disables periodic checkpoints.
        max_consecutive_skips: abort after this many unusable steps in a row --
            that is divergence, not a bad frame.
        max_loss: a step whose loss exceeds this is treated as unusable even
            though it is finite. The identity baseline scores ~1.2 and a healthy
            step stays under 10, so 1e4 cannot fire on a merely bad batch; it
            exists because the probe once reached 1e14 without a single non-finite
            value, which the finiteness check alone did not catch.
        seed: seeds python/numpy/torch **and** the sampler, so the batch order is
            part of what the seed fixes.
    """

    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 2000
    warmup_steps: int = 100
    lr_schedule: str = "cosine"
    min_lr_factor: float = 0.05
    grad_clip: float = 10.0
    eval_every: int = 250
    eval_batches: Optional[int] = None
    log_every: int = 50
    checkpoint_every: int = 0
    max_consecutive_skips: int = 20
    max_loss: float = 1e4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_weights: TransitionLossWeights = field(default_factory=TransitionLossWeights)
    metrics: MetricConfig = field(default_factory=MetricConfig)


class TransitionTrainer:
    """Trains one arm of the Phase 1.5 ablation.

    Args:
        probe: the model. Its ``config.arm`` names the arm.
        extractor: frozen Phase 1. Never trained, never checkpointed here.
        config: hyper-parameters, shared across arms.
        distributed: wrap the probe in ``DistributedDataParallel``. Execution
            only -- it changes throughput and nothing about the objective.
    """

    def __init__(
        self,
        probe: TransitionProbe,
        extractor: FrozenPhase1Extractor,
        config: TransitionTrainConfig = TransitionTrainConfig(),
        *,
        distributed: bool = False,
        manifest: Optional[LagPairManifest] = None,
    ):
        set_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        self.module = probe.to(self.device)
        self.extractor = extractor.to(self.device)
        self.manifest = manifest
        self.distributed = distributed
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )
        else:
            self.model = self.module
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step = 0
        self.history: list[dict] = []
        self.skipped_steps = 0
        self.consecutive_skips = 0

    # -- bookkeeping -------------------------------------------------------

    @property
    def arm(self) -> str:
        return self.module.config.arm

    @property
    def is_main(self) -> bool:
        if not self.distributed:
            return True
        import torch.distributed as dist  # noqa: PLC0415

        return dist.get_rank() == 0

    def _barrier(self) -> None:
        if self.distributed:
            import torch.distributed as dist  # noqa: PLC0415

            dist.barrier()

    def provenance(self) -> dict:
        """Everything needed to say two runs were the same experiment."""
        breakdown = self.module.parameter_breakdown()
        return {
            "arm": self.arm,
            "seed": self.config.seed,
            "parameter_count": breakdown["total"],
            "trainable_parameter_count": self.module.trainable_parameter_count(),
            "parameter_breakdown": breakdown,
            "phase1_checkpoint": self.extractor.metadata.get("checkpoint_path"),
            "phase1_sha256": self.extractor.metadata.get("checkpoint_sha256"),
            "phase1_step": self.extractor.metadata.get("step"),
            "latent_contract": self.extractor.contract,
            "manifest_hash": (
                self.manifest.content_hash() if self.manifest is not None else None
            ),
            "manifest_pairs": len(self.manifest) if self.manifest is not None else None,
            "manifest_domains": (
                self.manifest.metadata.get("num_domains") if self.manifest else None
            ),
            "train": dataclasses.asdict(self.config),
            "probe": {
                "num_blocks": self.module.config.num_blocks,
                "d_cond": self.module.config.conditioner.d_cond,
                "history_length": self.module.config.history_length,
                "irreps": str(self.module.irreps),
            },
        }

    # -- learning rate -----------------------------------------------------

    def _set_lr(self) -> float:
        cfg = self.config
        t = self.step + 1
        if cfg.warmup_steps > 0 and t <= cfg.warmup_steps:
            scale = t / cfg.warmup_steps
        elif cfg.lr_schedule == "cosine":
            span = max(1, cfg.max_steps - cfg.warmup_steps)
            progress = min(max((t - cfg.warmup_steps) / span, 0.0), 1.0)
            scale = cfg.min_lr_factor + (1.0 - cfg.min_lr_factor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        elif cfg.lr_schedule == "constant":
            scale = 1.0
        else:
            raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
        lr = cfg.learning_rate * scale
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- one step ----------------------------------------------------------

    def _forward(self, batch: LagPairBatch, *, model=None):
        """Target, prediction and loss for one batch.

        The future frame is read **here**, to build the target, and is never
        passed to the probe -- whose signature could not accept it anyway.
        """
        batch = batch.to(self.device)
        target = build_transition_target(batch.current, batch.future)
        bundle = (
            self.extractor.oracle_bundle(batch.current)
            if self.module.conditioner.requires_oracle
            else self.extractor(batch.current)
        )
        prediction = (model if model is not None else self.model)(
            batch.current, bundle, history=batch.history, lag_ps=batch.lag_ps
        )
        total, components = transition_loss(
            prediction, target, weights=self.config.loss_weights
        )
        return total, components, prediction, target, batch

    def train_step(self, batch: LagPairBatch) -> dict:
        self.model.train()
        lr = self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)
        total, components, _, _, _ = self._forward(batch)

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), self.config.grad_clip
        )
        # A loss can diverge without ever becoming non-finite: the probe's
        # body-order-3 blocks once reached 1e14 while every value stayed a
        # perfectly good float, so the finiteness test below passed and the run
        # would have burned its whole budget silently. Treat an absurd magnitude
        # as the divergence it is.
        absurd = abs(components["total"]) > self.config.max_loss
        bad = absurd or not (
            math.isfinite(components["total"]) and bool(torch.isfinite(grad_norm))
        )
        if self._any_rank(bad):
            self.skipped_steps += 1
            self.consecutive_skips += 1
            if self.consecutive_skips > self.config.max_consecutive_skips:
                raise RuntimeError(
                    f"{self.consecutive_skips} consecutive unusable steps at step "
                    f"{self.step} (last loss {components['total']:.4g}, grad norm "
                    f"{float(grad_norm):.4g}): that is divergence, not a bad frame"
                )
        else:
            self.optimizer.step()
            self.consecutive_skips = 0

        self.step += 1
        components["grad_norm"] = float(grad_norm)
        components["lr"] = lr
        components["skipped"] = float(self.skipped_steps)
        return components

    def _any_rank(self, flag: bool) -> bool:
        """A skip must be collective, or the replicas stop being copies."""
        if not self.distributed:
            return flag
        import torch.distributed as dist  # noqa: PLC0415

        signal = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(signal, op=dist.ReduceOp.SUM)
        return bool(signal.item() > 0)

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, loader: Iterable, *, split: str = "val", max_batches: Optional[int] = None
    ) -> tuple[dict, list[dict]]:
        """Loss and geometry metrics over a loader.

        Args:
            max_batches: ``None`` evaluates the **whole** loader. It deliberately
                does not fall back to ``config.eval_batches``: that knob exists to
                keep the periodic in-training check cheap, and letting it leak in
                here once silently reduced a 72,080-pair / 181-domain validation
                to 1,600 pairs from 4 domains -- with `shuffle=False` the same
                first four domains every time, so nothing looked wrong. The
                training loop passes the cap explicitly; every other caller wants
                the full set.

        Returns:
            ``(summary, records)``. ``summary`` holds the aggregated rows keyed by
            ``lag_ns`` with both ``micro`` and ``domain_macro`` averages plus the
            identity baseline; ``records`` is one tidy row per pair, ready for a CSV.
        """
        self.module.eval()
        losses: list[dict] = []
        records: list[dict] = []
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            total, components, prediction, target, moved = self._forward(
                batch, model=self.module
            )
            losses.append(components)
            records.extend(
                metric_records(
                    prediction,
                    target,
                    domains=[p.domain for p in moved.pairs],
                    lag_ps=[p.lag_ps for p in moved.pairs],
                    split=split,
                    config=self.config.metrics,
                )
            )
        if not losses:
            return {}, []

        summary = {
            f"loss_{k}": float(np.mean([row[k] for row in losses]))
            for k in losses[0]
            if isinstance(losses[0][k], float)
        }
        summary["rows"] = aggregate_metric_records(records)
        return summary, records

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        *,
        log=print,
        checkpoint_path: Optional[str] = None,
    ) -> list[dict]:
        start = time.time()
        if log is print:
            # Redirected stdout is block-buffered, so a bare print makes a long
            # run look hung for tens of minutes at a time -- a 40k-step run
            # reported step 2000 while it was actually at step 20000. Progress
            # logs are worthless if they arrive in 8 kB batches.
            log = functools.partial(print, flush=True)
        if not self.is_main:
            log = None
        epoch = 0
        self._set_epoch(train_loader, epoch)
        iterator = iter(train_loader)

        while self.step < self.config.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                self._set_epoch(train_loader, epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            components = self.train_step(batch)

            if log and self.step % self.config.log_every == 0:
                log(
                    f"[{self.arm}] step {self.step:6d} | loss={components['total']:.4f} "
                    f"| rmsd={components['translation_rmse_angstrom']:.3f}A "
                    f"| rot={components['rotation_error_deg']:.2f}deg "
                    f"| grad={components['grad_norm']:.2f} | lr={components['lr']:.2e} "
                    f"| ep{epoch} | {time.time() - start:.0f}s"
                )
            if val_loader is not None and self.step % self.config.eval_every == 0:
                if self.is_main:
                    # eval_batches is a *training-cadence* knob: it keeps the
                    # periodic check cheap. It must never reach the final
                    # measurement, which has to see the whole validation set.
                    summary, _ = self.evaluate(
                        val_loader, max_batches=self.config.eval_batches
                    )
                    self.history.append({"step": self.step, **_flatten(summary)})
                    if log:
                        log(f"  {_format_validation(self.arm, self.step, summary)}")
                self._barrier()
            if (
                checkpoint_path
                and self.config.checkpoint_every
                and self.step % self.config.checkpoint_every == 0
            ):
                if self.is_main:
                    self.save_checkpoint(checkpoint_path)
                self._barrier()

        if checkpoint_path and self.is_main:
            self.save_checkpoint(checkpoint_path)
        self._barrier()
        return self.history

    @staticmethod
    def _set_epoch(loader: Iterable, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # -- checkpoints -------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Atomic, and never persists non-finite weights."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        bad = [
            k
            for k, v in self.module.state_dict().items()
            if v.is_floating_point() and not torch.isfinite(v).all()
        ]
        if bad:
            raise RuntimeError(
                f"refusing to checkpoint at step {self.step}: non-finite weights in "
                f"{bad[:5]}. The previous checkpoint at {path} is kept."
            )
        tmp = f"{path}.tmp"
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "probe_config": self.module.config,
                "train_config": self.config,
                "latent_irreps": self.extractor.contract["physics_latent_irreps"],
                "step": self.step,
                "history": self.history,
                "skipped_steps": self.skipped_steps,
                "provenance": self.provenance(),
            },
            tmp,
        )
        os.replace(tmp, path)

    def load_state(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.module.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = payload["step"]
        self.history = payload["history"]
        self.skipped_steps = payload.get("skipped_steps", 0)

    @staticmethod
    def load_checkpoint(
        path: str, extractor: FrozenPhase1Extractor, device: str = "cpu"
    ) -> tuple[TransitionProbe, "TransitionTrainer"]:
        """Rebuild probe and trainer exactly as saved."""
        payload = torch.load(path, map_location=device, weights_only=False)
        probe = TransitionProbe(
            payload["probe_config"], latent_irreps=payload["latent_irreps"]
        )
        probe.load_state_dict(payload["state_dict"])
        config = dataclasses.replace(payload["train_config"], device=device)
        trainer = TransitionTrainer(probe, extractor, config)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.step = payload["step"]
        trainer.history = payload["history"]
        return probe, trainer


def _flatten(summary: dict) -> dict:
    """Turn the nested aggregation into flat keys a history JSON can hold."""
    out = {k: v for k, v in summary.items() if k != "rows"}
    for row in summary.get("rows", []):
        lag = row.get("lag_ns")
        for key, value in row.items():
            if isinstance(value, (int, float)):
                out[f"lag{lag:g}_{key}"] = value
    return out


def _format_validation(arm: str, step: int, summary: dict) -> str:
    parts = [f"val[{arm}] {step}"]
    for row in summary.get("rows", []):
        parts.append(
            f"| {row['lag_ns']:g}ns rmsd {row['ca_rmsd_micro']:.3f}"
            f"/{row['ca_rmsd_identity_micro']:.3f} base"
            f" rot {row['rotation_geodesic_deg_micro']:.1f}"
            f"/{row['rotation_geodesic_deg_identity_micro']:.1f}"
        )
    return " ".join(parts)
