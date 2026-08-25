"""Phase 1 training loop.

Three properties this module is responsible for, all of which are easy to get
subtly wrong and invisible afterwards:

**The split is by domain, and it happens before any frame is enumerated.**
Consecutive mdCATH frames of one protein are highly correlated, so a frame-level
split measures memorisation, not generalisation.

**The normaliser is fitted on the training split only.** Fitting on everything
leaks validation statistics into the objective.

**Seeds and the full config are stored in the checkpoint.** A checkpoint whose
config is not recoverable cannot be reloaded into a model that produces the same
output contract, which is a Phase 1 completion requirement.

Data-parallel training is supported through :class:`DistributedDataParallel`, but
only as an execution detail: ``distributed=True`` changes throughput and nothing
about the objective. Two things are *not* free and are handled explicitly here --
the normaliser must be identical on every rank (otherwise each rank optimises a
differently-scaled loss and the averaged gradient is meaningless), and evaluation
runs on the main rank alone behind a barrier, so validation metrics are computed
once over one loader rather than silently averaged over disjoint shards.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.adapters.mdcath import TrainingExample
from ..data.collate import collate_batches
from ..data.contracts import HierarchicalProteinBatch
from ..geometry.frames import frames_from_batch, link_backbone_to_atom_positions
from ..models.local_physics import LocalPhysicsConfig, LocalPhysicsModel
from ..physics.losses import LossWeights, TargetNormalizer, phase1_loss
from ..physics.projection import ResidueSumProjector
from .metrics import merge_metrics, vector_metrics

__all__ = ["TrainConfig", "collate_examples", "Phase1Trainer", "set_seed"]


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_examples(
    examples: Sequence[TrainingExample],
) -> tuple[HierarchicalProteinBatch, Optional[Tensor]]:
    """Collate dataset items into a batch and its hidden-force target."""
    batch = collate_batches([e.batch for e in examples], validate=False)
    hidden = [e.hidden_force_target for e in examples]
    if any(h is None for h in hidden):
        return batch, None
    return batch, torch.cat(hidden, dim=0)


@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters.

    Args:
        seed: seeds python/numpy/torch. Recorded in the checkpoint.
        max_steps / eval_every: a mini-subset run is step-bounded, not
            epoch-bounded, because trajectory counts vary wildly by domain.
        grad_clip: max global gradient norm. Force NLL can spike early when the
            predicted variance is still far from the error scale.
        normalizer_batches: how many training batches to fit the normaliser on.
        warmup_steps / lr_schedule / min_lr_factor: learning-rate shape. Warmup
            matters here for a specific reason: the heads emit a log-variance and
            the NLL divides the squared error by ``exp(logvar)``, so a large first
            step can drive the variance to a value that takes thousands of steps
            to recover from. ``lr_schedule='cosine'`` decays to
            ``min_lr_factor * learning_rate`` at ``max_steps``.
        eval_batches: cap on validation batches per evaluation. Under DDP the
            non-main ranks wait at a barrier while rank 0 evaluates, so an
            unbounded evaluation is an unbounded stall.
        checkpoint_every: periodic checkpoint interval in steps; 0 disables.
        max_consecutive_skips: abort after this many non-finite steps in a row.
            One bad frame in a corpus of millions should cost one batch, not the
            run -- but a *sustained* run of them is not bad data, it is a broken
            model, and continuing would quietly train on nothing.
        nonfinite_dump_dir / max_nonfinite_dumps: write the weights and the batch
            to disk the moment a step goes non-finite. Without this the evidence
            is gone: the offending batch replays clean through every neighbouring
            checkpoint, so the failure needs the weights from that exact step, and
            those are never the ones a periodic checkpoint happens to hold.
    """

    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 200
    eval_every: int = 50
    grad_clip: float = 10.0
    normalizer_batches: int = 8
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 10
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    min_lr_factor: float = 0.1
    eval_batches: Optional[int] = None
    checkpoint_every: int = 0
    max_consecutive_skips: int = 20
    nonfinite_dump_dir: Optional[str] = None
    max_nonfinite_dumps: int = 3


class Phase1Trainer:
    """Trains a :class:`LocalPhysicsModel` against mdCATH force labels."""

    def __init__(
        self,
        model: LocalPhysicsModel,
        config: TrainConfig = TrainConfig(),
        normalizer: Optional[TargetNormalizer] = None,
        *,
        distributed: bool = False,
    ):
        set_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        #: The unwrapped model. Everything that reads configuration, saves state
        #: or evaluates goes through this; only the training forward goes through
        #: the DDP wrapper, because DDP's forward is what installs the gradient
        #: hooks and it must be paired one-to-one with a backward.
        self.module = model.to(self.device)
        self.distributed = distributed
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )
        else:
            self.model = self.module
        self.normalizer = normalizer or TargetNormalizer()
        self.projector = ResidueSumProjector(model.config.target_scope)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step = 0
        self.history: list[dict] = []
        #: Batches whose loss or gradient was non-finite and which were therefore
        #: not applied. mdCATH has known upstream defects (see the adapter
        #: docstring); the audits catch what can be enumerated ahead of time, and
        #: this catches what cannot.
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self.skipped_domains: list[tuple[int, tuple]] = []

    @property
    def is_main(self) -> bool:
        """True on rank 0, and always true outside distributed training."""
        if not self.distributed:
            return True
        import torch.distributed as dist  # noqa: PLC0415

        return dist.get_rank() == 0

    def _barrier(self) -> None:
        if self.distributed:
            import torch.distributed as dist  # noqa: PLC0415

            dist.barrier()

    # -- learning rate -----------------------------------------------------

    def _set_lr(self) -> float:
        """Apply the warmup/decay schedule for the step about to be taken."""
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

    # -- normalisation -----------------------------------------------------

    def fit_normalizer(self, loader: Iterable) -> TargetNormalizer:
        """Fit target scales on the **training** loader only."""
        atom_forces, residue_forces, torques = [], [], []
        for i, (batch, _) in enumerate(loader):
            if i >= self.config.normalizer_batches:
                break
            targets = self.projector(batch)
            valid = targets.valid
            if batch.atoms.forces is not None:
                sel = batch.atoms.is_heavy
                if batch.atoms.force_valid is not None:
                    sel = sel & batch.atoms.force_valid
                atom_forces.append(batch.atoms.forces[sel])
            residue_forces.append(targets.force[valid])
            torques.append(targets.torque[valid])
        if not residue_forces:
            raise ValueError("no batches available to fit the normaliser")
        self.normalizer = TargetNormalizer.fit(
            torch.cat(atom_forces) if atom_forces else torch.cat(residue_forces),
            torch.cat(residue_forces),
            torch.cat(torques),
        )
        if self.distributed:
            self.normalizer = self._broadcast_normalizer(self.normalizer)
        return self.normalizer

    def _broadcast_normalizer(self, normalizer: TargetNormalizer) -> TargetNormalizer:
        """Force rank 0's scales onto every rank.

        Each rank fits on its own shard, so the three numbers differ slightly.
        Left alone that means every rank divides the loss by a different constant,
        and the averaged gradient no longer descends any single objective. The
        difference is small enough to be invisible in the loss curve, which is
        exactly why it is broadcast rather than trusted.
        """
        import torch.distributed as dist  # noqa: PLC0415

        values = torch.tensor(
            [normalizer.atom_force, normalizer.residue_force, normalizer.residue_torque],
            dtype=torch.float64, device=self.device,
        )
        dist.broadcast(values, src=0)
        return TargetNormalizer(*(float(v) for v in values))

    # -- one step ----------------------------------------------------------

    def _loss_for(
        self,
        batch: HierarchicalProteinBatch,
        hidden_target: Optional[Tensor],
        *,
        model: Optional[torch.nn.Module] = None,
    ) -> tuple[Tensor, dict, object]:
        batch = batch.to(self.device)
        if hidden_target is not None:
            hidden_target = hidden_target.to(self.device)
        output = (model if model is not None else self.model)(batch)
        linked = link_backbone_to_atom_positions(batch)
        frames = frames_from_batch(linked)
        targets = self.projector(batch)
        total, components = phase1_loss(
            output, batch, targets, frames,
            hidden_force_target=hidden_target,
            atom_selection=self.projector.atom_selection(batch),
            weights=self.config.loss_weights,
            normalizer=self.normalizer,
        )
        return total, components, (output, targets)

    def train_step(
        self, batch: HierarchicalProteinBatch, hidden_target: Optional[Tensor]
    ) -> dict:
        self.model.train()
        lr = self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)
        total, components, _ = self._loss_for(batch, hidden_target)

        # backward always runs, even for a doomed batch: DDP installs its
        # gradient hooks during forward and expects exactly one backward per
        # forward. Only the optimiser step is withheld, so nothing non-finite
        # ever reaches the weights or the Adam moments -- the next
        # zero_grad(set_to_none=True) discards the poisoned gradients.
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), self.config.grad_clip
        )

        local_bad = not (math.isfinite(components["total"]) and bool(torch.isfinite(grad_norm)))
        if self._any_rank(local_bad):
            self.skipped_steps += 1
            self.consecutive_skips += 1
            if local_bad and len(self.skipped_domains) < 64:
                diagnosis = self._diagnose_nonfinite(components, grad_norm)
                self.skipped_domains.append((self.step, batch.domain_id, diagnosis))
                self._dump_nonfinite(batch, hidden_target, components, diagnosis)
            if self.consecutive_skips > self.config.max_consecutive_skips:
                raise RuntimeError(
                    f"{self.consecutive_skips} consecutive non-finite steps at step "
                    f"{self.step}. That is no longer a bad frame -- the model or the "
                    f"learning rate is diverging. Offending batches so far: "
                    f"{self.skipped_domains[-3:]}"
                )
        else:
            self.optimizer.step()
            self.consecutive_skips = 0

        self.step += 1
        components["grad_norm"] = float(grad_norm)
        components["lr"] = lr
        components["skipped"] = float(self.skipped_steps)
        return components

    def _dump_nonfinite(
        self,
        batch: HierarchicalProteinBatch,
        hidden_target: Optional[Tensor],
        components: dict,
        diagnosis: dict,
    ) -> None:
        """Freeze everything needed to reproduce this step offline.

        Only the rank that saw the failure writes, and only for the first few, so
        a systematic failure does not fill the disk. The weights are the point:
        this batch is reproducible across runs and learning rates but replays
        clean through the checkpoints on either side of it, so the state at *this*
        step is the one piece of evidence that has been missing.
        """
        directory = self.config.nonfinite_dump_dir
        if not directory or self.skipped_steps > self.config.max_nonfinite_dumps:
            return
        os.makedirs(directory, exist_ok=True)
        rank = int(os.environ.get("RANK", 0))
        path = os.path.join(directory, f"nonfinite_step{self.step}_rank{rank}.pt")
        torch.save(
            {
                "step": self.step,
                "rank": rank,
                "state_dict": {k: v.detach().cpu() for k, v in self.module.state_dict().items()},
                "grads": {k: p.grad.detach().cpu()
                          for k, p in self.module.named_parameters() if p.grad is not None},
                "batch": batch.to("cpu"),
                "hidden_target": None if hidden_target is None else hidden_target.detach().cpu(),
                "components": components,
                "diagnosis": diagnosis,
                "normalizer": self.normalizer,
                "model_config": self.module.config,
                "train_config": self.config,
            },
            path,
        )

    def _diagnose_nonfinite(self, components: dict, grad_norm: Tensor) -> dict:
        """Say *what* went wrong, not just that something did.

        NaN and Inf have different causes -- Inf is overflow, NaN is 0/0, inf-inf
        or inf*0 -- and a bad loss and a bad gradient point at different halves of
        the model. Reporting "non-finite" for all four cases throws away the one
        piece of evidence that would distinguish them, which is exactly what
        happened the first time this fired.
        """

        def kind(value: float) -> str:
            if value != value:
                return "nan"
            return "inf" if math.isinf(value) else "finite"

        report = {
            "loss": {k: kind(v) for k, v in components.items() if kind(v) != "finite"},
            "grad_norm": kind(float(grad_norm)),
        }
        # First parameter whose gradient is bad, in module order. With the loss
        # finite this localises the failure to a specific block.
        for name, param in self.module.named_parameters():
            if param.grad is None:
                continue
            g = param.grad
            n_nan, n_inf = int(torch.isnan(g).sum()), int(torch.isinf(g).sum())
            if n_nan or n_inf:
                report["first_bad_grad"] = {
                    "param": name, "nan": n_nan, "inf": n_inf, "numel": g.numel(),
                }
                break
        return report

    def _any_rank(self, flag: bool) -> bool:
        """True if *any* rank saw the condition.

        The skip decision has to be collective. If one rank withheld its
        optimiser step while the others took theirs, the replicas would silently
        stop being copies of each other, and DDP would keep averaging gradients
        across models that no longer agree.
        """
        if not self.distributed:
            return flag
        import torch.distributed as dist  # noqa: PLC0415

        signal = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(signal, op=dist.ReduceOp.SUM)
        return bool(signal.item() > 0)

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def _metrics_for(self, output, batch, targets, hidden_target) -> dict:
        rows = {}
        if batch.atoms.forces is not None:
            sel = self.projector.atom_selection(batch)
            if batch.atoms.force_valid is not None:
                sel = sel & batch.atoms.force_valid
            rows.update(vector_metrics(
                output.atom_force_mean, batch.atoms.forces, sel, prefix="atom_force_"
            ))
        valid = targets.valid & batch.residues.mask
        rows.update(vector_metrics(
            output.residue_force_mean, targets.force, valid, prefix="residue_force_"
        ))
        rows.update(vector_metrics(
            output.residue_torque_mean, targets.torque, valid, prefix="torque_"
        ))
        if output.residue_hidden_force is not None and hidden_target is not None:
            rows.update(vector_metrics(
                output.residue_hidden_force, hidden_target.to(output.residue_hidden_force.device),
                valid, prefix="hidden_force_",
            ))
        return rows

    def evaluate(self, loader: Iterable, max_batches: Optional[int] = None) -> dict:
        """Loss components and metrics over a loader. Never backpropagated.

        Runs on the unwrapped module: a DDP forward with no matching backward
        leaves the reducer expecting a reduction that never arrives.
        """
        if max_batches is None:
            max_batches = self.config.eval_batches
        self.module.eval()
        losses, metrics = [], []
        for i, (batch, hidden) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            total, components, (output, targets) = self._loss_for(
                batch, hidden, model=self.module
            )
            losses.append({k: v for k, v in components.items()})
            metrics.append(self._metrics_for(output, batch.to(self.device), targets, hidden))
        if not losses:
            return {}
        out = {k: float(np.mean([row[k] for row in losses])) for k in losses[0]}
        out.update(merge_metrics(metrics))
        return out

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        *,
        log: Optional[callable] = print,
        checkpoint_path: Optional[str] = None,
    ) -> list[dict]:
        """Run to ``max_steps``, evaluating and checkpointing periodically.

        Args:
            checkpoint_path: written every ``config.checkpoint_every`` steps and
                once at the end, on the main rank only.
        """
        start = time.time()
        main = self.is_main
        if not main:
            log = None
        epoch = 0
        self._set_epoch(train_loader, epoch)
        train_iter = iter(train_loader)
        while self.step < self.config.max_steps:
            try:
                batch, hidden = next(train_iter)
            except StopIteration:
                # A new epoch needs a new sampler seed, otherwise every epoch
                # replays the same shard order on every rank.
                epoch += 1
                self._set_epoch(train_loader, epoch)
                train_iter = iter(train_loader)
                batch, hidden = next(train_iter)
            before = self.skipped_steps
            components = self.train_step(batch, hidden)
            if log and self.skipped_steps > before:
                detail = self.skipped_domains[-1][2] if self.skipped_domains else {}
                # Rank 0's batch and rank 0's diagnosis. Under DDP the offending
                # data is often on another rank, and the gradient all-reduce
                # copies its NaN here -- so a finite loss beside a NaN gradient
                # means "not this rank", not "not the data". The per-rank dumps
                # in nonfinite_dump_dir are what identify the culprit.
                log(f"  !! step {self.step}: batch skipped ({self.skipped_steps} total) "
                    f"| rank0 {detail} | rank0 domains {batch.domain_id}")
            if log and self.step % self.config.log_every == 0:
                log(
                    f"step {self.step:6d} | total={components['total']:.4f} "
                    f"| atom_nll={components['atom_force_nll']:.4f} "
                    f"| res_nll={components['residue_force_nll']:.4f} "
                    f"| grad={components['grad_norm']:.2f} "
                    f"| lr={components['lr']:.2e} | ep{epoch} | {time.time()-start:.0f}s"
                )
            if val_loader is not None and self.step % self.config.eval_every == 0:
                if main:
                    val = self.evaluate(val_loader)
                    self.history.append({"step": self.step, **val})
                    if log:
                        log(
                            f"  val {self.step}: total={val.get('total', float('nan')):.4f} "
                            f"| atom {val.get('atom_force_relative_rmse', float('nan')):.3f}"
                            f"/{val.get('atom_force_angular_error_deg', float('nan')):.1f}deg "
                            f"| res {val.get('residue_force_relative_rmse', float('nan')):.3f} "
                            f"| tau {val.get('torque_relative_rmse', float('nan')):.3f}"
                            f"/{val.get('torque_angular_error_deg', float('nan')):.1f}deg"
                        )
                self._barrier()
            if (
                checkpoint_path
                and self.config.checkpoint_every
                and self.step % self.config.checkpoint_every == 0
            ):
                if main:
                    self.save_checkpoint(checkpoint_path)
                self._barrier()
        if checkpoint_path and main:
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
        """Store weights, optimiser state, normaliser, seed and both configs.

        Written to a temporary file and renamed, because a periodic checkpoint
        that is interrupted mid-write leaves a truncated file where the only
        recoverable state used to be.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # Never let a poisoned model overwrite the last good one. A periodic
        # checkpoint is the only thing standing between a NaN and losing hours of
        # training, and it is worthless if it faithfully persists the NaN.
        nonfinite = [k for k, v in self.module.state_dict().items()
                     if v.is_floating_point() and not torch.isfinite(v).all()]
        if nonfinite:
            raise RuntimeError(
                f"refusing to checkpoint at step {self.step}: non-finite weights in "
                f"{nonfinite[:5]}. The previous checkpoint at {path} is kept."
            )
        tmp = f"{path}.tmp"
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_config": self.module.config,
                "train_config": self.config,
                "normalizer": self.normalizer,
                "step": self.step,
                "history": self.history,
                "skipped_steps": self.skipped_steps,
                "skipped_domains": self.skipped_domains,
                "latent_contract": self.module.latent_contract(),
            },
            tmp,
        )
        os.replace(tmp, path)

    def load_state(self, path: str) -> None:
        """Restore weights, optimiser, normaliser and step count in place."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.module.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.normalizer = payload["normalizer"]
        self.step = payload["step"]
        self.history = payload["history"]
        self.skipped_steps = payload.get("skipped_steps", 0)
        self.skipped_domains = payload.get("skipped_domains", [])

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> tuple[LocalPhysicsModel, "Phase1Trainer"]:
        """Rebuild the model and trainer exactly as saved."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = LocalPhysicsModel(payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        train_config = dataclasses.replace(payload["train_config"], device=device)
        trainer = Phase1Trainer(model, train_config, payload["normalizer"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.step = payload["step"]
        trainer.history = payload["history"]
        return model, trainer

    def config_snapshot(self) -> dict:
        """JSON-serialisable record of everything that defines this run."""
        return {
            "train": {k: (v if not dataclasses.is_dataclass(v) else asdict(v))
                      for k, v in asdict(self.config).items()},
            "model": {
                "target_scope": self.module.config.target_scope,
                "predict_hidden_force": self.module.predict_hidden_force,
                "use_energy_branch": self.module.config.use_energy_branch,
                "num_cycles": self.module.config.encoder.num_cycles,
                "lmax": self.module.config.encoder.irreps.lmax,
                "atom_cutoff": self.module.config.graph.atom_cutoff,
                "residue_knn": self.module.config.graph.residue_knn,
            },
            "distributed": {
                "enabled": self.distributed,
                "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "normalizer": self.normalizer.as_dict(),
            "latent_contract": self.module.latent_contract(),
        }
