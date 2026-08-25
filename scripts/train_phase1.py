#!/usr/bin/env python
"""Train the Phase 1 local-physics model on an mdCATH subset.

    python scripts/train_phase1.py --config configs/phase1_small.yaml
    python scripts/train_phase1.py --config configs/phase1_small.yaml --dry-run

Multi-GPU, four cards, physical GPUs 4-7::

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        scripts/train_phase1.py --config configs/phase1_full.yaml

``batch_size`` is **per rank**, so the effective batch is ``batch_size *
world_size``; the config comments record which of the two a number refers to.

The split is by **domain** and is applied before any frame is enumerated, so no
protein appears on both sides. The target normaliser is fitted on the training
split only. Both configs, the seed and the normaliser are written into the
checkpoint so the run is reproducible and reloadable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_md.data.adapters.mdcath import (  # noqa: E402
    MdCathConfig,
    MdCathDataset,
    split_domains,
)
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.graph.hierarchy import GraphConfig  # noqa: E402
from force_md.physics.losses import LossWeights  # noqa: E402
from force_md.training.phase1_module import (  # noqa: E402
    Phase1Trainer,
    TrainConfig,
    collate_examples,
    set_seed,
)


def _tuple_or_none(value):
    return None if value is None else tuple(value)


def setup_distributed() -> tuple[int, int, int, bool]:
    """Join the torchrun process group, if there is one.

    Returns ``(rank, world_size, local_rank, distributed)``. Run without
    ``torchrun`` this is a no-op and the script trains on a single device, so the
    same file serves both paths and there is no second, drifting copy.

    ``local_rank`` indexes *visible* devices. With ``CUDA_VISIBLE_DEVICES=4,5,6,7``
    local rank 0 is physical GPU 4; nothing here should ever name a physical index.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0, False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # device_id binds the process group to this rank's device, so a collective
    # cannot be issued against whatever device happened to be current.
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    return dist.get_rank(), dist.get_world_size(), local_rank, True


def build_configs(raw: dict) -> tuple[MdCathConfig, LocalPhysicsConfig, TrainConfig, dict]:
    d, m, t = raw["data"], raw["model"], raw["train"]
    data_cfg = MdCathConfig(
        data_dir=d["data_dir"],
        esm2_cache_dir=d.get("esm2_cache_dir"),
        quarantine_path=d.get("quarantine_path"),
        coord_quarantine_path=d.get("coord_quarantine_path"),
        represented_scope=d.get("represented_scope", "heavy_atom"),
        temperatures=_tuple_or_none(d.get("temperatures")),
        replicas=_tuple_or_none(d.get("replicas")),
        frames_per_trajectory=d.get("frames_per_trajectory", 4),
        ps_per_frame=d.get("ps_per_frame"),
        max_domains=d.get("max_domains"),
        max_residues=d.get("max_residues"),
        allow_fake_plm=d.get("allow_fake_plm", False),
        check_pbc=d.get("check_pbc", True),
    )
    enc = m["encoder"]
    model_cfg = LocalPhysicsConfig(
        encoder=EncoderConfig(
            irreps=IrrepsConfig(**enc["irreps"]),
            **{k: v for k, v in enc.items() if k != "irreps"},
        ),
        graph=GraphConfig(**m["graph"]),
        target_scope=m.get("target_scope", "heavy_atom"),
        predict_hidden_force=m.get("predict_hidden_force"),
        use_energy_branch=m.get("use_energy_branch", True),
        isotropic_uncertainty=m.get("isotropic_uncertainty", False),
        validate_inputs=m.get("validate_inputs", True),
    )
    train_cfg = TrainConfig(
        loss_weights=LossWeights(**t.pop("loss_weights", {})), **t
    )
    if train_cfg.nonfinite_dump_dir is None:
        train_cfg = dataclasses.replace(train_cfg, nonfinite_dump_dir="__OUT_DIR__")
    return data_cfg, model_cfg, train_cfg, d


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(root / "configs" / "phase1_small.yaml"))
    ap.add_argument("--out-dir", default=str(root / "runs" / "phase1_small"))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to restore weights, optimiser and step from.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build everything and report sizes without training.")
    args = ap.parse_args()

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    raw = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, train_cfg, data_raw = build_configs(raw)
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.device is not None:
        overrides["device"] = args.device
    elif distributed:
        # local_rank indexes visible devices, so this is correct under any
        # CUDA_VISIBLE_DEVICES; the config's own `device` is ignored here.
        overrides["device"] = f"cuda:{local_rank}"
    if train_cfg.nonfinite_dump_dir == "__OUT_DIR__":
        overrides["nonfinite_dump_dir"] = os.path.join(args.out_dir, "nonfinite")
    if overrides:
        train_cfg = type(train_cfg)(**{**train_cfg.__dict__, **overrides})

    # Every rank must draw the same split and the same initial weights.
    set_seed(train_cfg.seed)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)

    # ---- data: split by domain BEFORE enumerating frames -----------------
    probe = MdCathDataset(data_cfg)
    train_domains, val_domains = split_domains(
        probe.domains, data_raw.get("val_fraction", 0.2), data_raw.get("split_seed", 0)
    )
    overlap = set(train_domains) & set(val_domains)
    assert not overlap, f"domain leak between splits: {overlap}"
    log(f"domains: {len(probe.domains)} total -> {len(train_domains)} train / "
        f"{len(val_domains)} val (split by domain, seed "
        f"{data_raw.get('split_seed', 0)})")

    train_ds = MdCathDataset(data_cfg, domains=train_domains)
    val_ds = MdCathDataset(data_cfg, domains=val_domains)
    log(f"frames: {len(train_ds)} train / {len(val_ds)} val")

    bs = data_raw.get("batch_size", 2)
    workers = data_raw.get("num_workers", 0)
    # Measured on this machine: num_workers 0 -> 20 batch/s, 8 -> 255 batch/s.
    # Frame loading is an h5 read plus a topology lookup and is nowhere near the
    # GPU's speed once it is off the main process.
    loader_kw = dict(collate_fn=collate_examples, num_workers=workers,
                     persistent_workers=workers > 0)
    if workers > 0:
        loader_kw["prefetch_factor"] = data_raw.get("prefetch_factor", 4)
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                           shuffle=True, drop_last=True, seed=train_cfg.seed)
        if distributed else None
    )
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                              shuffle=train_sampler is None, **loader_kw)
    # Validation is evaluated by rank 0 alone over the whole loader, so it is not
    # sharded: a sharded metric averaged across ranks is not the metric.
    #
    # `val_frames` subsamples with an even stride rather than capping the number
    # of batches. The index is ordered by (domain, temperature, replica, frame),
    # so truncating an unshuffled loader would evaluate the first two domains and
    # call it validation; a stride spans every domain, and being fixed it makes
    # successive evaluations comparable.
    val_eval = val_ds
    val_frames = data_raw.get("val_frames")
    if val_frames and val_frames < len(val_ds):
        picks = (
            torch.linspace(0, len(val_ds) - 1, int(val_frames))
            .round().long().unique().tolist()
        )
        val_eval = torch.utils.data.Subset(val_ds, picks)
        log(f"validation subsampled: {len(val_eval)} of {len(val_ds)} frames "
            f"(even stride over all {len(val_domains)} val domains)")
    val_loader = DataLoader(val_eval, batch_size=bs, shuffle=False, **loader_kw)

    model = LocalPhysicsModel(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {n_params:,} parameters | irreps {model.irreps} | "
        f"hidden residual {model.predict_hidden_force} | "
        f"energy branch {model_cfg.use_energy_branch}")
    log(f"world size {world_size} | per-rank batch {bs} | "
        f"effective batch {bs * world_size} | device {train_cfg.device}")

    trainer = Phase1Trainer(model, train_cfg, distributed=distributed)
    normalizer = trainer.fit_normalizer(train_loader)
    log(f"normaliser (train split only, rank 0): atom={normalizer.atom_force:.2f} "
        f"residue={normalizer.residue_force:.2f} torque={normalizer.residue_torque:.2f}")

    if args.resume:
        trainer.load_state(args.resume)
        log(f"resumed from {args.resume} at step {trainer.step}")

    if is_main:
        snapshot = trainer.config_snapshot()
        snapshot["split"] = {"train_domains": train_domains, "val_domains": val_domains}
        Path(args.out_dir, "config_snapshot.json").write_text(json.dumps(snapshot, indent=1))

    if args.dry_run:
        log("dry-run: nothing trained")
        if distributed:
            dist.destroy_process_group()
        return 0

    ckpt = os.path.join(args.out_dir, "last.pt")
    log(f"\ntraining to step {train_cfg.max_steps} on {train_cfg.device}")
    history = trainer.fit(train_loader, val_loader, log=print if is_main else None,
                          checkpoint_path=ckpt)
    if is_main:
        Path(args.out_dir, "history.json").write_text(json.dumps(history, indent=1))

    if history and is_main:
        final = history[-1]
        print("\nfinal validation:")
        for key in ("total", "atom_force_rmse", "atom_force_rmse_zero",
                    "atom_force_relative_rmse", "atom_force_angular_error_deg",
                    "residue_force_rmse", "residue_force_relative_rmse",
                    "residue_force_angular_error_deg",
                    "torque_rmse", "torque_relative_rmse"):
            if key in final:
                print(f"  {key:34s} {final[key]:10.4f}")
        print("\nRMSE is only meaningful against rmse_zero (predicting no force);")
        print("a relative_rmse near 1.0 means the model has learned nothing.")
    log(f"\ncheckpoint -> {ckpt}")
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
