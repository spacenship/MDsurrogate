#!/usr/bin/env python
"""Train Stage 3 by rotating through fixed-size mdCATH domain chunks.

The unit of rotation is a domain, not a frame.  The default is 500 domains
per chunk.  A run first builds one global, leakage-safe frame split and one
normalizer from the training portion, then trains chunk 000, chunk 001, ...
without retaining previous chunks in memory. Periodic atomic checkpoints
contain the complete Stage 3 upstream state, optimizer, global normalizer,
per-rank RNG states and exact next step/chunk. torchrun enables DDP.
The normalizer is also saved immediately as a standalone JSON artifact, so a
fresh invocation with the same data/split configuration can reuse it.

This runner is intentionally separate from ``train_physics.py``: the latter
remains the small bounded smoke entry point used by the existing tests.

Example (the esm3 environment, physical GPUs 6--7 visible; cuda:0 is physical
GPU 6 inside this process):

    CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 \
      /home/ubuntu/miniforge3/envs/esm3/bin/python \
      experiments/heavy_flow/train_physics_chunks.py \
      --data-dir data --esm2-cache-dir esm2_cache \
      --quarantine-path mdcath_force_quarantine.json \
      --coord-quarantine-path mdcath_coord_quarantine.json \
      --device cuda:0

An interrupted v2 chunk resumes after the latest optimizer update. Legacy
architectures require explicit --warm-start and new Stage 3 training. No data
files are replaced; fitting a missing normalizer requires --fit-normalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
import numpy as np

from experiments.heavy_flow.physics_runtime import capture_rng, rank_world, rank_zero_call, restore_rng

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.data.units import MDCATH_UNITS
from force_md.heavy_flow import (
    ForceLossConfig,
    ForceNormalizer,
    HeavyFlowPhysicsFrameDataset,
    PhysicsSplitManifest,
    build_domain_chunks,
    domain_order,
)

from experiments.heavy_flow.train_physics import (
    build_stage3_model,
    load_yaml,
    save_stage3_checkpoint,
    train_physics_stream_steps,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class _ChunkProgress:
    """Log completed optimizer steps and a recent wall-clock throughput ETA."""

    def __init__(self, *, chunk_index: int, steps: int, steps_per_epoch: int,
                 step_offset: int, global_step_target: int,
                 log_every: int, log_seconds: float, start_step: int = 0):
        self.chunk_index = chunk_index
        self.steps = steps
        self.steps_per_epoch = steps_per_epoch
        self.step_offset = step_offset
        self.global_step_target = global_step_target
        self.log_every = log_every
        self.log_seconds = log_seconds
        self.started = self.last_time = time.perf_counter()
        self.last_step = start_step
        self.first_step = start_step + 1

    def __call__(self, completed: int, loss: float) -> None:
        now = time.perf_counter()
        if (completed != self.first_step and completed != self.steps
                and completed % self.log_every != 0
                and now - self.last_time < self.log_seconds):
            return
        seconds_per_step = (now - self.last_time) / (completed - self.last_step)
        global_step = self.step_offset + completed
        print(json.dumps({
            "event": "train_progress",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_index": self.chunk_index,
            "chunk_step": completed,
            "chunk_steps": self.steps,
            "epoch": (completed - 1) // self.steps_per_epoch + 1,
            "global_step": global_step,
            "global_step_target": self.global_step_target,
            "chunk_progress_percent": round(100 * completed / self.steps, 2),
            "progress_percent": round(100 * global_step / self.global_step_target, 2),
            "loss": loss,
            "chunk_elapsed_seconds": round(now - self.started, 3),
            "seconds_per_step": round(seconds_per_step, 4),
            "chunk_eta_seconds": round(seconds_per_step * (self.steps - completed), 1),
            "eta_seconds": round(seconds_per_step * (self.global_step_target - global_step), 1),
            "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,
            "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None,
        }), flush=True)
        self.last_time = now
        self.last_step = completed


def _default_manifest_path(data_dir: str | Path) -> Optional[Path]:
    root = Path(data_dir)
    for candidate in (root / "mdcath_manifest.json", root.parent / "mdcath_manifest.json"):
        if candidate.exists():
            return candidate
    return None


def _reader(
    args: argparse.Namespace,
    config: dict[str, Any],
    domains: Iterable[str],
) -> MdCathDataset:
    data_config = dict(config.get("data", {}))
    return MdCathDataset(
        MdCathConfig(
            data_dir=args.data_dir,
            represented_scope=str(data_config.get("represented_scope", "heavy_atom")),
            frames_per_trajectory=args.frames_per_trajectory,
            esm2_cache_dir=args.esm2_cache_dir,
            allow_fake_plm=args.allow_fake_plm,
            load_plm=not bool(getattr(args, "corpus_manifest", None)),
            quarantine_path=args.quarantine_path,
            coord_quarantine_path=args.coord_quarantine_path,
        ),
        domains=list(domains),
    )


def _keys_for_chunk(
    split: PhysicsSplitManifest,
    chunk_domains: set[str],
    max_train_frames: Optional[int],
) -> list[Any]:
    keys = [key for key in split.train if key.domain in chunk_domains]
    if max_train_frames is not None:
        keys = keys[:max_train_frames]
    return keys


def _force_arrays_for_normalizer(
    args: argparse.Namespace,
    config: dict[str, Any],
    chunks: list[Any],
    split: PhysicsSplitManifest,
):
    """Yield represented forces and masks without building full samples."""
    for chunk in chunks:
        keys = _keys_for_chunk(split, set(chunk.domains), args.max_train_frames)
        if not keys:
            continue
        reader = _reader(args, config, chunk.domains)
        try:
            for key in keys:
                yield reader.load_force_arrays(
                    key.domain, key.temperature, key.replica, key.frame
                )
        finally:
            reader.close()


def _resolved_path(value: Optional[str | Path]) -> Optional[str]:
    return None if value is None else str(Path(value).resolve())


def _key_digest(keys: Iterable[Any]) -> str:
    encoded = _canonical([
        key.as_dict() if hasattr(key, "as_dict") else key for key in keys
    ]).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalizer_fit_keys(
    chunks: list[Any], split: PhysicsSplitManifest, max_train_frames: Optional[int]
) -> list[Any]:
    keys: list[Any] = []
    for chunk in chunks:
        keys.extend(
            _keys_for_chunk(split, set(chunk.domains), max_train_frames)
        )
    return keys


def _normalizer_fit_spec(
    args: argparse.Namespace,
    config: dict[str, Any],
    order: list[str],
    manifest_path: Optional[Path],
    chunks: list[Any],
    split: PhysicsSplitManifest,
) -> dict[str, Any]:
    """Describe exactly which train force rows the artifact represents."""
    fit_keys = _normalizer_fit_keys(chunks, split, args.max_train_frames)
    data_config = dict(config.get("data", {}))
    return {
        # Retain the full config as provenance; only its data section affects
        # force fitting and participates in cache compatibility below.
        "config": config,
        "data_dir": _resolved_path(args.data_dir),
        "esm2_cache_dir": _resolved_path(args.esm2_cache_dir),
        "quarantine_path": _resolved_path(args.quarantine_path),
        "coord_quarantine_path": _resolved_path(args.coord_quarantine_path),
        "represented_scope": str(data_config.get("represented_scope", "heavy_atom")),
        "force_unit": MDCATH_UNITS.force,
        "frames_per_trajectory": int(args.frames_per_trajectory),
        "max_train_frames_per_chunk": args.max_train_frames,
        "chunk_size": int(args.chunk_size),
        "domain_order_manifest": _resolved_path(manifest_path),
        "domain_order": order,
        "split": {
            "seed": int(split.seed),
            "temporal_block_size": int(split.temporal_block_size),
            "same_domain_validation_fraction": float(
                data_config["same_domain_validation_fraction"]
            ),
            "unseen_domain_validation_fraction": float(
                data_config["unseen_domain_validation_fraction"]
            ),
            "train_count": len(fit_keys),
            "train_key_sha256": _key_digest(fit_keys),
        },
    }


def _default_normalizer_path(output: Path) -> Path:
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    if stem.endswith("_latest"):
        stem = stem[: -len("_latest")]
    return output.with_name(f"{stem}_normalizer.json")


def _save_normalizer_artifact(
    path: Path, normalizer: ForceNormalizer, fit_spec: dict[str, Any]
) -> None:
    """Atomically persist the scalar and the data identity it was fit on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "stage3_force_normalizer",
        "schema_version": 1,
        "normalizer": normalizer.as_dict(),
        "fit_spec": fit_spec,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _normalizer_data_identity(fit_spec: Any) -> dict[str, Any]:
    """Exclude model/training settings from normalizer cache compatibility.

    Schema-v1 artifacts already contain the full model config. Project both
    old and current specs here so existing force statistics remain reusable
    after memory/performance tuning, without rewriting the original artifact.
    All other fields, including train-key digest and data config, stay strict.
    """
    if not isinstance(fit_spec, dict):
        raise ValueError("invalid normalizer fit_spec: expected a mapping")
    identity = dict(fit_spec)
    if "config" in identity:
        config = identity["config"]
        if not isinstance(config, dict):
            raise ValueError("invalid normalizer fit_spec config: expected a mapping")
        identity["config"] = {"data": config.get("data", {})}
    return identity


def _load_normalizer_artifact(
    path: Path, fit_spec: dict[str, Any]
) -> ForceNormalizer:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read normalizer artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("stage") != "stage3_force_normalizer":
        raise ValueError(f"invalid Stage 3 normalizer artifact: {path}")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported normalizer artifact schema: {path}")
    saved_identity = _normalizer_data_identity(payload.get("fit_spec"))
    current_identity = _normalizer_data_identity(fit_spec)
    if _canonical(saved_identity) != _canonical(current_identity):
        changed = sorted(
            key for key in saved_identity.keys() | current_identity.keys()
            if key not in saved_identity or key not in current_identity
            or _canonical(saved_identity[key]) != _canonical(current_identity[key])
        )
        raise ValueError(
            f"normalizer artifact {path} was fit for a different data/split "
            f"configuration (changed fields: {', '.join(changed)}); provide the "
            "matching data/split settings or a new --normalizer-path to fit separately"
        )
    normalizer_data = payload.get("normalizer")
    if not isinstance(normalizer_data, dict):
        raise ValueError(f"normalizer artifact has no normalizer payload: {path}")
    return ForceNormalizer.from_dict(normalizer_data)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Move Adam moments after loading a CPU-mapped checkpoint."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_save_stage3(
    path: Path,
    model: torch.nn.Module,
    *,
    normalizer: Any,
    split_manifest: PhysicsSplitManifest,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    step: int,
    chunk_rotation: dict[str, Any],
    metrics: dict[str, Any],
    audit: Optional[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_stage3_checkpoint(
        temporary,
        model,
        normalizer=normalizer,
        split_manifest=split_manifest,
        config=config,
        audit=audit,
        metrics=metrics,
        optimizer=optimizer,
        step=step,
        chunk_rotation=chunk_rotation,
    )
    temporary.replace(path)


def _chunk_checkpoint_path(output: Path, index: int) -> Path:
    suffix = output.suffix or ".pt"
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    if stem.endswith("_latest"):
        stem = stem[: -len("_latest")]
    return output.with_name(f"{stem}_chunk_{index:03d}{suffix}")


from force_md.heavy_flow.checkpoint import architecture_config, validate_checkpoint, load_normalizer_artifact, warm_start_stage3
from force_md.heavy_flow.history import HistoryConfig


def _load_resume(
    path: Path,
    config: dict[str, Any],
    *,
    device: str,
    lr: float,
) -> tuple[Any, torch.optim.Optimizer, Any, int, int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("stage") != "stage3_atom_physics":
        raise ValueError(f"resume checkpoint is not a Stage 3 checkpoint: {path}")
    validate_checkpoint(payload)
    rotation = payload.get("chunk_rotation")
    if not isinstance(rotation, dict):
        raise ValueError(
            "resume checkpoint has no chunk_rotation metadata; use a checkpoint "
            "written by train_physics_chunks.py"
        )
    checkpoint_config = payload.get("config")
    if _canonical(checkpoint_config) != _canonical(config):
        raise ValueError("resume checkpoint config differs from --config")
    model = build_stage3_model(config, device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.training_provenance = payload.get("provenance", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if "optimizer_state" not in payload:
        raise ValueError("chunk-rotation resume checkpoint has no optimizer_state")
    optimizer.load_state_dict(payload["optimizer_state"])
    _move_optimizer_state(optimizer, device)
    normalizer_data = payload.get("normalizer")
    if not isinstance(normalizer_data, dict):
        raise ValueError("resume checkpoint has no force normalizer")
    from force_md.heavy_flow.force_losses import ForceNormalizer

    normalizer = ForceNormalizer.from_dict(normalizer_data)
    next_chunk = int(rotation.get("next_chunk", -1))
    global_step = int(payload.get("step", 0))
    if next_chunk < 0 or global_step < 0:
        raise ValueError("resume checkpoint has invalid chunk or global step")
    return model, optimizer, normalizer, next_chunk, global_step, rotation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heavy_flow/stage3.yaml")
    parser.add_argument("--corpus-manifest")
    parser.add_argument("--corpus-chunk-index", type=int)
    parser.add_argument("--initialize-from", help="Carry completed previous corpus chunk weights, Adam and scale")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--domain-order-manifest", default=None,
                        help="Downloader manifest defining seeded domain order; auto-detected next to data-dir")
    parser.add_argument("--esm2-cache-dir", default=None,
                        help="Required by the mdCATH adapter; no fake embeddings are used unless explicitly opted in")
    parser.add_argument("--quarantine-path", default=None)
    parser.add_argument("--coord-quarantine-path", default=None)
    parser.add_argument("--allow-fake-plm", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Process at most this many chunks from --start-chunk")
    parser.add_argument("--frames-per-trajectory", type=int, default=4)
    parser.add_argument("--max-train-frames", type=int, default=None,
                        help="Explicit bounded smoke cap per chunk; omit for all train frames")
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    parser.add_argument("--steps-per-chunk", type=int, default=None,
                        help="Override full-epoch steps per chunk")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefetch", type=int, default=1,
                        help="CPU batch prefetch depth per rank; 0 disables, default 1")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save latest after N optimizer steps; 0 disables step trigger")
    parser.add_argument("--checkpoint-seconds", type=float, default=0,
                        help="Also save after N seconds at a completed step; default 0 disables timer")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log progress every N completed optimizer steps (also first/last)")
    parser.add_argument("--log-seconds", type=float, default=60.0,
                        help="Also log after this many seconds, at the next completed step")
    parser.add_argument("--output", default="outputs/heavy_flow/stage3_v2/chunk_rotation_latest.pt")
    parser.add_argument(
        "--normalizer-path",
        default=None,
        help=(
            "Standalone JSON cache for the train-split force normalizer; "
            "defaults to <output-stem>_normalizer.json"
        ),
    )
    parser.add_argument("--reuse-normalizer", default=None, help="reuse compatible existing RMS and original fit provenance without scanning data")
    parser.add_argument("--fit-normalizer", action="store_true", help="explicit opt-in to a full normalizer scan when no artifact exists")
    parser.add_argument("--warm-start", default=None, help="compatible weights only; no old optimizer; new Stage 3 training required")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if args.device != "cpu":
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        # A first force-only normalizer fit on rank zero can take hours.
        # torchrun still terminates peers promptly when any worker exits.
        dist.init_process_group(backend="gloo" if args.device == "cpu" else "nccl",
                                timeout=timedelta(hours=24))
    try:
        _run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(args: argparse.Namespace) -> None:
    rank, world = rank_world()
    if args.prefetch < 0 or args.checkpoint_every < 0 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds < 0:
        raise ValueError("prefetch and checkpoint intervals must be non-negative and finite")
    if args.log_every < 1 or not math.isfinite(args.log_seconds) or args.log_seconds <= 0:
        raise SystemExit("log-every and log-seconds must be positive and finite")
    if args.chunk_size < 1 or args.batch_size < 1 or args.epochs_per_chunk < 1:
        raise SystemExit("chunk-size, batch-size and epochs-per-chunk must be positive")
    if args.steps_per_chunk is not None and args.steps_per_chunk < 1:
        raise SystemExit("steps-per-chunk must be positive")
    if args.max_train_frames is not None and args.max_train_frames < 1:
        raise SystemExit("max-train-frames must be positive when supplied")
    if args.max_chunks is not None and args.max_chunks < 1:
        raise SystemExit("max-chunks must be positive when supplied")
    config = architecture_config(load_yaml(args.config))
    if getattr(args, "warm_start", None) and (args.resume or getattr(args, "initialize_from", None)):
        raise ValueError("warm-start is exclusive with resume/initialize-from")
    if config.get("stage") != "stage3_atom_physics":
        raise SystemExit("--config must describe stage3_atom_physics")
    seed = int(config.get("data", {}).get("seed", 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    manifest_path = Path(args.domain_order_manifest) if args.domain_order_manifest else _default_manifest_path(args.data_dir)
    order = domain_order(args.data_dir, manifest_path=manifest_path)
    chunks = build_domain_chunks(order, chunk_size=args.chunk_size)
    from force_md.heavy_flow.corpus_plan import load_plan, chunk_entries, fixed_frame_split, digest
    plan = load_plan(args.corpus_manifest) if getattr(args, "corpus_manifest", None) else None
    if plan:
        expected = [e['domain'] for e in chunk_entries(plan, args.corpus_chunk_index)]
        if order != expected or len(chunks) != 1 or args.chunk_size != plan['chunk_size']:
            raise ValueError("local shards/order must exactly match the selected corpus chunk")
    if getattr(args, "initialize_from", None) and (not plan or args.resume):
        raise ValueError("initialize-from requires corpus mode and is exclusive with resume")
    if args.start_chunk < 0 or args.start_chunk >= len(chunks):
        raise SystemExit(f"start-chunk must be in [0, {len(chunks) - 1}]")

    # Build the split once over all local domains.  A separate split per chunk
    # would change which domains are unseen and would make chunk checkpoints
    # incomparable.
    def make_split():
        index_reader = _reader(args, config, order)
        try:
            from force_md.heavy_flow.physics_dataset import build_physics_splits
            if plan:
                return fixed_frame_split(index_reader.index, plan, seed=plan['seed'])
            return build_physics_splits(
                index_reader,
                same_domain_fraction=float(config["data"]["same_domain_validation_fraction"]),
                unseen_domain_fraction=float(config["data"]["unseen_domain_validation_fraction"]),
                temporal_block_size=int(config["data"]["temporal_block_size"]),
                seed=int(config["data"]["seed"]),
            )
        finally:
            index_reader.close()
    split = rank_zero_call(make_split)

    output = Path(args.output)
    normalizer_path = (
        Path(args.normalizer_path)
        if args.normalizer_path
        else _default_normalizer_path(output)
    )
    normalizer_fit_spec = _normalizer_fit_spec(
        args, config, order, manifest_path, chunks, split
    )
    rotation_history: list[dict[str, Any]] = []
    resume_step = 0
    old_rotation: dict[str, Any] = {}
    if getattr(args, "initialize_from", None):
        model, optimizer, normalizer, previous_next, global_step, old_rotation = _load_resume(
            Path(args.initialize_from), config, device=args.device, lr=args.lr)
        if old_rotation.get('step_in_chunk', 0) or previous_next != old_rotation['total_chunks']:
            raise ValueError("initialize-from requires a completed checkpoint")
        if old_rotation.get('corpus_plan_digest'):
            if old_rotation['corpus_plan_digest'] != plan['digest'] or old_rotation['corpus_chunk_index'] + 1 != args.corpus_chunk_index:
                raise ValueError("previous checkpoint corpus identity/order mismatch")
        else:
            legacy = plan.get('legacy')
            payload = torch.load(args.initialize_from, map_location='cpu', weights_only=False, mmap=True)
            if not legacy or legacy['next_chunk'] != args.corpus_chunk_index or legacy['domain_order'] != old_rotation['domain_order'] or legacy['step'] != global_step or legacy['normalizer'] != payload['normalizer'] or legacy['split_digest'] != digest(payload['split_manifest']):
                raise ValueError("checkpoint does not match the imported legacy split")
        start_chunk = 0
        rank_zero_call(lambda: _save_normalizer_artifact(normalizer_path, normalizer, normalizer_fit_spec))
    elif args.resume:
        model, optimizer, normalizer, resume_next, global_step, old_rotation = _load_resume(
            Path(args.resume), config, device=args.device, lr=args.lr
        )
        if plan and (old_rotation.get('corpus_plan_digest') != plan['digest'] or old_rotation.get('corpus_chunk_index') != args.corpus_chunk_index):
            raise ValueError("resume corpus identity mismatch")
        if int(old_rotation.get("chunk_size", -1)) != args.chunk_size:
            raise ValueError("resume checkpoint chunk_size differs from --chunk-size")
        if old_rotation.get("domain_order") != order:
            raise ValueError("resume checkpoint domain order differs from current data/manifest")
        if int(old_rotation.get("total_chunks", -1)) != len(chunks):
            raise ValueError("resume checkpoint total chunk count differs from current data")
        if int(old_rotation.get("frames_per_trajectory", -1)) != args.frames_per_trajectory:
            raise ValueError(
                "resume checkpoint frames_per_trajectory differs from the current run"
            )
        if old_rotation.get("max_train_frames") != args.max_train_frames:
            raise ValueError(
                "resume checkpoint max_train_frames differs from the current run"
            )
        if "normalizer_fit_identity" in old_rotation and _canonical(old_rotation["normalizer_fit_identity"]) != _canonical(_normalizer_data_identity(normalizer_fit_spec)):
            raise ValueError("resume checkpoint data/split identity differs from current run")
        if args.start_chunk != 0 and args.start_chunk != resume_next:
            raise ValueError("when resuming, --start-chunk must be omitted or equal next_chunk")
        start_chunk = resume_next
        rotation_history = list(old_rotation.get("chunk_metrics", []))
        # The checkpoint is authoritative on resume.  Mirror its normalizer
        # into the standalone artifact so the next fresh invocation can reuse
        # it without scanning all train frames again.
        resume_step = int(old_rotation.get("step_in_chunk", 0))
        if resume_step < 0 or resume_step > global_step:
            raise ValueError("invalid resume step_in_chunk")
        if resume_step:
            expected = {"world_size": world, "batch_size": args.batch_size,
                        "epochs_per_chunk": args.epochs_per_chunk,
                        "steps_per_chunk": args.steps_per_chunk,
                        "train_key_sha256": _key_digest(split.train)}
            if old_rotation.get("execution") != expected:
                raise ValueError("mid-chunk resume requires the same world size, batch, schedule and train keys")
            if "rng_states" not in old_rotation:
                raise ValueError("mid-chunk resume missing per-rank RNG states")
            if old_rotation.get("partial_loss", {}).get("count") != resume_step:
                raise ValueError("mid-chunk resume loss statistics do not match saved step")
        rank_zero_call(lambda: _save_normalizer_artifact(normalizer_path, normalizer, normalizer_fit_spec))
        if start_chunk >= len(chunks):
            if rank == 0:
                print(json.dumps({"status": "already_complete", "next_chunk": start_chunk}, indent=2))
            return
    else:
        model = build_stage3_model(config, device=args.device)
        if getattr(args, "warm_start", None):
            report = warm_start_stage3(model, args.warm_start)
            if rank == 0:
                report_path = output.with_suffix(".warm_start.json")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"warm_start_report": str(report_path), "loaded": len(report["loaded"]), "not_loaded": len(report["not_loaded"]), "requires_stage3_force_training": True}), flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        def fit_or_load():
            if normalizer_path.exists():
                print(json.dumps({"event": "normalizer_load", "path": str(normalizer_path)}), flush=True)
                return _load_normalizer_artifact(normalizer_path, normalizer_fit_spec)
            if not getattr(args, "fit_normalizer", False):
                raise ValueError("no cached normalizer: use --reuse-normalizer or explicitly opt in with --fit-normalizer")
            print("fitting one global force normalizer over train frames (streaming) ...", flush=True)
            fitted = ForceNormalizer.fit_from_force_arrays(
                _force_arrays_for_normalizer(args, config, chunks, split), force_unit=MDCATH_UNITS.force)
            _save_normalizer_artifact(normalizer_path, fitted, normalizer_fit_spec)
            print(json.dumps({"event": "normalizer_saved", "path": str(normalizer_path)}), flush=True)
            return fitted
        if getattr(args, "reuse_normalizer", None):
            normalizer, source_provenance = rank_zero_call(lambda: load_normalizer_artifact(args.reuse_normalizer))
            model.training_provenance = {**getattr(model, "training_provenance", {}), "normalizer": source_provenance}
        else:
            normalizer = rank_zero_call(fit_or_load)
            model.training_provenance = {**getattr(model, "training_provenance", {}), "normalizer": {"source": str(normalizer_path), "fit_spec": normalizer_fit_spec}}
        global_step = 0
        start_chunk = args.start_chunk

    end_chunk = len(chunks) if args.max_chunks is None else min(
        len(chunks), start_chunk + args.max_chunks
    )
    global_step_target = global_step + sum(
        args.steps_per_chunk or (
            math.ceil(len(_keys_for_chunk(split, set(chunks[i].domains), args.max_train_frames))
                      / (args.batch_size * world)) * args.epochs_per_chunk
        )
        for i in range(start_chunk, end_chunk)
    ) - resume_step

    loss_config = ForceLossConfig(**config.get("loss", {}))
    if rank == 0:
        print(json.dumps({
        "device": str(args.device),
        "world_size": world,
        "global_batch_size": args.batch_size * world,
        "prefetch": args.prefetch,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_seconds": args.checkpoint_seconds,
        "domains": len(order),
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_range": [start_chunk, end_chunk],
        "global_train_frames": len(split.train),
        "global_step": global_step,
        "global_step_target": global_step_target,
        "log_every": args.log_every,
        "log_seconds": args.log_seconds,
        "normalizer": normalizer.as_dict(),
        "normalizer_path": str(normalizer_path),
        "manifest": str(manifest_path) if manifest_path else None,
    }, indent=2), flush=True)

    execution = {"world_size": world, "batch_size": args.batch_size,
                 "epochs_per_chunk": args.epochs_per_chunk, "steps_per_chunk": args.steps_per_chunk,
                 "train_key_sha256": _key_digest(split.train)}
    if old_rotation.get("rng_states") and len(old_rotation["rng_states"]) == world:
        restore_rng(old_rotation["rng_states"])
    for index in range(start_chunk, end_chunk):
        chunk = chunks[index]
        keys = _keys_for_chunk(split, set(chunk.domains), args.max_train_frames)
        if not keys:
            raise RuntimeError(f"chunk {index} has no train frames after the global split")
        reader = _reader(args, config, chunk.domains)
        try:
            dataset = HeavyFlowPhysicsFrameDataset(reader, keys, history_config=HistoryConfig.from_config(config))
            full_epoch_steps = math.ceil(len(dataset) / (args.batch_size * world))
            steps = args.steps_per_chunk or full_epoch_steps * args.epochs_per_chunk
            completed_before = resume_step if index == start_chunk else 0
            if completed_before >= steps:
                raise ValueError("resume step must precede chunk end")
            chunk_step_offset = global_step - completed_before
            if rank == 0:
                print(json.dumps({
                "event": "chunk_start",
                "chunk": chunk.as_dict(),
                "train_frames": len(dataset),
                "steps": steps,
                "resume_step": completed_before,
                "global_step": global_step,
            }), flush=True)
            progress = _ChunkProgress(
                chunk_index=index, steps=steps, steps_per_epoch=full_epoch_steps,
                step_offset=chunk_step_offset, global_step_target=global_step_target,
                log_every=args.log_every, log_seconds=args.log_seconds, start_step=completed_before)
            last_checkpoint = time.monotonic()
            statistics = dict(old_rotation.get("partial_loss", {})) if completed_before else {}
            statistics = statistics or {"count": 0, "sum": 0.0, "first": None, "last": None}

            def rotation_state(completed, rng_states):
                return {
                    **({'corpus_plan_digest': plan['digest'], 'corpus_chunk_index': args.corpus_chunk_index,
                        'normalizer_origin': old_rotation.get('normalizer_origin', 'imported_legacy_train' if plan.get('legacy') else 'first_corpus_chunk_train')}
                       if plan else {}),
                    "version": 2, "mode": "domain_chunk_rotation", "chunk_size": args.chunk_size,
                    "total_chunks": len(chunks), "domain_order": order,
                    "domain_order_manifest": str(manifest_path) if manifest_path else None,
                    "completed_chunks": [item["chunk"] for item in rotation_history],
                    "next_chunk": index + 1 if completed == steps else index,
                    "step_in_chunk": 0 if completed == steps else completed,
                    "current_chunk": chunk.as_dict(), "chunk_metrics": rotation_history,
                    "normalizer_scope": "fixed_training_reference" if plan else "global_train_split", "frames_per_trajectory": args.frames_per_trajectory,
                    "max_train_frames": args.max_train_frames, "execution": execution,
                    "rng_states": rng_states, "partial_loss": dict(statistics),
                    "normalizer_fit_identity": _normalizer_data_identity(normalizer_fit_spec),
                }

            def on_step(completed, loss):
                nonlocal last_checkpoint
                statistics["count"] += 1
                statistics["sum"] += loss
                statistics["first"] = loss if statistics["first"] is None else statistics["first"]
                statistics["last"] = loss
                if rank == 0:
                    progress(completed, loss)
                # Rank zero decides the wall-clock trigger, all ranks must
                # participate in RNG collection at the same optimizer step.
                due = torch.tensor(int(rank == 0 and completed < steps and (
                    (args.checkpoint_every > 0 and completed % args.checkpoint_every == 0) or
                    (args.checkpoint_seconds > 0 and time.monotonic() - last_checkpoint >= args.checkpoint_seconds))),
                    device=args.device)
                if world > 1:
                    dist.broadcast(due, src=0)
                if bool(due):
                    rotation = rotation_state(completed, capture_rng())
                    rank_zero_call(lambda: _atomic_save_stage3(
                        output, model, normalizer=normalizer, split_manifest=split, config=config,
                        optimizer=optimizer, step=chunk_step_offset + completed,
                        chunk_rotation=rotation, metrics=dict(statistics), audit=None))
                    last_checkpoint = time.monotonic()
                    if rank == 0:
                        print(json.dumps({"event": "checkpoint_saved", "path": str(output),
                                          "chunk": index, "chunk_step": completed,
                                          "global_step": chunk_step_offset + completed}), flush=True)

            losses = train_physics_stream_steps(
                model,
                dataset,
                optimizer,
                normalizer=normalizer,
                steps=steps,
                batch_size=args.batch_size,
                loss_config=loss_config,
                step_offset=chunk_step_offset,
                start_step=completed_before,
                prefetch=args.prefetch,
                progress_callback=on_step,
            )
            if torch.device(args.device).type == "cuda":
                print(json.dumps({"event": "rank_memory", "rank": rank, "chunk": index,
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}), flush=True)
            chunk_metric: dict[str, Any] = {
                "chunk": index,
                "domains": list(chunk.domains),
                "num_domains": chunk.num_domains,
                "train_frames": len(dataset),
                "steps": steps,
                "loss_first": statistics["first"],
                "loss_last": statistics["last"],
                "loss_mean": statistics["sum"] / statistics["count"],
                "global_step_start": chunk_step_offset,
                "global_step_end": global_step + len(losses),
            }
            rotation_history.append(chunk_metric)
            global_step += len(losses)
            rotation = rotation_state(steps, capture_rng())
            audit = {
                "status": "not_materialized",
                "note": "chunk rotation trains from a streaming frame index; run the dedicated target audit separately for a full baseline report",
                "split": "train",
                "chunk": index,
                "train_frames": len(dataset),
            }
            chunk_path = _chunk_checkpoint_path(output, index)
            rank_zero_call(lambda: _atomic_save_stage3(
                chunk_path,
                model,
                normalizer=normalizer,
                split_manifest=split,
                config=config,
                optimizer=optimizer,
                step=global_step,
                chunk_rotation=rotation,
                metrics=chunk_metric,
                audit=audit,
            ))
            rank_zero_call(lambda: _atomic_save_stage3(
                output,
                model,
                normalizer=normalizer,
                split_manifest=split,
                config=config,
                optimizer=optimizer,
                step=global_step,
                chunk_rotation=rotation,
                metrics=chunk_metric,
                audit=audit,
            ))
            if rank == 0:
                print(json.dumps({
                "event": "chunk_complete",
                "chunk": index,
                "checkpoint": str(chunk_path),
                "latest": str(output),
                "global_step": global_step,
                "loss_first": losses[0],
                "loss_last": losses[-1],
            }), flush=True)
        finally:
            reader.close()

    if rank == 0:
        print(json.dumps({
        "status": "complete" if end_chunk == len(chunks) else "partial",
        "next_chunk": end_chunk,
        "total_chunks": len(chunks),
        "global_step": global_step,
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
