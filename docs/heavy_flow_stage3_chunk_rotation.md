# Stage 3 mdCATH chunk rotation

> Historical v1 execution notes. For current architecture, normalizer reuse,
> v2 output paths, strict resume and explicit legacy warm start, use
> [heavy-flow v2](heavy_flow_v2_architecture.md). New normalizer scans now require `--fit-normalizer`.

`experiments/heavy_flow/train_physics_chunks.py` trains the Stage 3 upstream
over a deterministic domain order. The default chunk size is 500 domains, so
the current 1,000-domain corpus produces exactly two chunks. Domains, rather
than individual frames, are the rotation unit.

The runner creates one global leakage-safe frame split and one force
normalizer from the training split. The normalizer pass uses the adapter's
force-only path: it reads represented force rows, the heavy/all-atom selection
metadata, and the force-valid mask, but does not read coordinates, residue
semantics, or PLM embeddings. It then reads one full sample at a time from the
active chunk, keeps the GPU batch and bounded CPU prefetch batches in memory, and writes both a
per-chunk checkpoint and an atomic `*_latest.pt` checkpoint. A checkpoint
contains the complete Stage 3 upstream, optimizer moments, normalizer, global
step, domain order, completed chunks, and `next_chunk`.

Immediately after fitting, the runner also writes an atomic standalone JSON
normalizer artifact. By default it is
`outputs/heavy_flow/stage3/chunk_rotation_normalizer.json`; override it with
`--normalizer-path`. A fresh invocation automatically loads this file when its
recorded data directory, representation, manifest order, split, frame sampling,
chunk size, and train-key fingerprint match the current run. Model settings
(including edge chunk size and activation checkpointing) are provenance only
and do not invalidate the force statistics; existing schema-v1 artifacts are
compatible without rewriting or refitting. A data/split mismatch reports the
changed fields and requires matching settings or a separate `--normalizer-path`
for a new fit. Resuming a checkpoint treats the checkpoint's
normalizer as authoritative and refreshes the standalone artifact. If a process
stops inside a chunk, the new periodic checkpoint resumes at the next saved
optimizer step. Only work since that save is repeated. Old checkpoints still
resume at their recorded chunk boundary; completed chunks are never repeated.

The ESM-C implementation is selected by `configs/heavy_flow/stage3.yaml`.
The command below uses the `esm3` environment and the real ESM-C backend. The
mdCATH adapter still requires the existing ESM-2 cache for its residue-side
input; `--allow-fake-plm` is intentionally not included.

Before Stage 3 training, populate the strict ESM-C embedding cache. This pass
uses only each shard's residue topology, not trajectory coordinates, forces, or
the ESM-2 cache. It is content-addressed and resumable: rerunning the command
reuses entries already written. The current single-process runner uses physical
GPU 6 (`cuda:0`); GPU 7 remains available for another process.

```bash
nohup env CUDA_VISIBLE_DEVICES=6,7 \
  PYTHONUNBUFFERED=1 \
  ./scripts/run_precompute_esmc.sh \
  > outputs/heavy_flow/stage3/precompute_esmc.log 2>&1 &
echo $!
```

Check progress with `tail -f outputs/heavy_flow/stage3/precompute_esmc.log`.
The run must end with `"status": "complete"` before starting Stage 3.

```bash
nohup env CUDA_VISIBLE_DEVICES=6,7 \
  PYTHONUNBUFFERED=1 \
  ./scripts/run_stage3_chunk_rotation.sh \
  > outputs/heavy_flow/stage3/chunk_rotation.log 2>&1 &
echo $!
```

Inside this single process, `cuda:0` maps to physical GPU 6. The launcher keeps
single-GPU behavior by default; exposing GPU 7 alone does not enable DDP.
Progress is flushed as JSON `train_progress` events after the first and last
optimizer step, every 100 steps, or after 60 seconds at the next completed step.
Tune these thresholds with `--log-every` and `--log-seconds`. Events report the
chunk step/target, epoch, global step/target, progress percentages, loss, elapsed
seconds, recent seconds per step, and chunk/overall ETA in seconds. The global
target includes resumed steps and the remaining chunks selected for this run.
ETA extrapolates the latest logging interval, including sample loading and
forward/backward time; it excludes future checkpoint overhead and varies with
domain size. Code edits affect new processes only, not a running trainer.
Use `tail -f` on the log and resume with:

```bash
nohup env CUDA_VISIBLE_DEVICES=6,7 \
  PYTHONUNBUFFERED=1 \
  ./scripts/run_stage3_chunk_rotation.sh \
  --resume outputs/heavy_flow/stage3/chunk_rotation_latest.pt \
  > outputs/heavy_flow/stage3/chunk_rotation_resume.log 2>&1 &
```

To perform a bounded smoke of one chunk, add `--max-chunks 1
--max-train-frames 2 --steps-per-chunk 1`; omit those limits for the intended
training run. The smoke options are explicit and are never defaults.

## Optimized training on GPUs 6 and 7

Wait until **both GPUs are available**. Do not launch this alongside the old
GPU-6 trainer: the old process has no periodic saves and stopping it loses
work after its latest completed chunk. Code changes do not hot-patch it.

```bash
mkdir -p outputs/heavy_flow/stage3
nohup env CUDA_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/run_stage3_chunk_rotation.sh \
  > outputs/heavy_flow/stage3/chunk_rotation_ddp.log 2>&1 &
echo $!
```

This uses `esm3`, torchrun/NCCL, one process per GPU and **batch 1 per GPU**
(global batch 2). It writes `chunk_rotation_ddp_latest.pt`, separate from the
old single-GPU output, and reuses `chunk_rotation_normalizer.json`. Each rank
gets adjacent, non-overlapping frames from the same deterministic order. Tail
ranks with no real frame run a zero-weight placeholder to participate in DDP;
no duplicate contributes to the loss. Loss/gradients are weighted by active
proteins, matching the existing protein-level objective, not by atom count.
Unused pair/axial outputs and modality-dropout branches are handled by wrapping
the scalar force objective in DDP with unused-parameter detection.

Same frame coverage requires roughly half as many updates as batch-1 training;
this changes the optimization trajectory. LR remains `1e-4` (no automatic
scaling). Global step and ETA refer to synchronized optimizer updates, not
the combined count of per-rank frames. Two GPUs do not pool VRAM.

Defaults preserve FP32, edge tiles of 2048, and activation checkpointing in all
message blocks. No AMP, edge dropping, atom truncation, auto batch growth, or
automatic OOM skipping is enabled. Non-finite loss or gradients fail before
the optimizer update on all ranks; restart from the latest saved checkpoint.
An OOM also terminates the run rather than silently excluding a large frame.

Optimizations that do not change model width/depth:

- Spatial neighbors use destination tiles of 256 and stable tensor sorting;
  cutoff, source-ID tie breaking, directed edges and final ordering are kept.
  As with other floating-point kernel changes, distances precisely at cutoff
  boundaries can differ by rounding. Topology transfers are bulk CPU copies.
- Tensor-product aggregation reuses one accumulation buffer per block.
- Static bonds are cached for up to eight domains per dataset/rank.
- One background thread prepares CPU batches (`--prefetch 1`; `0` disables),
  without forking HDF5 handles or CUDA state. Additional queued/in-flight CPU
  batches are bounded by prefetch depth; only the current batch moves to GPU.
- Global split and normalizer fitting/cache I/O run on rank zero once.
- DDP gradient bucket views avoid a second persistent gradient allocation.

Logs retain step targets, progress and recent wall-clock ETA; rank-zero
progress includes peak allocated/reserved VRAM, and each rank reports its
own peak in `rank_memory` at chunk completion. Prefetch wait time and input
preparation are included in wall-clock step throughput.

### Periodic save and exact-position resume

Defaults save **every 500 updates within each chunk**, plus every chunk end.
The time-based trigger is disabled by default (`--checkpoint-seconds 0`).
Set `--checkpoint-every` and `--checkpoint-seconds` to override these defaults;
zero disables that individual trigger. The launcher also accepts
`CHECKPOINT_EVERY` and `CHECKPOINT_SECONDS`. Already-running processes retain
their startup settings until restarted. Rank zero
atomically replaces latest after all ranks have contributed their Python,
NumPy, CPU Torch and local CUDA RNG states. Files retain plain upstream keys
(no DDP `module.` prefix), optimizer, split, normalizer, cumulative metrics,
world size, per-rank batch, schedule, and position within the chunk.

```bash
nohup env CUDA_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/run_stage3_chunk_rotation.sh \
  --resume outputs/heavy_flow/stage3/chunk_rotation_ddp_latest.pt \
  > outputs/heavy_flow/stage3/chunk_rotation_ddp_resume.log 2>&1 &
echo $!
```

Mid-chunk resume rejects changed world size/batch/schedule/data identity.
Changing from single GPU to DDP is supported at a **chunk boundary**, including
legacy checkpoints; specify that checkpoint with `--resume`. It is not an
identical training trajectory across a batch-size/world-size change. CPU tests
verify exact same-world RNG/optimizer continuation, including dropout and
activation checkpointing; CUDA kernels need not be bitwise deterministic.

### Bounded real-config speed/VRAM check

On an available GPU 7, without writing a training checkpoint:

```bash
nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH=.:src OMP_NUM_THREADS=2 \
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/esm3/bin/python \
  experiments/heavy_flow/benchmark_physics.py --steps 3 \
  > outputs/heavy_flow/stage3/benchmark_physics.log 2>&1 &
echo $!
```

On 2026-09-08, H100 GPU 7, full `stage3.yaml`, real cached ESM-C and real
`1a0rP01` (1017 heavy atoms): cold step 2.83 s; two warm steps averaged 1.86 s;
peak allocated 4.18 GiB, reserved 4.27 GiB. This repeats one frame and is not
a full-corpus peak-memory bound, controlled before/after speedup, or two-GPU
throughput measurement. GPU 6 was left running its pre-existing job.
With `--check-resume`, the benchmark also saves the full model and Adam state
to a temporary checkpoint, reloads them and performs one GPU update. This
passed on GPU 7 with the real config (saved step 2, resumed step 3). The
temporary checkpoint is removed afterward; production outputs are untouched.

## Adding more mdCATH domains

The reproducible downloader uses the same seeded global order. To extend the
existing flat corpus without replacing its 1,000 files, request a larger
prefix in the same `data/` directory:

```bash
nohup env HF_XET_HIGH_PERFORMANCE=1 \
  /home/ubuntu/miniforge3/envs/esm3/bin/python scripts/download_mdcath.py \
  --num-domains 1500 --out-dir data \
  > outputs/heavy_flow/stage3/mdcath_extend_1500.log 2>&1 &
```

This adds the next 500 domains selected by the same seed and rewrites the
manifest to cover all 1,500 local shards. Re-run the trainer with a fresh
output path for that expanded corpus; an old checkpoint is rejected when its
domain order or total chunk count differs. `--chunk-index` is available for
isolated acquisition into `data/chunks/chunk_NNN`, but those isolated files
must be assembled into one manifest-covered data directory before continuing a
single global rotation run.
