# Stage 3: actual disk rotation

## Current v2 launcher (2026-09-12)

The launcher now defaults to a fresh v2 run from chunk 0, with 500 domains per
chunk. It trains only the training split within those 500 domains. During
training it downloads/audits/precomputes ESM-C for the next chunk on CPU.
After torchrun exits and a matching **force-trained v2** completed checkpoint
is validated, it commits state and deletes only the current chunk's listed
raw `.h5` files. Existing `data/` and v1 rotation directories are untouched.

Default paths:

- Plan: `outputs/heavy_flow/stage3_v2/corpus_plan.json`
- Raw slots/state: `data_rotation_v2/chunk_NNNN/`, `data_rotation_v2/state.json`
- Checkpoints: `outputs/heavy_flow/stage3_v2/corpus/corpus_NNNN_latest.pt`
- Fixed RMS source: `outputs/heavy_flow/stage3/chunk_rotation_normalizer.json`

Before downloading, each slot records `mdcath_manifest.json` with the exact
domain/file list, sizes, repository revision, plan digest and chunk index.
This is the selected download list, not a claim that every download completed.
The manifest, audit results, ESM cache and checkpoints survive raw-file deletion.
Only the current and next chunk's raw data are managed at once.

The first chunk uses `--reuse-normalizer`; no normalizer scan is started.
Subsequent chunks use `--initialize-from` to carry v2 weights, Adam, global step
and the same normalizer. Same-chunk restarts use `--resume`. Old state is rejected;
the default paths do not inherit old chunk numbers. To reuse old weights while
still starting from chunk 0, add `--warm-start OLD_STAGE3.pt` on the first run.
This is compatible-weight initialization with a report, followed by v2 force
training, not old-architecture resume. Do not pass `--initial-checkpoint` to the
fresh chunk-0 command.

Launch command (not executed during implementation):

```bash
cd /data1/miplab/wjyang/MDsurrogate
mkdir -p outputs/heavy_flow/stage3_v2
nohup env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  bash scripts/run_stage3_corpus.sh \
  --chunk-size 500 --frames-per-trajectory 4 --epochs-per-chunk 1 --batch-size 1 \
  --reuse-normalizer outputs/heavy_flow/stage3/chunk_rotation_normalizer.json \
  > outputs/heavy_flow/stage3_v2/corpus_rotation.log 2>&1 &
```

The launcher uses physical GPUs 6,7 and the `esm3` environment. This command
allows the network catalog and shard downloads, unlike the offline local-data
training command. Add `--plan-only` to inspect/create the pinned list without
downloading shards, training or eviction; a missing plan still requires a catalog
lookup. Repeating the launch command resumes v2 state rather than resetting to 0.
History and model settings come from `configs/heavy_flow/stage3.yaml`.

### Request pacing and retry

Shard downloads now wait at least `--download-interval 5` seconds after the
previous attempt ends. The serial preparation worker shares this limiter across
chunk boundaries. This paces shard attempts, not each HTTP redirect or the
initial Hugging Face catalog lookup.

`--download-retries 8` permits up to 9 total attempts per file. HTTP 408/429 and
500/502/503/504, URL/connection/timeouts, and incomplete transfers are retried.
The backoff starts at `--download-backoff 30` seconds, doubles, and is capped at
`--download-backoff-max 600`. Numeric or HTTP-date `Retry-After` is respected;
the server's requested delay can exceed the cap. Each retry logs `download_retry`
with chunk/domain, attempt, HTTP status and wait duration. Permanent HTTP errors
and local disk errors fail immediately. Exhaustion preserves existing data and
prevents that chunk from being trained or evicted.

Completed files with the expected size are skipped. Failed partial files restart
from byte 0 and become `.h5` only after size verification and atomic rename.
Resume with the existing plan/work/output paths; append logs to retain the
previous failure record:

```bash
cd /data1/miplab/wjyang/MDsurrogate
nohup env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  bash scripts/run_stage3_corpus.sh \
  --download-interval 5 --download-retries 8 \
  --download-backoff 30 --download-backoff-max 600 \
  >> outputs/heavy_flow/stage3_v2/corpus_rotation.log 2>&1 &
```

Retry/pipeline tests: **18 passed**, with mocked requests and time (no actual
network or sleeping). They cover pacing across chunks, numeric/date Retry-After,
exponential/capped backoff, exhaustion, non-retryable errors, incomplete transfers,
and skipping already completed files. The downloader was not restarted as part
of this code change.

Validation: **20 passed, 2 deselected** in corpus-plan/runtime CPU tests, covering
chunk-0 launch, normalizer reuse, checkpoint continuation, overlapping preparation,
manifest-before-download and retention, failed-training retention, and legacy
state/checkpoint rejection. Network, download and training actions were mocked in
pipeline lifecycle tests; existing runtime tests exercised small optimizer steps.
No actual corpus download, deletion of real shards, or full GPU training was run.

## Historical v1 execution record

The commands and paths below describe v1. Use the v2 command above for the
current architecture; see also [heavy-flow v2](heavy_flow_v2_architecture.md).

Use the `esm3` environment. The new launcher uses physical GPUs **6,7**
for two-rank DDP; it does not stop any existing process.

First create and inspect the immutable corpus plan (network catalog lookup,
but no shard downloads, training, or deletion):

```bash
bash scripts/run_stage3_corpus.sh --plan-only \
  --initial-checkpoint outputs/heavy_flow/stage3/chunk_rotation_ddp_latest.pt
```

Then launch from the repository root:

```bash
nohup bash scripts/run_stage3_corpus.sh \
  --initial-checkpoint outputs/heavy_flow/stage3/chunk_rotation_ddp_latest.pt \
  > outputs/heavy_flow/stage3/corpus_rotation.log 2>&1 &
```

The same command resumes an interrupted run. Do not run concurrently with an
existing GPU 6/7 training job. The work-directory lock prevents two copies of
this pipeline from running against the same directory.

## Ordering, splits, and continuation

`outputs/heavy_flow/stage3/corpus_plan.json` pins the HF repository revision,
seed, full domain order, shard sizes, train/validation/test domain lists,
and a content digest. Existing plans are reused, not regenerated from CLI seeds.
Defaults are seed 0, 500 domains/chunk, 80/10/10 train/validation/test.

When importing the completed 1000-domain legacy run, its order becomes the
first 1000 entries. Its historical training domains stay train; its unseen
validation domains stay validation; historical held-out frames remain held out.
New test domains are never selected from the legacy training data. Integer
rounding and locked historical validation can affect exact split proportions.
Continuation starts at chunk 2, entries `[1000:1500]`, not at chunk 0 again.
This ordering is anchored to the old run, not identical to a fresh whole-corpus shuffle.

Without an initial checkpoint the run starts at chunk 0. One force scale is
fit on that first chunk's training frames and kept fixed thereafter. Imported
runs retain the checkpoint's existing training-only scale. Neither is described
as a new full-corpus normalizer fit.

The existing checkpoint evaluation fields `same_domain_validation` and
`unseen_domain_validation` carry validation and test respectively in corpus mode;
split metadata records these roles explicitly. This launcher trains only train
frames; it does not add an evaluation loop or use test for model selection.

## Two-slot lifecycle

Prepare chunk 0 → train 0 while preparing 1 → save/join all ranks → delete 0
→ train 1 while preparing 2 → save/join → delete 1 → continue.

Preparation downloads only that exact manifest slice at the pinned revision,
checks sizes, runs force/coordinate audits, and precomputes genuine ESM-C caches
on CPU. Only after all preparation succeeds can a chunk train. Raw downloads
use `.part` files and atomic rename. Interrupted downloads restart that file.
No ESM-2 or fake PLM feature is needed on this heavy-flow path.

Model weights, Adam state, force scale, global step, and compatible rank RNG
states carry across chunk boundaries. Within-chunk resume retains the existing
strict schedule/split checks. Checkpoints save every 500 steps, with no timer.
Each chunk gets its own latest checkpoint and normalizer artifact.

Raw files are deleted only after a matching completed checkpoint is validated
and torchrun exits, closing all rank readers. Only manifest-listed `.h5` files
inside `data_rotation/chunk_NNNN` are removed; symlinks are rejected.
Checkpoints, audits, corpus plan, and ESM-C cache are retained. Deleted raw shards
can be downloaded again from the pinned revision. State is committed before
eviction so a restart can finish an interrupted cleanup.

**Existing `data/` files are deliberately not deleted or moved by this launcher.**
They continue to occupy disk in addition to the two managed slots. Free enough
space before launching; `--reserve-gb` defaults to 100 and is checked per shard.
The 1000-domain bound applies to managed raw shards, not to old data, retained
checkpoints, embeddings, or temporary files. Checkpoint storage also grows.

CPU cache preparation/download throughput can be slower than training. In that
case the next chunk waits; chunk changes also include a short DDP/model restart.
Progress events include download counts, chunk readiness, global chunk number,
training step progress, and completed raw-file eviction.

Implementation tests use CPU and mocked pipeline actions. Actual corpus
downloads, deletion, and a two-GPU end-to-end run must be verified separately.

### Residue order correction (2026-09-12)

Residue IDs are labels, not sorted sequence positions. The shared
`data/residue_order.py` maps `(chain, resid)` in first-occurrence source topology
order. The mdCATH geometry adapter, lightweight force loader, heavy PSF topology,
and ESM-C/ESM-2 sequence preparation now use this same mapping. Stable atom
regrouping remains available for interleaved residues; PSF indices and atomic
numbers are remapped consistently. CA validation no longer bridges different
chains or missing-backbone residue slots and does not assert PBC as the sole
possible cause of a long distance.

Regression evidence: `1fwxA01/320/0` frames 0 and 146 have source-order maximum
CA distances 4.038 and 4.011 Angstrom; sorting residue labels incorrectly gives
47.692 and 52.278 Angstrom. All 466 source-order adjacent residue pairs are bonded
in the PSF. Both frames now load with checks enabled, with raw coordinates,
force selection, residue mapping and PSF bond lengths verified by
`tests/test_residue_order.py`.

ESM-C cache keys include sequence hashes: unchanged sequences reuse existing
entries, corrected sequences create new entries during the launcher's existing
precompute step. No global cache deletion or force normalizer recomputation is
needed. ESM-2 reads now reject sequence mismatch, and its precompute script
regenerates mismatched entries. No bulk precomputation or training was run for
this fix. Restart the launcher to pick up the changed code; an already running
Python process retains imported code. Existing checkpoint weights remain shape
compatible, but continuation uses corrected input semantics and is not an exact
replay of the old run. A clean retrain is needed for a run trained exclusively
with corrected ordering; existing checkpoints must not be described that way.
