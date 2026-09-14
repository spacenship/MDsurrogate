"""Revision-pinned, two-slot disk rotation. Run with the esm3 Python environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
import urllib.error
import urllib.request

from force_md.heavy_flow.corpus_plan import build_plan, chunk_entries, load_plan
from force_md.heavy_flow.checkpoint import ARCHITECTURE_VERSION, architecture_config, load_normalizer_artifact, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class IncompleteDownload(RuntimeError):
    pass


def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


class ShardDownloader:
    """Serial shard requests; completed files are skipped by prepare().

    Failed transfers restart their .part file. Local disk errors fail immediately.
    Pacing persists across chunks because the one preparation worker reuses this object.
    """
    def __init__(self, interval=5.0, retries=8, backoff=30.0, backoff_max=600.0):
        if (not all(math.isfinite(x) and x > 0 for x in (interval, backoff, backoff_max))
                or retries < 0 or backoff_max < backoff):
            raise ValueError('download intervals must be positive, max >= base, retries >= 0')
        self.interval, self.retries = interval, retries
        self.backoff, self.backoff_max = backoff, backoff_max
        self.next_request = 0.0

    def download(self, url, temporary, target, expected_size, *, chunk, domain):
        delay = self.backoff
        for attempt in range(self.retries + 1):
            wait = self.next_request - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            server_wait = 0.0
            status = None
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                if temporary.stat().st_size != expected_size:
                    raise IncompleteDownload(f'incomplete download: {domain}')
                temporary.replace(target)
                return
            except urllib.error.HTTPError as error:
                status = error.code
                server_wait = retry_after_seconds(error.headers.get('Retry-After') if error.headers else None)
                error.close()
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise
                failure = error
            except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead, IncompleteDownload) as error:
                failure = error
            finally:
                self.next_request = time.monotonic() + self.interval
            if attempt == self.retries:
                raise RuntimeError(f'download retries exhausted for {domain} after {attempt + 1} attempts') from failure
            wait = max(self.interval, delay, server_wait)
            print(json.dumps({'event': 'download_retry', 'chunk': chunk, 'domain': domain,
                              'attempt': attempt + 1, 'next_attempt': attempt + 2,
                              'max_attempts': self.retries + 1, 'status': status,
                              'error_type': type(failure).__name__, 'wait_seconds': wait}), flush=True)
            self.next_request = time.monotonic() + wait
            delay = min(delay * 2, self.backoff_max)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def completed(path, plan, index):
    payload = checkpoint(path)
    validate_checkpoint(payload, require_trained=True)
    r = payload['chunk_rotation']
    if (payload.get('stage') != 'stage3_atom_physics' or not payload.get('optimizer_state')
            or r.get('corpus_plan_digest') != plan['digest']
            or r.get('corpus_chunk_index') != index or r.get('step_in_chunk') != 0
            or r.get('next_chunk') != r.get('total_chunks')):
        raise ValueError('refusing eviction: checkpoint is not a completed matching chunk')


def slot(root, index):
    root = Path(root).resolve()
    path = root / f'chunk_{index:04d}'
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError('unsafe rotation directory')
    return path


def evict(root, plan, index):
    directory = slot(root, index)
    targets = [directory / Path(e['path']).name for e in chunk_entries(plan, index)]
    if any(p.is_symlink() or p.resolve().parent != directory for p in targets):
        raise ValueError('unsafe shard eviction target')
    for path in targets:
        path.unlink(missing_ok=True)
    print(json.dumps({'event': 'raw_shards_evicted', 'chunk': index, 'files': len(targets)}), flush=True)


def prepare(args, plan, index):
    from huggingface_hub import hf_hub_url
    from download_mdcath import run_audits
    if not hasattr(args, '_downloader'):
        args._downloader = ShardDownloader(
            getattr(args, 'download_interval', 5.0), getattr(args, 'download_retries', 8),
            getattr(args, 'download_backoff', 30.0), getattr(args, 'download_backoff_max', 600.0))
    directory = slot(args.work_dir, index)
    directory.mkdir(parents=True, exist_ok=True)
    entries = chunk_entries(plan, index)
    manifest = directory / 'mdcath_manifest.json'
    atomic_json(manifest, {'shards': entries, 'repo_id': plan['repo_id'], 'seed': plan['seed'],
                           'revision': plan['revision'], 'corpus_plan_digest': plan['digest'],
                           'chunk_index': index})
    for n, entry in enumerate(entries):
        target = directory / Path(entry['path']).name
        temporary = target.with_suffix('.h5.part')
        if target.is_symlink() or temporary.is_symlink():
            raise ValueError('symlink shard target')
        if target.exists() and target.stat().st_size == entry['size']:
            continue
        if shutil.disk_usage(directory).free < entry['size'] + args.reserve_gb * 1e9:
            raise RuntimeError('insufficient disk space for next shard plus reserve; current checkpoint retained')
        url = hf_hub_url(plan['repo_id'], entry['path'], repo_type='dataset', revision=plan['revision'])
        args._downloader.download(url, temporary, target, entry['size'], chunk=index, domain=entry['domain'])
        print(json.dumps({'event': 'download', 'chunk': index, 'done': n + 1, 'total': len(entries)}), flush=True)
    if run_audits(str(directory), ROOT, manifest=manifest, output_dir=directory):
        raise RuntimeError(f'chunk {index} audit failed')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, str(ROOT / 'scripts/precompute_esmc.py'),
                    '--config', args.config, '--data-dir', str(directory),
                    '--domain-order-manifest', str(manifest), '--device', 'cpu'],
                   cwd=ROOT, env=env, check=True)
    print(json.dumps({'event': 'chunk_ready', 'chunk': index}), flush=True)
    return directory


def run_pipeline(args, plan):
    state_path = Path(args.work_dir) / 'state.json'
    initial = plan.get('legacy')
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'architecture_version': ARCHITECTURE_VERSION,
        'digest': plan['digest'], 'next_chunk': initial['next_chunk'] if initial else 0,
        'checkpoint': str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None}
    if state.get('architecture_version') != ARCHITECTURE_VERSION:
        raise ValueError('legacy corpus state: use a new v2 work directory to start at chunk 0')
    if state['digest'] != plan['digest']:
        raise ValueError('state/plan identity mismatch')
    index, previous = state['next_chunk'], state['checkpoint']
    if initial and not previous:
        raise ValueError('imported plan requires --initial-checkpoint on first run')
    total = math.ceil(len(plan['shards']) / plan['chunk_size'])
    # Recover a crash after state commit but before eviction, before prefetch starts.
    if index > (initial['next_chunk'] if initial else 0):
        completed(previous, plan, index - 1)
        evict(args.work_dir, plan, index - 1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        ready = pool.submit(prepare, args, plan, index) if index < total else None
        while index < total:
            directory = ready.result()
            ready = pool.submit(prepare, args, plan, index + 1) if index + 1 < total else None
            output = Path(args.output_dir).resolve() / f'corpus_{index:04d}_latest.pt'
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(ROOT / 'experiments/heavy_flow/train_physics_chunks.py'),
                   '--config', args.config, '--data-dir', str(directory),
                   '--domain-order-manifest', str(directory / 'mdcath_manifest.json'),
                   '--corpus-manifest', str(Path(args.plan).resolve()), '--corpus-chunk-index', str(index),
                   '--chunk-size', str(plan['chunk_size']), '--output', str(output),
                   '--quarantine-path', str(directory / 'mdcath_force_quarantine.json'),
                   '--coord-quarantine-path', str(directory / 'mdcath_coord_quarantine.json'),
                   '--device', 'cuda:0', '--checkpoint-every', '500', '--checkpoint-seconds', '0']
            for option, default in [('frames_per_trajectory', 4), ('epochs_per_chunk', 1), ('batch_size', 1)]:
                cmd += ['--' + option.replace('_', '-'), str(getattr(args, option, default))]
            if output.exists():
                cmd += ['--resume', str(output)]
            elif previous:
                cmd += ['--initialize-from', previous]
            else:
                cmd += ['--reuse-normalizer', str(Path(getattr(args, 'reuse_normalizer',
                    'outputs/heavy_flow/stage3/chunk_rotation_normalizer.json')).resolve())]
                if getattr(args, 'warm_start', None):
                    cmd += ['--warm-start', str(Path(args.warm_start).resolve())]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '6,7', 'OMP_NUM_THREADS': '2',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            print(json.dumps({'event': 'train_chunk', 'chunk': index, 'total_chunks': total}), flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            completed(output, plan, index)  # torchrun has joined every rank/reader.
            previous = str(output)
            atomic_json(state_path, {'architecture_version': ARCHITECTURE_VERSION,
                                    'digest': plan['digest'], 'next_chunk': index + 1, 'checkpoint': previous})
            evict(args.work_dir, plan, index)
            index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='outputs/heavy_flow/stage3_v2/corpus_plan.json')
    parser.add_argument('--repo-id', default='compsciencelab/mdCATH')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--test-fraction', type=float, default=.1)
    parser.add_argument('--initial-checkpoint')
    parser.add_argument('--work-dir', default='data_rotation_v2')
    parser.add_argument('--output-dir', default='outputs/heavy_flow/stage3_v2/corpus')
    parser.add_argument('--reuse-normalizer', default='outputs/heavy_flow/stage3/chunk_rotation_normalizer.json')
    parser.add_argument('--warm-start', help='compatible weights only for chunk 0; train v2 from the beginning')
    parser.add_argument('--frames-per-trajectory', type=int, default=4)
    parser.add_argument('--epochs-per-chunk', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--config', default='configs/heavy_flow/stage3.yaml')
    parser.add_argument('--reserve-gb', type=float, default=100)
    parser.add_argument('--download-interval', type=float, default=5.0, help='minimum seconds after each shard request before the next')
    parser.add_argument('--download-retries', type=int, default=8, help='retries per shard after the initial attempt')
    parser.add_argument('--download-backoff', type=float, default=30.0, help='initial retry delay in seconds; doubles on failure')
    parser.add_argument('--download-backoff-max', type=float, default=600.0, help='backoff cap; Retry-After may exceed this')
    parser.add_argument('--plan-only', action='store_true', help='Create/inspect immutable plan without downloading shards or training')
    args = parser.parse_args()
    try:
        args._downloader = ShardDownloader(args.download_interval, args.download_retries,
                                           args.download_backoff, args.download_backoff_max)
    except ValueError as error:
        parser.error(str(error))
    os.chdir(ROOT)
    from experiments.heavy_flow.train_physics import load_yaml
    architecture_config(load_yaml(args.config))
    if min(args.frames_per_trajectory, args.epochs_per_chunk, args.batch_size) < 1:
        parser.error('frame, epoch and batch counts must be positive')
    if args.initial_checkpoint and args.warm_start:
        parser.error('initial-checkpoint and warm-start are mutually exclusive')
    if args.initial_checkpoint:
        validate_checkpoint(checkpoint(args.initial_checkpoint), require_trained=True)
    else:
        # Fail before catalog lookup/download if the fixed RMS cannot be reused.
        load_normalizer_artifact(args.reuse_normalizer)
    if args.reserve_gb < 0 or not math.isfinite(args.reserve_gb):
        parser.error('reserve-gb must be finite and nonnegative')
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.work_dir) / '.rotation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path(args.plan).exists():
            plan = load_plan(args.plan)
            if plan['chunk_size'] != args.chunk_size:
                raise ValueError('existing plan chunk size differs; use its original --chunk-size')
            if plan.get('legacy') and not args.initial_checkpoint:
                raise ValueError('imported plan cannot start at chunk 0; use a new v2 plan path')
        else:
            from huggingface_hub import HfApi
            api = HfApi()
            revision = api.repo_info(args.repo_id, repo_type='dataset').sha
            shards = [(e.path, e.size) for e in api.list_repo_tree(args.repo_id, repo_type='dataset',
                       revision=revision, path_in_repo='data', recursive=True)
                      if e.path.endswith('.h5') and hasattr(e, 'size')]
            plan = build_plan(shards, repo_id=args.repo_id, revision=revision, seed=args.seed,
                              chunk_size=args.chunk_size, validation_fraction=args.validation_fraction,
                              test_fraction=args.test_fraction,
                              checkpoint=checkpoint(args.initial_checkpoint) if args.initial_checkpoint else None)
            atomic_json(args.plan, plan)
        print(json.dumps({'event': 'corpus_plan', 'domains': len(plan['shards']),
                          'splits': {k: len(v) for k, v in plan['splits'].items()},
                          'digest': plan['digest'], 'revision': plan['revision']}), flush=True)
        if not args.plan_only:
            run_pipeline(args, plan)


if __name__ == '__main__':
    main()
