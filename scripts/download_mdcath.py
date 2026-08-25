#!/usr/bin/env python
"""Download a reproducible subset of the mdCATH dataset into MDsurrogate/data/.

The subset is defined by a fixed seed over the *sorted* list of shard paths in
the HuggingFace repo, so re-running with the same ``--seed``/``--num-domains``
always selects the same domains -- and a larger ``--num-domains`` is a strict
superset of a smaller one (the selection is a prefix of one shuffled order).

This downloads into MDsurrogate's own directory. It never reads from, writes to,
or hard-links against MDensemble's copy of mdCATH.

Usage:
    python scripts/download_mdcath.py --num-domains 1000
    python scripts/download_mdcath.py --num-domains 1000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# huggingface_hub >= 1.x transfers via Xet; the old HF_HUB_ENABLE_HF_TRANSFER
# flag is deprecated and ignored. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DEFAULT_REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"
DATA_PREFIX = "data"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def list_shards(repo_id: str) -> list[tuple[str, int]]:
    """Return [(path_in_repo, size_bytes)] for every .h5 shard, sorted by path."""
    api = HfApi()
    entries = api.list_repo_tree(
        repo_id, repo_type=REPO_TYPE, path_in_repo=DATA_PREFIX, recursive=True
    )
    shards = [
        (e.path, e.size)
        for e in entries
        if e.path.endswith(".h5") and getattr(e, "size", None)
    ]
    return sorted(shards)


def select_subset(
    shards: list[tuple[str, int]], num_domains: int, seed: int
) -> list[tuple[str, int]]:
    """Deterministic prefix of a seeded shuffle, so subsets nest."""
    order = list(range(len(shards)))
    random.Random(seed).shuffle(order)
    return [shards[i] for i in order[:num_domains]]


def download_one(repo_id: str, path_in_repo: str, size: int, out_dir: Path) -> tuple[str, str, int]:
    """Returns (basename, status, bytes_downloaded). Skips already-complete files."""
    name = os.path.basename(path_in_repo)
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip", 0
    hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type=REPO_TYPE,
        local_dir=str(out_dir),
        # keep the flat layout: hf places it at out_dir/<path_in_repo>, so we
        # move it up afterwards if the repo nests it under data/.
    )
    nested = out_dir / path_in_repo
    if nested.exists() and nested != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(nested), str(dest))
    if not dest.exists():
        return name, "missing", 0
    got = dest.stat().st_size
    if got != size:
        return name, f"size-mismatch({got}!={size})", got
    return name, "ok", got


def run_audits(data_dir: str, root: Path) -> int:
    """Regenerate both quarantine files over the freshly downloaded shards.

    This runs as part of downloading, not as a step someone is told about in a
    README, because mdCATH ships frames that are unusable while looking entirely
    normal -- ``forces`` byte-identical to ``coords``, coordinates saturated at
    INT_MAX, and final frames that were allocated but never written. None of
    those announce themselves; a collapsed frame in particular has ordinary
    forces and coordinates of exactly zero, so it passes every magnitude check
    and is drawn once per epoch by evenly spaced sampling.

    The adapter refuses to run when a configured quarantine file is absent, so
    the two halves agree: the data is not usable until this has been done.
    """
    import subprocess  # noqa: PLC0415

    for script, out in (("audit_mdcath_forces.py", "mdcath_force_quarantine.json"),
                        ("audit_mdcath_coords.py", "mdcath_coord_quarantine.json")):
        log(f"\n=== {script} ===")
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / script),
             "--data-dir", data_dir, "--out", str(root / out)],
            check=False,
        )
        if result.returncode != 0:
            log(f"AUDIT FAILED: {script} exited {result.returncode}. Training will "
                f"refuse to start until {out} exists.")
            return result.returncode
    log("\nquarantine files regenerated; the data is ready for training")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--num-domains", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=200.0,
                    help="Abort before starting a new file if free space drops below this.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-audit", action="store_true",
                    help="Do not regenerate the quarantine files after downloading. "
                         "Only for resuming an interrupted download that will be "
                         "audited afterwards.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"listing shards in {args.repo_id} ...")
    shards = list_shards(args.repo_id)
    log(f"repo has {len(shards)} shards, {sum(s for _, s in shards) / 1e12:.2f} TB")

    if args.num_domains > len(shards):
        log(f"ERROR: requested {args.num_domains} > available {len(shards)}")
        return 2

    subset = select_subset(shards, args.num_domains, args.seed)
    total = sum(s for _, s in subset)
    have = sum(
        (out_dir / os.path.basename(p)).stat().st_size
        for p, s in subset
        if (out_dir / os.path.basename(p)).exists()
    )
    log(f"selected {len(subset)} domains (seed={args.seed}): {total / 1e9:.0f} GB total, "
        f"{have / 1e9:.0f} GB already present, {(total - have) / 1e9:.0f} GB to fetch")

    manifest = out_dir.parent / "mdcath_manifest.json"
    manifest.write_text(json.dumps(
        {"repo_id": args.repo_id, "seed": args.seed, "num_domains": args.num_domains,
         "shards": [{"path": p, "size": s} for p, s in subset]}, indent=1))
    log(f"manifest written -> {manifest}")

    if args.dry_run:
        log("dry-run: nothing downloaded")
        return 0

    done = 0
    fetched = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, args.repo_id, p, s, out_dir): (p, s)
            for p, s in subset
        }
        for fut in as_completed(futures):
            p, s = futures[fut]
            done += 1
            try:
                name, status, got = fut.result()
                if status not in ("ok", "skip"):
                    failures.append((name, status))
                fetched += got
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                failures.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
            if done % 10 == 0 or done == len(subset):
                free_gb = shutil.disk_usage(out_dir).free / 1e9
                el = time.time() - t0
                rate = fetched / el / 1e6 if el > 0 else 0.0
                log(f"{done}/{len(subset)} done | {fetched / 1e9:.0f} GB fetched "
                    f"({rate:.0f} MB/s) | free {free_gb:.0f} GB | failures {len(failures)}")
            if shutil.disk_usage(out_dir).free / 1e9 < args.min_free_gb:
                log(f"ABORT: free space below --min-free-gb={args.min_free_gb}")
                for f in futures:
                    f.cancel()
                break

    log(f"finished in {(time.time() - t0) / 60:.1f} min | fetched {fetched / 1e9:.0f} GB "
        f"| failures {len(failures)}")
    for name, why in failures[:20]:
        log(f"  FAIL {name}: {why}")
    if failures:
        return 1

    if args.skip_audit:
        log("\n--skip-audit given: the quarantine files were NOT regenerated. "
            "They are inputs to training, not optional extras -- run "
            "scripts/audit_mdcath_forces.py and scripts/audit_mdcath_coords.py "
            "before training on this data.")
        return 0
    return run_audits(str(out_dir), Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
