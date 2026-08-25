#!/usr/bin/env python
"""Find mdCATH trajectories whose `forces` dataset is a copy of `coords`.

A fraction of the official mdCATH shards ship a `forces` array that is
byte-identical to `coords`. Those force labels are meaningless, and force is the
whole supervision signal in Phase 1, so they must be found and masked rather
than trained on. The defect is per *trajectory*, not per file -- the observed
pattern is replicas 0-3 corrupt at every temperature with replica 4 intact -- so
dropping whole shards would also throw away good data.

Read-only. Writes a quarantine JSON consumed by the mdCATH adapter.

Usage:
    python scripts/audit_mdcath_forces.py --data-dir data --out mdcath_force_quarantine.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def audit_file(path: str, num_probe_frames: int = 3) -> tuple[dict[str, list[str]], list[str]]:
    """Return ({domain: [bad trajectory keys]}, [warnings]) for one shard."""
    bad: list[str] = []
    warn: list[str] = []
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            warn.append(f"expected 1 domain group, found {len(keys)}")
        domain = keys[0]
        g = f[domain]
        for temp in sorted(k for k in g.keys() if k.isdigit()):
            for rep in sorted(g[temp].keys()):
                r = g[temp][rep]
                if "coords" not in r or "forces" not in r:
                    warn.append(f"{temp}/{rep}: missing coords or forces")
                    bad.append(f"{temp}/{rep}")
                    continue
                c, fo = r["coords"], r["forces"]
                if c.shape != fo.shape:
                    warn.append(f"{temp}/{rep}: shape {c.shape} vs {fo.shape}")
                    bad.append(f"{temp}/{rep}")
                    continue
                nf = c.shape[0]
                probes = sorted({0, nf // 2, nf - 1})[:num_probe_frames]
                if all(np.array_equal(c[j], fo[j]) for j in probes):
                    bad.append(f"{temp}/{rep}")
    return {domain: bad}, warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data-dir", default=str(root / "data"))
    ap.add_argument("--out", default=str(root / "mdcath_force_quarantine.json"))
    ap.add_argument("--manifest", default=str(root / "mdcath_manifest.json"),
                    help="If present, also verify on-disk sizes against it.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not files:
        print(f"no .h5 shards in {args.data_dir}", file=sys.stderr)
        return 2
    print(f"auditing {len(files)} shard(s) in {args.data_dir}", flush=True)

    size_problems: list[str] = []
    if os.path.exists(args.manifest):
        man = json.load(open(args.manifest))
        want = {os.path.basename(s["path"]): s["size"] for s in man["shards"]}
        for p in files:
            b = os.path.basename(p)
            if b in want and os.path.getsize(p) != want[b]:
                size_problems.append(b)
        print(f"size check: {len(files) - len(size_problems)}/{len(files)} match manifest",
              flush=True)

    quarantine: dict[str, list[str]] = {}
    warnings: dict[str, list[str]] = {}
    n_traj = n_bad = 0
    errors: list[tuple[str, str]] = []
    t0 = time.time()

    for i, p in enumerate(files):
        try:
            per_domain, warn = audit_file(p)
            for domain, bad in per_domain.items():
                with h5py.File(p, "r") as f:
                    total = sum(
                        len(f[domain][t].keys())
                        for t in f[domain].keys() if t.isdigit()
                    )
                n_traj += total
                n_bad += len(bad)
                if bad:
                    quarantine[domain] = bad
                if warn:
                    warnings[domain] = warn
        except Exception as e:  # noqa: BLE001
            errors.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} | corrupt {n_bad}/{n_traj} traj "
                  f"| {time.time()-t0:.0f}s", flush=True)

    payload = {
        "data_dir": os.path.abspath(args.data_dir),
        "num_shards": len(files),
        "num_trajectories": n_traj,
        "num_corrupt_trajectories": n_bad,
        "num_affected_domains": len(quarantine),
        "size_mismatch": size_problems,
        "errors": errors,
        "warnings": warnings,
        # domain -> ["<temperature>/<replica>", ...] whose forces must not be trained on
        "quarantine": quarantine,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))

    pct = 100.0 * n_bad / max(n_traj, 1)
    print(f"\nshards {len(files)} | trajectories {n_traj} | corrupt {n_bad} ({pct:.2f}%) "
          f"| affected domains {len(quarantine)}")
    print(f"size mismatches: {len(size_problems)} | errors: {len(errors)}")
    print(f"quarantine written -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
