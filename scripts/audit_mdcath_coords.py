#!/usr/bin/env python
"""Find mdCATH frames whose coordinates are physically impossible.

    python scripts/audit_mdcath_coords.py --data-dir data --out mdcath_coord_quarantine.json

Two defects, both of which destroy a frame while leaving it superficially valid.

**Saturated.** Every atom sits at ``2.147483647e6 nm = 2.14748e7 A``. That number
is ``INT_MAX / 1000``: the XTC writer stores coordinates as int32 in units of
1/1000 nm, and an overflow saturates at INT_MAX.

**Collapsed.** Every atom sits at the *same* position, usually exactly the
origin -- a final frame that was allocated but never written. Forces in such a
frame are perfectly normal, which is why nothing downstream objects.

A magnitude test finds the first and is blind to the second: a frame of all
zeros has ``|coord| = 0`` and passes any threshold. The invariant that catches
both is **spatial extent** -- a protein has to occupy space. This audit checks
extent, and reports magnitude only to classify what it found.

The collapsed frames matter more than their count suggests, because they are
*last* frames, and ``np.linspace(0, n-1, k)`` always includes ``n-1``. Every
sampling scheme built on evenly spaced frames therefore draws them with
certainty.

``check_pbc`` in the adapter catches neither, and cannot be asked to: it compares
consecutive CA-CA distances, and when every atom coincides those distances are
exactly zero, so a destroyed frame looks more intact than a real one.

The output is consumed by ``MdCathConfig.coord_quarantine_path``, which drops the
listed frames while the index is built, so a corrupt frame is never sampled
rather than being detected at load time and killing a multi-hour run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

#: A 250-residue domain spans roughly 60 A. Anything past 10,000 A is not a
#: protein that drifted, it is a broken value; the real defect is 2.1e7, so this
#: threshold is three orders of magnitude clear of both sides.
SANE_ANGSTROM = 1.0e4

#: The smallest radius of gyration a real protein can have. The smallest domain
#: here is 50 residues and spans tens of Angstrom; a destroyed frame measures
#: exactly 0. Anything under 1 A is not a compact protein, it is a collapsed one.
MIN_EXTENT_ANGSTROM = 1.0


def audit_domain(path: str) -> tuple[str, dict, float, int, dict]:
    import h5py  # noqa: PLC0415

    domain = os.path.basename(path)[len("mdcath_dataset_") : -len(".h5")]
    bad: dict[str, list[int]] = {}
    kinds: dict[str, int] = {"saturated": 0, "collapsed": 0, "nonfinite": 0}
    worst = 0.0
    total = 0
    with h5py.File(path, "r") as f:
        g = f[domain]
        for temp in sorted(k for k in g.keys() if k.isdigit()):
            for rep in sorted(g[temp].keys()):
                ds = g[temp][rep].get("coords")
                if ds is None or ds.shape[0] == 0:
                    continue
                coords = np.asarray(ds[:], dtype=np.float32)
                total += coords.shape[0]
                flat = coords.reshape(coords.shape[0], -1)
                finite = np.isfinite(flat).all(axis=1)
                magnitude = np.abs(np.where(np.isfinite(flat), flat, 0.0)).max(axis=1)
                worst = max(worst, float(magnitude.max()))
                # Extent, not magnitude: this is what a collapsed frame fails.
                centred = coords - coords.mean(axis=1, keepdims=True)
                extent = np.linalg.norm(centred, axis=2).max(axis=1)

                collapsed = extent < MIN_EXTENT_ANGSTROM
                saturated = magnitude > SANE_ANGSTROM
                hit = np.nonzero(~finite | saturated | collapsed)[0]
                if len(hit):
                    bad[f"{temp}/{rep}"] = [int(i) for i in hit]
                    kinds["nonfinite"] += int((~finite).sum())
                    kinds["saturated"] += int((saturated & finite).sum())
                    kinds["collapsed"] += int((collapsed & finite & ~saturated).sum())
    return domain, bad, worst, total, kinds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="mdcath_coord_quarantine.json")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"auditing {len(paths)} shards with {args.workers} workers", flush=True)

    quarantine: dict[str, dict[str, list[int]]] = {}
    worst_overall = 0.0
    frames_total = frames_bad = 0
    done = 0
    tally = {"saturated": 0, "collapsed": 0, "nonfinite": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_domain, p): p for p in paths}
        for fut in as_completed(futures):
            domain, bad, worst, total, kinds = fut.result()
            worst_overall = max(worst_overall, worst)
            frames_total += total
            if bad:
                quarantine[domain] = bad
                frames_bad += sum(len(v) for v in bad.values())
                for k in tally:
                    tally[k] += kinds[k]
                label = ", ".join(f"{v} {k}" for k, v in kinds.items() if v)
                print(f"  {domain}: {sum(len(v) for v in bad.values())} bad frames in "
                      f"{len(bad)} trajectories ({label})", flush=True)
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(paths)} shards", flush=True)

    payload = {
        "data_dir": args.data_dir,
        "num_shards": len(paths),
        "sane_angstrom": SANE_ANGSTROM,
        "min_extent_angstrom": MIN_EXTENT_ANGSTROM,
        "kinds": tally,
        "frames_scanned": frames_total,
        "frames_quarantined": frames_bad,
        "num_affected_domains": len(quarantine),
        "max_abs_coord_seen": worst_overall,
        "note": (
            "Frames a protein cannot occupy. 'saturated': every atom at "
            "INT_MAX/1000 nm = 2.14748e7 A, an int32 overflow in the upstream XTC "
            "writer. 'collapsed': every atom at one point, usually the origin -- a "
            "final frame allocated but never written, which passes every magnitude "
            "test and is drawn with certainty by evenly spaced sampling because it "
            "is the last frame. Dropped, not repaired."
        ),
        "quarantine": quarantine,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nscanned {frames_total:,} frames")
    print(f"quarantined {frames_bad:,} frames across {len(quarantine)} domains")
    print(f"  by kind: {tally}")
    print(f"max |coord| seen: {worst_overall:.6g} A")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
