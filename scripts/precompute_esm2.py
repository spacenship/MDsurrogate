#!/usr/bin/env python
"""Precompute frozen ESM-2 residue embeddings for the mdCATH domains on disk.

Run this once. Training loads the cache and never runs a PLM forward pass: a
protein's sequence is constant across its ~440 frames and 25 trajectories, so
embedding per frame would recompute a constant 11000 times per domain.

Downloads the checkpoint on first use (~2.5 GB) into ``HF_HOME``. Nothing else
in the project needs ``transformers`` or network access.

Usage:
    python scripts/precompute_esm2.py --out-dir esm2_cache --device cuda:0
    python scripts/precompute_esm2.py --dry-run          # sizes only, no download
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_md.conditioning.esm2 import (  # noqa: E402
    ESM2_EMBED_DIM,
    Esm2Config,
    Esm2EmbeddingCache,
    compute_esm2_embeddings,
)
from force_md.data import residue_constants as rc  # noqa: E402


def domain_sequences(paths: list[str]) -> dict[str, str]:
    """One-letter sequence per domain, in residue order.

    Read from the per-atom ``resname`` array rather than any PDB text, and
    canonicalised through the CHARMM-aware vocabulary so ``HSD``/``HSE``/``HSP``
    become ``H`` instead of the unknown token.
    """
    out: dict[str, str] = {}
    for p in paths:
        with h5py.File(p, "r") as f:
            domain = list(f.keys())[0]
            g = f[domain]
            resid = np.asarray(g["resid"][:])
            resname = np.array([r.decode() for r in g["resname"][:]])
            _, first = np.unique(resid, return_index=True)
            order = np.sort(first)
            out[domain] = "".join(rc.one_letter(resname[i]) for i in order)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(root / "data"))
    ap.add_argument("--out-dir", default=str(root / "esm2_cache"))
    ap.add_argument("--model", default=Esm2Config.model_name)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report sequences and cache size without downloading.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no shards in {args.data_dir}", file=sys.stderr)
        return 2

    print(f"reading sequences from {len(paths)} shard(s) ...", flush=True)
    seqs = domain_sequences(paths)
    lengths = np.array([len(s) for s in seqs.values()])
    n_bytes = int(lengths.sum()) * ESM2_EMBED_DIM * 4
    print(f"  {len(seqs)} domains | residues min={lengths.min()} max={lengths.max()} "
          f"total={lengths.sum()}")
    print(f"  cache size at float32: {n_bytes / 1e9:.2f} GB  -> {args.out_dir}")
    print(f"  checkpoint download (first run): ~2.5 GB into HF_HOME="
          f"{os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    unknown = sum(s.count("X") for s in seqs.values())
    print(f"  unknown residues (X): {unknown}")

    if args.dry_run:
        print("dry-run: nothing downloaded or written")
        return 0

    config = Esm2Config(model_name=args.model, revision=args.revision, layer=args.layer)
    cache = Esm2EmbeddingCache(args.out_dir)

    todo = {d: s for d, s in seqs.items() if args.overwrite or not cache.exists(d)}
    print(f"\ncomputing {len(todo)} embedding(s) on {args.device} "
          f"({len(seqs) - len(todo)} already cached)", flush=True)

    items = sorted(todo.items(), key=lambda kv: len(kv[1]))  # group similar lengths
    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        embeddings = compute_esm2_embeddings(
            [s for _, s in chunk], config, device=args.device,
            batch_size=args.batch_size,
        )
        for (domain, seq), emb in zip(chunk, embeddings):
            cache.save(domain, seq, emb, config)
        done = start + len(chunk)
        if done % 50 < args.batch_size:
            print(f"  {done}/{len(items)} | {time.time()-t0:.0f}s", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
