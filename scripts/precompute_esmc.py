#!/usr/bin/env python
"""Precompute the strict ESM-C sequence cache used by Stage 3.

This command has no dependency on trajectory frames or the ESM-2 cache.  It
reads each mdCATH shard's residue topology, computes one ESM-C embedding per
unique sequence, and stores detached FP32 entries in the content-addressed
cache configured for Stage 3.  Existing entries are reused, so an interrupted
run can simply be started again.

Example (physical GPU 6; GPU 7 is intentionally not used by this process):

    CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 \
      /home/ubuntu/miniforge3/envs/esm3/bin/python scripts/precompute_esmc.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import torch

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow import ESMCConfig, ESMCEmbeddingCache, ESMCEncoder, HeavyFlowCondition
from force_md.heavy_flow.chunk_rotation import domain_order


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def _default_manifest_path(data_dir: str | Path) -> Optional[Path]:
    root = Path(data_dir)
    for candidate in (root / "mdcath_manifest.json", root.parent / "mdcath_manifest.json"):
        if candidate.exists():
            return candidate
    return None


def _sequence_condition(tokens: torch.Tensor, device: torch.device) -> HeavyFlowCondition:
    """Make the smallest valid condition accepted by ESMCEncoder.precompute."""
    tokens = tokens.to(device=device, dtype=torch.int64).reshape(1, -1)
    length = int(tokens.shape[1])
    empty_atoms = torch.empty((1, 0), dtype=torch.int64, device=device)
    return HeavyFlowCondition(
        sequence_tokens=tokens,
        residue_mask=torch.ones((1, length), dtype=torch.bool, device=device),
        atom_type=empty_atoms,
        atom_name=empty_atoms.clone(),
        atom_to_residue=empty_atoms.clone(),
        atom_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
        x_history=torch.empty((1, 1, 0, 3), dtype=torch.float32, device=device),
        bond_index=torch.empty((1, 2, 0), dtype=torch.int64, device=device),
        bond_type=torch.empty((1, 0), dtype=torch.int64, device=device),
        temperature=torch.zeros((1,), dtype=torch.float32, device=device),
        lag=torch.ones((1,), dtype=torch.float32, device=device),
    )


def _sequence_hash(tokens: torch.Tensor) -> str:
    values = tokens.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _build_esmc_config(
    config: dict[str, Any], *, cache_dir: str, device: str
) -> ESMCConfig:
    sequence = dict(config.get("sequence_encoder", {}))
    if "family" in sequence:
        sequence["model_family"] = sequence.pop("family")
    allowed = {
        "model_family", "model_name", "revision", "backend", "model_cache_dir",
        "preprocessing_version", "projected_dim", "stub_dim", "stub_seed",
    }
    values = {key: sequence[key] for key in allowed if key in sequence}
    values.update(cache_dir=cache_dir, device=device, cache_only=False)
    return ESMCConfig(**values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heavy_flow/stage3.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--domain-order-manifest", default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="Embedding cache; defaults to sequence_encoder.cache_dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the domain count without loading ESM-C weights")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.max_domains is not None and args.max_domains < 1:
        raise SystemExit("--max-domains must be positive")
    if args.log_every < 1:
        raise SystemExit("--log-every must be positive")

    config = _load_yaml(args.config)
    if config.get("stage") != "stage3_atom_physics":
        raise SystemExit("--config must describe stage3_atom_physics")
    sequence_config = dict(config.get("sequence_encoder", {}))
    cache_dir = Path(
        args.cache_dir
        or sequence_config.get("cache_dir", "outputs/heavy_flow/esmc_cache")
    )
    manifest_path = (
        Path(args.domain_order_manifest)
        if args.domain_order_manifest
        else _default_manifest_path(args.data_dir)
    )
    domains = domain_order(args.data_dir, manifest_path=manifest_path)
    if args.start_index >= len(domains):
        raise SystemExit(f"--start-index must be less than {len(domains)}")
    end = len(domains) if args.max_domains is None else min(
        len(domains), args.start_index + args.max_domains
    )
    selected = domains[args.start_index:end]

    print(json.dumps({
        "event": "precompute_start",
        "device": args.device,
        "domains_total": len(domains),
        "domain_range": [args.start_index, end],
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path) if manifest_path else None,
        "dry_run": args.dry_run,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    cache_dir.mkdir(parents=True, exist_ok=True)
    esmc = ESMCEncoder(_build_esmc_config(config, cache_dir=str(cache_dir), device=args.device))
    cache = ESMCEmbeddingCache(cache_dir)
    reader = MdCathDataset(
        MdCathConfig(
            data_dir=args.data_dir,
            represented_scope=str(config.get("data", {}).get("represented_scope", "heavy_atom")),
        ),
        domains=selected,
        build_index=False,
    )
    created = 0
    reused = 0
    try:
        for offset, domain in enumerate(selected, start=args.start_index):
            token_values = torch.from_numpy(reader.sequence_tokens_for(domain))
            digest = _sequence_hash(token_values)
            key = esmc._key(digest)
            existed = cache.exists(key)
            condition = _sequence_condition(token_values, device)
            hashes = esmc.precompute(condition)
            if hashes != (digest,):
                raise RuntimeError(f"unexpected ESM-C sequence hash for {domain}")
            if existed:
                reused += 1
                status = "reused"
            else:
                created += 1
                status = "created"
            if status == "created" or (offset - args.start_index) % args.log_every == 0:
                print(json.dumps({
                    "event": "domain_complete",
                    "index": offset,
                    "domain": domain,
                    "residues": int(token_values.numel()),
                    "status": status,
                    "cache_key": key,
                    "created": created,
                    "reused": reused,
                }), flush=True)
    finally:
        reader.close()

    print(json.dumps({
        "status": "complete",
        "domains_processed": len(selected),
        "created": created,
        "reused": reused,
        "cache_entries": len(list(cache_dir.glob("*.pt"))),
        "cache_dir": str(cache_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
