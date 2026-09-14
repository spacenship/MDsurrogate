"""Deterministic domain chunks for streaming mdCATH training.

The mdCATH files are large enough that a training run should not treat the
whole corpus as one in-memory dataset.  This module defines the reproducible
unit used by the Stage 3 chunk runner: a chunk is a fixed number of domains,
and the order comes from the downloader manifest when one is available.

The helpers do not touch the files or download anything.  They only resolve
the domain order and split it into non-overlapping chunks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DomainChunk",
    "domain_order",
    "build_domain_chunks",
    "load_domain_chunks",
]


def _domain_from_path(path: str) -> str:
    name = os.path.basename(path)
    prefix, suffix = "mdcath_dataset_", ".h5"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"not an mdCATH shard name: {path!r}")
    return name[len(prefix) : -len(suffix)]


def _local_domains(data_dir: str | Path) -> set[str]:
    root = Path(data_dir)
    return {
        _domain_from_path(path.name)
        for path in root.glob("*.h5")
        if path.is_file()
    }


def domain_order(
    data_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[str]:
    """Return the reproducible domain order available in ``data_dir``.

    If a downloader manifest is supplied, its shard order is authoritative.
    This is important because ``download_mdcath.py`` uses a seeded shuffle;
    sorting the filenames would silently change which domains belong to each
    chunk.  Missing manifest shards fail closed instead of training on a
    partial corpus.
    """
    local = _local_domains(data_dir)
    if not local:
        raise ValueError(f"no mdCATH .h5 shards found in {data_dir}")

    if manifest_path is None:
        return sorted(local)

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"domain-order manifest does not exist: {path}")
    payload = json.loads(path.read_text())
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest has no non-empty 'shards' list: {path}")
    ordered = [_domain_from_path(str(item["path"])) for item in shards]
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"manifest contains duplicate domains: {path}")
    missing = [domain for domain in ordered if domain not in local]
    if missing:
        raise ValueError(
            f"{len(missing)} manifest domain(s) are missing from {data_dir}, "
            f"e.g. {missing[:3]}"
        )
    # Extra local files are not silently mixed into a manifest-defined run.
    extra = sorted(local - set(ordered))
    if extra:
        raise ValueError(
            f"{len(extra)} local domain(s) are not listed by {path}, "
            f"e.g. {extra[:3]}; use a manifest covering the complete data dir"
        )
    return ordered


@dataclass(frozen=True)
class DomainChunk:
    """One non-overlapping slice of the global domain order."""

    index: int
    domains: tuple[str, ...]
    chunk_size: int
    total_chunks: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= self.total_chunks:
            raise ValueError("chunk index is outside the total chunk range")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not self.domains:
            raise ValueError("a domain chunk cannot be empty")

    @property
    def num_domains(self) -> int:
        return len(self.domains)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "domains": list(self.domains),
            "num_domains": self.num_domains,
            "chunk_size": self.chunk_size,
            "total_chunks": self.total_chunks,
        }


def build_domain_chunks(
    domains: Sequence[str] | Iterable[str],
    *,
    chunk_size: int = 500,
) -> list[DomainChunk]:
    """Split domains into ordered chunks, with at most ``chunk_size`` items."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ordered = list(domains)
    if not ordered:
        raise ValueError("cannot build chunks from an empty domain order")
    if len(set(ordered)) != len(ordered):
        raise ValueError("domain order contains duplicates")
    total = (len(ordered) + chunk_size - 1) // chunk_size
    return [
        DomainChunk(
            index=index,
            domains=tuple(ordered[start : start + chunk_size]),
            chunk_size=chunk_size,
            total_chunks=total,
        )
        for index, start in enumerate(range(0, len(ordered), chunk_size))
    ]


def load_domain_chunks(
    data_dir: str | Path,
    *,
    chunk_size: int = 500,
    manifest_path: str | Path | None = None,
) -> list[DomainChunk]:
    """Resolve the local corpus and return its deterministic chunk plan."""
    return build_domain_chunks(
        domain_order(data_dir, manifest_path=manifest_path),
        chunk_size=chunk_size,
    )
