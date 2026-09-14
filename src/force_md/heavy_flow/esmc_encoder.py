"""Frozen sequence-only ESM-C embeddings with an explicit cache boundary.

The real backend uses EvolutionaryScale's current Hugging Face-compatible
``esm.models.esmc`` implementation.  ESM-C is loaded from the documented
``biohub/ESMC-300M`` checkpoint unless an explicit model identifier is passed.
The deterministic backend is named ``stub`` everywhere in metadata and is
only for unit tests/synthetic smoke tests; it is not an ESM-C substitute.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from ..data.residue_constants import RESIDUE_TYPES, RESNAME_TO_ONE_LETTER
from .types import HeavyFlowCondition

__all__ = [
    "ESMCConfig",
    "ESMCUnavailable",
    "ESMCCacheMiss",
    "ESMCEmbeddingCache",
    "ESMCEncoding",
    "ESMCEncoder",
]


DEFAULT_ESMC_MODEL_NAME = "biohub/ESMC-300M"
_RESIDUE_ID_TO_ONE_LETTER = tuple(RESNAME_TO_ONE_LETTER[name] for name in RESIDUE_TYPES)


class ESMCUnavailable(RuntimeError):
    """Raised instead of silently using another protein language model."""


class ESMCCacheMiss(FileNotFoundError):
    """Raised by strict cache-only mode when a sequence was not precomputed."""


@dataclass(frozen=True)
class ESMCConfig:
    model_family: str = "ESMC-300M"
    model_name: Optional[str] = None
    revision: Optional[str] = None
    backend: str = "esmc"
    cache_dir: Optional[str] = None
    model_cache_dir: Optional[str] = None
    device: Optional[str] = None
    cache_only: bool = False
    preprocessing_version: str = "heavy_flow_esmc_sequence_v1"
    projected_dim: int = 64
    stub_dim: int = 32
    stub_seed: int = 0

    def __post_init__(self) -> None:
        if self.backend not in {"esmc", "stub"}:
            raise ValueError("ESMC backend must be 'esmc' or explicit test-only 'stub'")
        if self.model_family != "ESMC-300M":
            raise ValueError("Stage 1 currently defines the ESM-C-300M family only")
        if self.projected_dim <= 0 or self.stub_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.cache_only and self.cache_dir is None:
            raise ValueError("cache_only=True requires cache_dir")


@dataclass(frozen=True)
class ESMCEncoding:
    """Both sides of the frozen/trainable PLM boundary."""

    frozen_embedding: Tensor  # [B,L,D], detached FP32
    projected_embedding: Tensor  # [B,L,P], trainable projection output
    sequence_sha256: tuple[str, ...]
    provenance: dict[str, Any]


def _sequence_hash(tokens: Tensor) -> str:
    values = tokens.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _residue_sequence(tokens: Tensor) -> str:
    """Convert the repository's canonical residue IDs into an ESM sequence.

    ``HeavyFlowCondition.sequence_tokens`` stores IDs from
    ``force_md.data.residue_constants.RESIDUE_TYPES`` rather than ESM tokenizer
    IDs.  Converting through the canonical vocabulary avoids accidentally
    treating those internal IDs as model-specific token IDs.
    """

    values = tokens.detach().to(device="cpu", dtype=torch.int64).flatten().tolist()
    letters = []
    for value in values:
        index = int(value)
        letters.append(_RESIDUE_ID_TO_ONE_LETTER[index] if 0 <= index < len(_RESIDUE_ID_TO_ONE_LETTER) else "X")
    return "".join(letters)


def _resolved_dimension(model: Any) -> Optional[int]:
    candidates = [
        getattr(model, "embed_dim", None),
        getattr(model, "embedding_dim", None),
        getattr(model, "d_model", None),
    ]
    config = getattr(model, "config", None)
    if config is not None:
        candidates += [getattr(config, "embed_dim", None), getattr(config, "hidden_size", None)]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def _revision_candidates(model: Any) -> list[str]:
    candidates = [
        getattr(model, "_commit_hash", None),
        getattr(model, "revision", None),
        getattr(model, "model_revision", None),
    ]
    config = getattr(model, "config", None)
    if config is not None:
        candidates += [
            getattr(config, "_commit_hash", None),
            getattr(config, "revision", None),
        ]
    return [value.strip() for value in candidates if isinstance(value, str) and value.strip()]


def _resolved_hf_revision(model: Any, model_name: str, requested: Optional[str]) -> str:
    """Resolve a cache-safe revision, failing closed when it is unavailable."""

    candidates = _revision_candidates(model)
    if candidates:
        return candidates[0]
    if requested:
        # An explicitly supplied revision is part of the user-controlled
        # provenance contract when the loader cannot expose the resolved SHA.
        return requested
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha.strip():
            return sha.strip()
    except Exception as exc:  # pragma: no cover - depends on network/cache state
        raise ESMCUnavailable(
            "could not resolve the Hugging Face ESM-C revision; pass an exact "
            "commit hash via ESMCConfig(revision=...)"
        ) from exc
    raise ESMCUnavailable(
        "loaded ESM-C model exposes no resolved revision; pass an exact commit "
        "hash via ESMCConfig(revision=...)"
    )


class _SequenceOnlyStub(nn.Module):
    def __init__(self, dim: int, seed: int):
        super().__init__()
        # A deterministic buffer, not a trainable or claimed ESM-C weight.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randn((32, dim), generator=generator, dtype=torch.float32)
        self.register_buffer("token_table", table, persistent=False)
        self.embedding_dim = dim

    @torch.no_grad()
    def embed(self, tokens: Tensor) -> Tensor:
        ids = tokens.to(dtype=torch.long).clamp_min(0).remainder(self.token_table.shape[0])
        positions = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
        base = self.token_table.to(tokens.device).index_select(0, ids)
        phase = (positions.to(torch.float32) + 1.0) * torch.arange(
            1, self.embedding_dim + 1, device=tokens.device, dtype=torch.float32
        )[None, :] * 0.007
        return base + torch.sin(phase) + torch.cos(phase * 0.37)


class _RealESMCBackend(nn.Module):
    """Adapter for the installed ``esm.models.esmc`` Hugging Face API."""

    def __init__(self, config: ESMCConfig):
        super().__init__()
        try:
            module = importlib.import_module("esm.models.esmc")
            model_class = getattr(module, "EsmcForMaskedLM")
            tokenizer_class = getattr(module, "EsmcTokenizer")
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ESMCUnavailable(
                "ESMCConfig(backend='esmc') requires the EvolutionaryScale 'esm' "
                "package with esm.models.esmc.EsmcForMaskedLM and EsmcTokenizer. "
                "No ESM-2/ESM3 fallback is allowed."
            ) from exc

        self.model_name = config.model_name or DEFAULT_ESMC_MODEL_NAME
        model_cache_dir = config.model_cache_dir or config.cache_dir
        load_kwargs: dict[str, Any] = {}
        if config.revision is not None:
            load_kwargs["revision"] = config.revision
        if model_cache_dir is not None:
            load_kwargs["cache_dir"] = model_cache_dir
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = model_class.from_pretrained(
                self.model_name,
                device=device,
                **load_kwargs,
            )
            self.tokenizer = tokenizer_class()
        except Exception as exc:  # pragma: no cover - weight/network dependent
            raise ESMCUnavailable(
                f"could not load ESM-C checkpoint {self.model_name!r}. "
                "Download the Hugging Face weights in the esm3 environment or "
                f"provide a local cache/path: {exc}"
            ) from exc

        self.revision = _resolved_hf_revision(self.model, self.model_name, config.revision)
        self.embedding_dim = _resolved_dimension(self.model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - malformed external model
            return torch.device("cpu")

    def _trim_special_tokens(self, hidden: Tensor, encoded: Any, sequence_length: int) -> Tensor:
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
        if input_ids is None:
            active_hidden = hidden
        else:
            ids = input_ids[0]
            active = attention_mask[0].to(torch.bool) if attention_mask is not None else torch.ones_like(ids, dtype=torch.bool)
            ids = ids[active]
            active_hidden = hidden[active]
            special_ids = {
                int(value)
                for value in getattr(self.tokenizer, "all_special_ids", ())
                if value is not None
            }
            if special_ids:
                active_hidden = active_hidden[~torch.isin(ids, torch.tensor(sorted(special_ids), device=ids.device))]

        if active_hidden.shape[0] == sequence_length:
            return active_hidden
        # Keep a narrow fallback for tokenizer revisions that do not expose
        # all_special_ids but use the conventional BOS/EOS pair.
        if hidden.shape[0] == sequence_length + 2:
            return hidden[1:-1]
        raise ESMCUnavailable(
            f"ESM-C tokenizer produced {active_hidden.shape[0]} residue embeddings "
            f"for a sequence of length {sequence_length}; special-token handling "
            "is unresolved for this installed revision"
        )

    @torch.no_grad()
    def embed(self, tokens: Tensor) -> Tensor:  # pragma: no cover - requires weights
        sequence = _residue_sequence(tokens)
        encoded = self.tokenizer(sequence, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._model_device()) for key, value in encoded.items()}
        output = self.model(**encoded)
        embedding = getattr(output, "last_hidden_state", None)
        if embedding is None and isinstance(output, dict):
            embedding = output.get("last_hidden_state")
        if embedding is None and isinstance(output, (tuple, list)) and output:
            embedding = output[0]
        if embedding is None:
            raise ESMCUnavailable("ESM-C output has no last_hidden_state embedding")
        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        if embedding.ndim != 3 or embedding.shape[0] != 1:
            raise ESMCUnavailable(f"ESM-C output must have shape [1,T,D], got {tuple(embedding.shape)}")
        return self._trim_special_tokens(embedding[0], encoded, len(sequence)).contiguous()


class ESMCEmbeddingCache:
    """Content-addressed cache with model/revision/preprocess/sequence metadata."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*, model_name: str, revision: str, preprocessing_version: str, sequence_sha256: str) -> str:
        payload = {
            "model_name": model_name,
            "revision": revision,
            "preprocessing_version": preprocessing_version,
            "sequence_sha256": sequence_sha256,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.pt"

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def save(self, key: str, embedding: Tensor, metadata: dict[str, Any]) -> None:
        value = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        payload = {"embedding": value, "metadata": dict(metadata)}
        torch.save(payload, self.path_for(key))

    def load(self, key: str, *, expected_metadata: dict[str, Any]) -> Tensor:
        path = self.path_for(key)
        if not path.is_file():
            raise ESMCCacheMiss(f"strict ESM-C cache miss for key {key} at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "embedding" not in payload or "metadata" not in payload:
            raise ValueError(f"invalid ESM-C cache entry: {path}")
        if payload["metadata"] != expected_metadata:
            raise ValueError(f"ESM-C cache metadata mismatch for {path}; refusing stale embeddings")
        embedding = payload["embedding"]
        if not isinstance(embedding, Tensor) or embedding.dtype != torch.float32 or embedding.requires_grad:
            raise ValueError(f"ESM-C cache embedding must be detached float32: {path}")
        return embedding


class ESMCEncoder(nn.Module):
    """Frozen ESM-C sequence encoder plus a trainable projection."""

    def __init__(self, config: Optional[ESMCConfig] = None):
        super().__init__()
        self.config = config or ESMCConfig()
        if self.config.backend == "stub":
            self.backend = _SequenceOnlyStub(self.config.stub_dim, self.config.stub_seed)
            self.model_name = "ESMC-300M::explicit-test-stub"
            self.revision = f"stub-seed-{self.config.stub_seed}"
            input_dim: Optional[int] = self.config.stub_dim
        else:
            self.backend = _RealESMCBackend(self.config)
            self.model_name = self.backend.model_name
            self.revision = self.backend.revision
            input_dim = self.backend.embedding_dim
        if input_dim is None:
            self.projection = nn.LazyLinear(self.config.projected_dim)
        else:
            self.projection = nn.Linear(input_dim, self.config.projected_dim)
        self.cache = ESMCEmbeddingCache(self.config.cache_dir) if self.config.cache_dir else None

        for parameter in self.backend.parameters():
            parameter.requires_grad_(False)
        self.backend.eval()

    @property
    def provenance(self) -> dict[str, Any]:
        input_dim = getattr(self.backend, "embedding_dim", None)
        if input_dim is None and not isinstance(self.projection, nn.LazyLinear):
            input_dim = self.projection.in_features
        return {
            "model_family": self.config.model_family,
            "model_name": self.model_name,
            "resolved_revision": self.revision,
            "embedding_dim": input_dim,
            "projected_dim": self.config.projected_dim,
            "preprocessing_version": self.config.preprocessing_version,
            "sequence_only": True,
            "frozen": True,
            "backend": self.config.backend,
        }

    def _metadata(self, sequence_sha256: str) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "resolved_revision": self.revision,
            "preprocessing_version": self.config.preprocessing_version,
            "sequence_sha256": sequence_sha256,
        }

    def _key(self, sequence_sha256: str) -> str:
        return ESMCEmbeddingCache.key(
            model_name=self.model_name,
            revision=self.revision,
            preprocessing_version=self.config.preprocessing_version,
            sequence_sha256=sequence_sha256,
        )

    @torch.no_grad()
    def _compute_one(self, tokens: Tensor) -> Tensor:
        result = self.backend.embed(tokens)
        result = result.detach().to(device=tokens.device, dtype=torch.float32).contiguous()
        if result.ndim != 2 or result.shape[0] != tokens.shape[0]:
            raise RuntimeError("sequence backend returned an invalid [L,D] embedding")
        return result

    @torch.no_grad()
    def precompute(self, condition: HeavyFlowCondition) -> tuple[str, ...]:
        """Compute and write missing entries, even when forward is cache-only."""
        condition.validate()
        if self.cache is None:
            raise ValueError("precompute requires ESMCConfig(cache_dir=...)")
        hashes = []
        for row in range(condition.batch_size):
            tokens = condition.sequence_tokens[row][condition.residue_mask[row]]
            digest = _sequence_hash(tokens)
            hashes.append(digest)
            key = self._key(digest)
            if not self.cache.exists(key):
                self.cache.save(key, self._compute_one(tokens), self._metadata(digest))
        return tuple(hashes)

    def _frozen_batch(self, condition: HeavyFlowCondition) -> tuple[Tensor, tuple[str, ...]]:
        values: list[Tensor] = []
        hashes: list[str] = []
        for row in range(condition.batch_size):
            tokens = condition.sequence_tokens[row][condition.residue_mask[row]]
            digest = _sequence_hash(tokens)
            hashes.append(digest)
            key = self._key(digest)
            if self.config.cache_only:
                if self.cache is None:
                    raise ValueError("cache_only=True requires a cache")
                value = self.cache.load(key, expected_metadata=self._metadata(digest))
                value = value.to(condition.sequence_tokens.device)
            elif self.cache is not None and self.cache.exists(key):
                value = self.cache.load(key, expected_metadata=self._metadata(digest)).to(
                    condition.sequence_tokens.device
                )
            else:
                value = self._compute_one(tokens)
                if self.cache is not None:
                    self.cache.save(key, value, self._metadata(digest))
            values.append(value)
        dim = values[0].shape[-1] if values else 0
        padded = torch.zeros(
            (condition.batch_size, condition.sequence_length, dim),
            dtype=torch.float32,
            device=condition.sequence_tokens.device,
        )
        for row, value in enumerate(values):
            padded[row, condition.residue_mask[row]] = value
        return padded.detach(), tuple(hashes)

    def forward(self, condition: HeavyFlowCondition) -> ESMCEncoding:
        condition.validate()
        frozen, hashes = self._frozen_batch(condition)
        projected = self.projection(frozen)
        return ESMCEncoding(frozen, projected, hashes, self.provenance)
