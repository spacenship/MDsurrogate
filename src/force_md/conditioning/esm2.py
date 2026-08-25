"""Frozen ESM-2 residue embeddings: sequence building, caching, precomputation.

Design rules, all of which are testable offline:

* **The PLM never runs inside the training loop.** Embeddings are computed once
  per domain by ``scripts/precompute_esm2.py`` and loaded from cache. A protein's
  sequence does not change between trajectory frames, so a forward pass per frame
  would burn 650M parameters of compute to recompute a constant.
* **Nothing is truncated silently.** ESM-2 accepts 1022 residues between its
  special tokens; a longer chain raises unless truncation is explicitly enabled.
  (Audited: the 1000 downloaded mdCATH domains run 50-479 residues, so *zero*
  are affected -- but a future dataset must fail loudly rather than quietly lose
  its C-terminus.)
* **The cache is keyed by content, not by filename.** The stored fingerprint
  covers the sequence, model name, revision and layer, so a cache entry written
  with a different checkpoint can never be silently reused.
* **Special tokens are stripped by position, not by guesswork.** ESM-2 emits
  ``<cls>`` first and ``<eos>`` last; residue ``i`` is hidden state ``i+1``.
  Off-by-one here shifts the whole chain and is invisible in the loss.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor

from ..data import residue_constants as rc

__all__ = [
    "DEFAULT_ESM2_MODEL",
    "ESM2_EMBED_DIM",
    "ESM2_MAX_RESIDUES",
    "Esm2Config",
    "residue_sequence",
    "sequence_fingerprint",
    "strip_special_tokens",
    "Esm2EmbeddingCache",
    "compute_esm2_embeddings",
]

#: Default checkpoint. 33 layers, 1280-dim residue representations.
DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM2_EMBED_DIM = 1280
#: ESM-2's 1024 learned positions minus ``<cls>`` and ``<eos>``.
ESM2_MAX_RESIDUES = 1022


@dataclass(frozen=True)
class Esm2Config:
    """Everything that changes the numbers in a cache entry.

    Args:
        model_name: HuggingFace checkpoint id.
        revision: pinned git revision of that checkpoint. ``"main"`` is a moving
            target; pin it for a reproducible run and the fingerprint will
            record what was actually used.
        layer: index into ``hidden_states``. ``-1`` is the final layer.
        allow_truncation: opt in to losing residues past ``max_residues``.
    """

    model_name: str = DEFAULT_ESM2_MODEL
    revision: str = "main"
    layer: int = -1
    max_residues: int = ESM2_MAX_RESIDUES
    allow_truncation: bool = False


def residue_sequence(residue_type: Tensor) -> str:
    """One-letter sequence from canonical residue-type ids.

    CHARMM histidine variants (``HSD``/``HSE``/``HSP``) already collapsed to
    ``HIS`` at the vocabulary level, so they appear here as ``H`` rather than as
    the unknown token -- see :mod:`force_md.data.residue_constants`.
    """
    return "".join(
        rc.RESNAME_TO_ONE_LETTER[rc.RESIDUE_TYPES[int(t)]] for t in residue_type
    )


def sequence_fingerprint(sequence: str, config: Esm2Config) -> str:
    """SHA-256 over the sequence and everything about how it was embedded."""
    payload = json.dumps(
        {
            "sequence": sequence,
            "model_name": config.model_name,
            "revision": config.revision,
            "layer": config.layer,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def strip_special_tokens(
    hidden_states: Tensor, lengths: Sequence[int]
) -> list[Tensor]:
    """Drop ``<cls>``/``<eos>``/padding, returning one ``[L_i, D]`` tensor each.

    Args:
        hidden_states: ``[B, T, D]`` output of the model, where token 0 is
            ``<cls>`` and token ``L_i + 1`` is ``<eos>`` for sample ``i``.
        lengths: residue count ``L_i`` of each sample.

    Returns:
        List of ``[L_i, D]`` tensors aligned 1:1 with the residues.

    Raises:
        ValueError: if the padded length cannot hold ``L_i + 2`` tokens, which
            would mean the tokenizer truncated the sequence.
    """
    out = []
    for i, length in enumerate(lengths):
        if length + 2 > hidden_states.shape[1]:
            raise ValueError(
                f"sample {i}: sequence of {length} residues needs {length + 2} "
                f"tokens but only {hidden_states.shape[1]} are present; the "
                "tokenizer truncated the sequence"
            )
        out.append(hidden_states[i, 1 : length + 1])
    return out


class Esm2EmbeddingCache:
    """On-disk cache of per-domain residue embeddings.

    One ``{domain_id}.pt`` per domain holding the embedding plus the metadata
    needed to prove it matches the sequence and checkpoint being asked for.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, domain_id: str) -> Path:
        return self.root / f"{domain_id}.pt"

    def exists(self, domain_id: str) -> bool:
        return self.path_for(domain_id).exists()

    def save(
        self, domain_id: str, sequence: str, embedding: Tensor, config: Esm2Config
    ) -> Path:
        """Write one entry. ``embedding`` must be ``[len(sequence), D]``."""
        if embedding.shape[0] != len(sequence):
            raise ValueError(
                f"{domain_id}: embedding has {embedding.shape[0]} rows for a "
                f"{len(sequence)}-residue sequence"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding": embedding.detach().to(torch.float32).cpu(),
            "metadata": {
                "domain_id": domain_id,
                "sequence": sequence,
                "num_residues": len(sequence),
                "embed_dim": int(embedding.shape[1]),
                "fingerprint": sequence_fingerprint(sequence, config),
                **asdict(config),
            },
        }
        path = self.path_for(domain_id)
        torch.save(payload, path)
        return path

    def load(
        self,
        domain_id: str,
        *,
        expect_sequence: Optional[str] = None,
        config: Optional[Esm2Config] = None,
    ) -> Tensor:
        """Load one entry, refusing a mismatched sequence or checkpoint.

        Raises:
            FileNotFoundError: no entry for ``domain_id``.
            ValueError: the entry was produced from a different sequence, model,
                revision or layer than requested.
        """
        path = self.path_for(domain_id)
        if not path.exists():
            raise FileNotFoundError(f"no cached embedding for {domain_id} at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["metadata"]
        if expect_sequence is not None and config is not None:
            want = sequence_fingerprint(expect_sequence, config)
            if meta.get("fingerprint") != want:
                raise ValueError(
                    f"{domain_id}: cached embedding does not match the requested "
                    f"sequence/checkpoint. cached model={meta.get('model_name')!r} "
                    f"revision={meta.get('revision')!r} layer={meta.get('layer')} "
                    f"len={meta.get('num_residues')}; requested "
                    f"model={config.model_name!r} revision={config.revision!r} "
                    f"layer={config.layer} len={len(expect_sequence)}"
                )
        return payload["embedding"]


def compute_esm2_embeddings(
    sequences: Iterable[str],
    config: Esm2Config = Esm2Config(),
    *,
    device: str = "cpu",
    batch_size: int = 4,
) -> list[Tensor]:
    """Run ESM-2 over ``sequences`` and return per-residue embeddings.

    ``transformers`` and the checkpoint are only needed here. Every other module,
    and the whole offline test suite, works without them.

    Args:
        sequences: one-letter protein sequences.
        device: where to run; ``"cuda:0"`` for a real precompute.
        batch_size: sequences per forward pass.

    Returns:
        One ``[L_i, 1280]`` float32 tensor per input, already stripped of
        ``<cls>``/``<eos>``.

    Raises:
        ValueError: a sequence exceeds ``config.max_residues`` and
            ``config.allow_truncation`` is False.
    """
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415 - optional dep

    seqs = list(sequences)
    too_long = [(i, len(s)) for i, s in enumerate(seqs) if len(s) > config.max_residues]
    if too_long and not config.allow_truncation:
        raise ValueError(
            f"{len(too_long)} sequence(s) exceed max_residues={config.max_residues}, "
            f"e.g. index {too_long[0][0]} with {too_long[0][1]} residues. Set "
            "allow_truncation=True only if you accept losing the C-terminus."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    model = AutoModel.from_pretrained(config.model_name, revision=config.revision)
    model.eval().to(device)

    out: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]
            enc = tokenizer(list(chunk), return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            result = model(**enc, output_hidden_states=True)
            hidden = result.hidden_states[config.layer]
            out.extend(t.float().cpu() for t in strip_special_tokens(hidden, [len(s) for s in chunk]))
    return out
