"""Conditioning inputs: frozen PLM embeddings, residue identity, temperature."""

from .esm2 import (
    DEFAULT_ESM2_MODEL,
    ESM2_EMBED_DIM,
    ESM2_MAX_RESIDUES,
    Esm2Config,
    Esm2EmbeddingCache,
    compute_esm2_embeddings,
    residue_sequence,
    sequence_fingerprint,
    strip_special_tokens,
)
from .residue import ResidueConditioner
from .temperature import TEMPERATURE_FEATURE_DIM, temperature_features

__all__ = [
    "DEFAULT_ESM2_MODEL",
    "ESM2_EMBED_DIM",
    "ESM2_MAX_RESIDUES",
    "Esm2Config",
    "Esm2EmbeddingCache",
    "compute_esm2_embeddings",
    "residue_sequence",
    "sequence_fingerprint",
    "strip_special_tokens",
    "ResidueConditioner",
    "temperature_features",
    "TEMPERATURE_FEATURE_DIM",
]
