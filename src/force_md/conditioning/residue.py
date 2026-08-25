"""Residue-level conditioning: PLM projection, residue identity, temperature.

Everything this module produces is an SE(3) **invariant** (``l=0``) scalar. That
is the whole point of keeping residue semantics on their own node type: chemical
identity has no orientation, so it must not be concatenated onto a vector or
``l=2`` channel. Scalars reach the equivariant part of the network in exactly two
sanctioned ways, both implemented in :mod:`force_md.nn`:

1. as **gates** multiplying an ``l>0`` feature by a scalar function, and
2. as **weights of tensor-product paths**, where a scalar MLP predicts the path
   weights from invariant edge/node features.

Broadcasting the 1280-dim PLM vector onto every child atom is deliberately *not*
one of them: it multiplies memory by the atom-per-residue factor and gives the
atom branch no information the residue branch does not already hold.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..data.contracts import HierarchicalProteinBatch
from ..data.residue_constants import NUM_RESIDUE_TYPES
from .esm2 import ESM2_EMBED_DIM
from .temperature import TEMPERATURE_FEATURE_DIM, temperature_features

__all__ = ["ResidueConditioner"]


class ResidueConditioner(nn.Module):
    """Invariant residue features from PLM embedding, identity and temperature.

    Args:
        plm_dim: width of the cached PLM embedding (1280 for ESM-2 650M).
        plm_projection_dim: width after projection.
        out_channels: width of the emitted invariant feature.
        use_plm: ablation hook. When False the PLM branch is skipped entirely and
            the module carries no dependence on the embedding, so a "no PLM" run
            is a config change rather than a different class.
        use_temperature: ablation hook for the temperature branch.

    Shape:
        input ``batch`` -> output ``[N_res, out_channels]``, invariant.

    Masked residues (nonstandard type, missing frame atoms) are zeroed. Zeroing
    is not by itself sufficient -- a zero feature is still a value the network
    can key on -- so the residue mask is carried forward and applied again in the
    losses.
    """

    def __init__(
        self,
        *,
        plm_dim: int = ESM2_EMBED_DIM,
        plm_projection_dim: int = 128,
        out_channels: int = 64,
        num_residue_types: int = NUM_RESIDUE_TYPES,
        type_embedding_dim: int = 32,
        use_plm: bool = True,
        use_temperature: bool = True,
    ):
        super().__init__()
        self.use_plm = use_plm
        self.use_temperature = use_temperature
        self.plm_dim = plm_dim
        self.out_channels = out_channels

        self.type_embedding = nn.Embedding(num_residue_types, type_embedding_dim)

        in_dim = type_embedding_dim
        if use_plm:
            # LayerNorm first: ESM-2 hidden states are not scale-controlled and
            # their magnitude varies by layer, which would otherwise dominate the
            # concatenation.
            self.plm_norm = nn.LayerNorm(plm_dim)
            self.plm_projection = nn.Linear(plm_dim, plm_projection_dim)
            in_dim += plm_projection_dim
        if use_temperature:
            in_dim += TEMPERATURE_FEATURE_DIM

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, batch: HierarchicalProteinBatch) -> Tensor:
        """Returns ``[N_res, out_channels]`` invariant residue scalars."""
        residues = batch.residues
        parts = [self.type_embedding(residues.residue_type)]

        if self.use_plm:
            if residues.plm_embedding.shape[1] != self.plm_dim:
                raise ValueError(
                    f"plm_embedding has width {residues.plm_embedding.shape[1]} but "
                    f"this conditioner was built for {self.plm_dim}. Rebuild the "
                    "cache or construct the model with plm_dim set to match."
                )
            parts.append(self.plm_projection(self.plm_norm(residues.plm_embedding)))

        if self.use_temperature:
            per_graph = temperature_features(batch.temperature)
            parts.append(per_graph[residues.batch_index])

        features = self.mlp(torch.cat(parts, dim=-1))
        return features * residues.mask.unsqueeze(-1).to(features.dtype)
