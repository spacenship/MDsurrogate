"""Invariant geometry residue tokens and global self-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from .equivariant_pooling import EquivariantResiduePoolingOutput, irrep_chunks

__all__ = [
    "SeqGeoResidueContext",
    "SequenceGeometryFusion",
    "ResidueGlobalContext",
    "equivariant_to_invariant",
    "invariant_width",
]


def invariant_width(irreps) -> int:
    """Width of the scalar invariant summary used by the Transformer."""

    irreps = o3.Irreps(irreps)
    return sum(multiplicity for multiplicity, irrep, _ in irrep_chunks(irreps))


def equivariant_to_invariant(features: Tensor, irreps) -> Tensor:
    """Convert e3nn channels to rotation-invariant scalar summaries.

    Scalar channels are retained. Every non-scalar multiplicity contributes
    its channel norm; no x/y/z component is exposed to the standard attention
    stack.
    """

    irreps = o3.Irreps(irreps)
    if features.shape[-1] != irreps.dim:
        raise ValueError("feature width does not match irreps")
    values: list[Tensor] = []
    for multiplicity, irrep, sl in irrep_chunks(irreps):
        chunk = features[..., sl].reshape(*features.shape[:-1], multiplicity, irrep.dim)
        if irrep.l == 0:
            # Stage 1 uses even scalars. For a possible odd scalar, its
            # absolute value is the parity-safe invariant representation.
            values.append(chunk.squeeze(-1) if irrep.p == 1 else chunk.abs().squeeze(-1))
        else:
            # Match the geometry norm convention (mean over irrep
            # components) while avoiding an undefined derivative for an
            # exactly zero equivariant channel.  Subtracting epsilon keeps a
            # zero feature mapped to zero; the norm arithmetic itself stays
            # in fp32 under AMP.
            h32 = chunk.float()
            sq_norm = h32.square().mean(dim=-1)
            eps_norm = 1e-4
            stable_norm = torch.sqrt(sq_norm + eps_norm ** 2) - eps_norm
            values.append(stable_norm.to(dtype=features.dtype))
    return torch.cat(values, dim=-1) if values else features.new_empty((*features.shape[:-1], 0))


@dataclass(frozen=True)
class SeqGeoResidueContext:
    """Invariant geometry context broadcast back to atoms."""

    geometry_invariant: Tensor  # [B, L, I]
    geometry_equivariant: Tensor  # [B, L, D_geo]
    joint_scalar: Tensor  # [B, L, joint_scalar_dim]
    residue_mask: Tensor  # [B, L]
    pair_bias: Optional[Tensor] = None  # [B, heads, L, L]
    residue_centers: Optional[Tensor] = None  # [B, L, 3]


class _PairBiasedSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("joint_scalar_dim must be divisible by attention_heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor, pair_bias: Optional[Tensor], residue_mask: Tensor) -> Tensor:
        batch, length, _ = query.shape
        q = self.query(query).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = self.key(key_value).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = self.value(key_value).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_width ** 0.5)
        if pair_bias is not None:
            scores = scores + pair_bias
        key_mask = ~residue_mask[:, None, None, :]
        scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, self.width)
        attended = self.output(attended)
        return attended * residue_mask[..., None].to(attended.dtype)


class _FusionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm_query = nn.LayerNorm(width)
        self.self_attention = _PairBiasedSelfAttention(width, heads, dropout)
        self.norm_mlp = nn.LayerNorm(width)
        hidden = width * 4
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, pair_bias: Optional[Tensor], residue_mask: Tensor) -> Tensor:
        normalized = self.norm_query(values)
        attended = self.self_attention(normalized, normalized, pair_bias, residue_mask)
        values = values + self.dropout(attended)
        values = values + self.dropout(self.mlp(self.norm_mlp(values)))
        return values * residue_mask[..., None].to(values.dtype)


class ResidueGlobalContext(nn.Module):
    """Self-attention over pooled invariant geometry; no second ESM input."""

    def __init__(
        self,
        *,
        geometry_irreps,
        joint_scalar_dim: int = 384,
        fusion_blocks: int = 4,
        attention_heads: int = 8,
        pair_bias: bool = True,
        dropout: float = 0.1,
        pair_rbf_count: int = 16,
        spatial_contact_cutoff: float = 6.0,
        geometry_modality_dropout: float = 0.0,
    ):
        super().__init__()
        geometry_irreps = o3.Irreps(geometry_irreps)
        if fusion_blocks < 1 or attention_heads < 1 or joint_scalar_dim < 1:
            raise ValueError("fusion_blocks, attention_heads, and joint_scalar_dim must be positive")
        if not 0.0 <= geometry_modality_dropout <= 1.0:
            raise ValueError("modality dropout must be in [0, 1]")
        self.geometry_irreps = geometry_irreps
        self.geometry_input_irreps = o3.Irreps(f"{invariant_width(geometry_irreps)}x0e")
        self.geometry_invariant_dim = invariant_width(geometry_irreps)
        self.joint_scalar_dim = joint_scalar_dim
        self.attention_heads = attention_heads
        self.use_pair_bias = pair_bias
        self.dropout_probability = dropout
        self.geometry_modality_dropout = geometry_modality_dropout
        self.spatial_contact_cutoff = float(spatial_contact_cutoff)
        self.geometry_projection = nn.Linear(self.geometry_invariant_dim, joint_scalar_dim)
        self.blocks = nn.ModuleList(
            _FusionBlock(joint_scalar_dim, attention_heads, dropout) for _ in range(fusion_blocks)
        )
        if pair_bias:
            self.pair_rbf_count = pair_rbf_count
            self.pair_bias_network = nn.Sequential(
                nn.Linear(pair_rbf_count + 5, max(32, joint_scalar_dim // 4)),
                nn.SiLU(),
                nn.Linear(max(32, joint_scalar_dim // 4), attention_heads),
            )
        else:
            self.pair_rbf_count = 0
            self.pair_bias_network = None

    def _pair_features(
        self,
        chain_break: Optional[Tensor],
        residue_mask: Tensor,
        residue_centers: Optional[Tensor],
        *,
        geometry_dropped: bool,
    ) -> Tensor:
        batch, length = residue_mask.shape
        device = residue_mask.device
        dtype = residue_centers.dtype if residue_centers is not None else torch.float32
        index = torch.arange(length, device=device)
        separation = (index[None, :, None] - index[None, None, :]).abs().to(dtype)
        # Normalize each protein independently of padding in other batch rows.
        extent = ((index[None, :] + 1) * residue_mask).amax(dim=1).sub(1).clamp_min(1)
        raw_separation = separation.expand(batch, -1, -1)
        separation = raw_separation / extent[:, None, None].to(dtype)
        active_pair = residue_mask[:, :, None] & residue_mask[:, None, :]
        if chain_break is None:
            chain_ids = torch.zeros((batch, length), dtype=torch.long, device=device)
        else:
            breaks = chain_break.to(device=device, dtype=torch.long)
            chain_ids = torch.cat(
                (torch.zeros((batch, 1), dtype=torch.long, device=device), breaks.cumsum(dim=1)), dim=1
            )
        same_chain = active_pair & (chain_ids[:, :, None] == chain_ids[:, None, :])
        adjacency = (raw_separation == 1).to(dtype) * same_chain.to(dtype)
        chain_break_pair = (raw_separation == 1).to(dtype) * (~same_chain).to(dtype)

        if residue_centers is None or geometry_dropped:
            distance = torch.zeros((batch, length, length), dtype=dtype, device=device)
        else:
            distance = torch.cdist(residue_centers.to(dtype), residue_centers.to(dtype))
            distance = distance.masked_fill(~active_pair, 0.0)
        centers = torch.linspace(0.0, self.spatial_contact_cutoff, self.pair_rbf_count, device=device, dtype=dtype)
        width = self.spatial_contact_cutoff / max(self.pair_rbf_count - 1, 1)
        rbf = torch.exp(-((distance[..., None] - centers) / width) ** 2)
        contact = (distance <= self.spatial_contact_cutoff).to(dtype) * active_pair.to(dtype)
        features = torch.cat(
            (
                separation[..., None], same_chain.to(dtype)[..., None], adjacency[..., None],
                contact[..., None], chain_break_pair[..., None], rbf,
            ), dim=-1
        )
        return features

    def _pair_bias(
        self,
        chain_break: Optional[Tensor],
        residue_mask: Tensor,
        residue_centers: Optional[Tensor],
        *,
        geometry_dropped: bool,
    ) -> Optional[Tensor]:
        if self.pair_bias_network is None:
            return None
        features = self._pair_features(chain_break, residue_mask, residue_centers, geometry_dropped=geometry_dropped)
        bias = self.pair_bias_network(features).permute(0, 3, 1, 2)
        active = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
        return bias.masked_fill(~active, 0.0)

    def forward(
        self,
        pooling: EquivariantResiduePoolingOutput,
        *,
        chain_break: Optional[Tensor] = None,
        geometry_dropout: bool = False,
    ) -> SeqGeoResidueContext:
        residue_mask = pooling.residue_mask
        if self.training and self.geometry_modality_dropout:
            geometry_dropout = bool(torch.rand((), device=residue_mask.device) < self.geometry_modality_dropout)
        geometry_invariant = equivariant_to_invariant(pooling.padded_features, pooling.irreps)
        geometry_used = torch.zeros_like(geometry_invariant) if geometry_dropout else geometry_invariant
        values = self.geometry_projection(geometry_used) * residue_mask[..., None]
        pair = self._pair_bias(
            chain_break, residue_mask, pooling.residue_centers, geometry_dropped=geometry_dropout
        ) if self.use_pair_bias else None
        for block in self.blocks:
            values = block(values, pair, residue_mask)
        return SeqGeoResidueContext(
            geometry_invariant=geometry_invariant,
            geometry_equivariant=pooling.padded_features,
            joint_scalar=values,
            residue_mask=residue_mask,
            pair_bias=pair,
            residue_centers=pooling.residue_centers,
        )


# Import compatibility only; checkpoint compatibility is versioned separately.
SequenceGeometryFusion = ResidueGlobalContext
