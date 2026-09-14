"""Equivariant atom-to-residue pooling for the clean-slate heavy-flow path.

The pooling weights are scalars. Consequently a weighted sum commutes with
every representation carried by an e3nn feature: scalar, polar/axial vector,
and higher-order tensor channels are pooled without flattening or changing
their parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

__all__ = ["EquivariantResiduePoolingOutput", "EquivariantAtomToResiduePool", "irrep_chunks"]


def irrep_chunks(irreps: o3.Irreps) -> list[tuple[int, o3.Irrep, slice]]:
    """Return multiplicity/irrep/slice triples in the flat feature layout."""

    irreps = o3.Irreps(irreps)
    chunks: list[tuple[int, o3.Irrep, slice]] = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dim
        chunks.append((multiplicity, irrep, slice(offset, offset + width)))
        offset += width
    return chunks


@dataclass(frozen=True)
class EquivariantResiduePoolingOutput:
    """Packed and padded residue features plus the pooling bookkeeping."""

    residue_features: Tensor  # [R, D], active residues in batch-local order
    padded_features: Tensor  # [B, L, D], zero on inactive residues
    irreps: o3.Irreps
    residue_batch: Tensor  # [R]
    local_residue: Tensor  # [R]
    residue_mask: Tensor  # [B, L]
    attention_weights: Tensor  # [M], zero for masked atoms
    residue_centers: Optional[Tensor] = None  # [B, L, 3]

    @property
    def num_residues(self) -> int:
        return int(self.residue_features.shape[0])


class EquivariantAtomToResiduePool(nn.Module):
    """Pool packed atom features into residue slots with masked attention.

    ``attention_features`` must contain only rotation-invariant scalar data,
    such as Stage 1 scalar channels concatenated with element/atom-name
    embeddings. The feature being pooled is kept in its original irreps.
    """

    def __init__(self, irreps: o3.Irreps | str, attention_input_dim: int, attention_hidden_dim: int = 64):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if attention_input_dim <= 0 or attention_hidden_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        self.attention_mlp = nn.Sequential(
            nn.Linear(attention_input_dim, attention_hidden_dim), nn.SiLU(), nn.Linear(attention_hidden_dim, 1)
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        atom_batch: Tensor,
        atom_to_residue: Tensor,
        residue_mask: Tensor,
        attention_features: Tensor,
        atom_mask: Optional[Tensor] = None,
        atom_positions: Optional[Tensor] = None,
    ) -> EquivariantResiduePoolingOutput:
        if atom_features.ndim != 2 or atom_features.shape[-1] != self.irreps.dim:
            raise ValueError("atom_features must have shape [M, irreps.dim]")
        if atom_batch.ndim != 1 or atom_to_residue.ndim != 1:
            raise ValueError("atom_batch and atom_to_residue must be one-dimensional")
        if atom_batch.shape[0] != atom_features.shape[0] or atom_to_residue.shape[0] != atom_features.shape[0]:
            raise ValueError("packed atom metadata does not match atom_features")
        if attention_features.shape[:1] != atom_features.shape[:1]:
            raise ValueError("attention_features must have one row per packed atom")
        if residue_mask.ndim != 2 or residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must be a [B, L] boolean tensor")
        if atom_mask is None:
            atom_mask = torch.ones(atom_features.shape[0], dtype=torch.bool, device=atom_features.device)
        else:
            atom_mask = atom_mask.to(device=atom_features.device, dtype=torch.bool)
        if atom_mask.shape != (atom_features.shape[0],):
            raise ValueError("packed atom_mask must have shape [M]")
        atom_batch = atom_batch.to(device=atom_features.device, dtype=torch.long)
        atom_to_residue = atom_to_residue.to(device=atom_features.device, dtype=torch.long)
        residue_mask = residue_mask.to(device=atom_features.device, dtype=torch.bool)
        batch_size, sequence_length = residue_mask.shape

        active_residues = torch.nonzero(residue_mask, as_tuple=False)
        residue_batch = active_residues[:, 0]
        local_residue = active_residues[:, 1]
        residue_count = active_residues.shape[0]
        lookup = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=atom_features.device)
        if residue_count:
            lookup[residue_batch, local_residue] = torch.arange(residue_count, device=lookup.device)

        # Masked rows may carry sentinel metadata, so clamp only for indexing;
        # invalid rows are still excluded by ``atom_mask`` below.
        safe_batch = atom_batch.clamp(0, max(batch_size - 1, 0))
        safe_residue = atom_to_residue.clamp(0, max(sequence_length - 1, 0))
        group_for_atom = lookup[safe_batch, safe_residue]
        valid = atom_mask
        if valid.any() and (group_for_atom[valid] < 0).any():
            raise ValueError("an active atom maps to an inactive or out-of-range residue")

        logits = self.attention_mlp(attention_features).squeeze(-1)
        weights = logits.new_zeros(logits.shape)
        pooled = atom_features.new_zeros((residue_count, self.irreps.dim))
        for residue_id in range(residue_count):
            atom_indices = torch.nonzero(valid & (group_for_atom == residue_id), as_tuple=False).flatten()
            if atom_indices.numel() == 0:
                continue
            local_logits = logits[atom_indices]
            local_weights = torch.softmax(local_logits - local_logits.max(), dim=0)
            weights[atom_indices] = local_weights
            pooled[residue_id] = (atom_features[atom_indices] * local_weights[:, None]).sum(dim=0)

        padded = atom_features.new_zeros((batch_size, sequence_length, self.irreps.dim))
        if residue_count:
            padded[residue_batch, local_residue] = pooled

        centers: Optional[Tensor] = None
        if atom_positions is not None:
            if atom_positions.shape != (atom_features.shape[0], 3):
                raise ValueError("atom_positions must have shape [M, 3]")
            centers = atom_positions.new_zeros((batch_size, sequence_length, 3))
            for residue_id in range(residue_count):
                atom_indices = torch.nonzero(valid & (group_for_atom == residue_id), as_tuple=False).flatten()
                if atom_indices.numel():
                    centers[residue_batch[residue_id], local_residue[residue_id]] = atom_positions[atom_indices].mean(dim=0)

        return EquivariantResiduePoolingOutput(
            residue_features=pooled,
            padded_features=padded,
            irreps=self.irreps,
            residue_batch=residue_batch,
            local_residue=local_residue,
            residue_mask=residue_mask,
            attention_weights=weights,
            residue_centers=centers,
        )
