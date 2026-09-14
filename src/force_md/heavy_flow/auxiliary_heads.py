"""Auxiliary seq--geometry objectives for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .atom_graph import COVALENT_EDGE
from .context_encoder import HeavyFlowContextEncoding
from .seq_geo_fusion import equivariant_to_invariant, invariant_width
from .types import HeavyFlowCondition

__all__ = ["AuxiliaryHeadOutput", "HeavyFlowAuxiliaryHeads"]


@dataclass(frozen=True)
class AuxiliaryHeadOutput:
    """Predictions, clean targets, and masks for two independent objectives."""

    residue_geometry_prediction: Tensor  # [B, L, I]
    residue_geometry_target: Tensor  # [B, L, I]
    residue_geometry_mask: Tensor  # [B, L]
    bond_distance_prediction: Tensor  # [E_cov]
    bond_distance_target: Tensor  # [E_cov]
    bond_distance_mask: Tensor  # [E_cov]

    def losses(self) -> dict[str, Tensor]:
        if self.residue_geometry_mask.any():
            residue = (
                self.residue_geometry_prediction[self.residue_geometry_mask]
                - self.residue_geometry_target[self.residue_geometry_mask]
            ).square().mean()
        else:
            residue = self.residue_geometry_prediction.sum() * 0.0
        if self.bond_distance_mask.any():
            bond = (
                self.bond_distance_prediction[self.bond_distance_mask]
                - self.bond_distance_target[self.bond_distance_mask]
            ).square().mean()
        else:
            bond = self.bond_distance_prediction.sum() * 0.0
        return {
            "masked_residue_geometry": residue,
            "bond_distance_denoising": bond,
            "total": residue + bond,
        }


class HeavyFlowAuxiliaryHeads(nn.Module):
    """Masked residue-token reconstruction and bonded-distance denoising.

    The caller can pass a context built from noised coordinates and a clean
    ``target_encoding``. No force or future-state target is consumed here.
    """

    def __init__(self, *, atom_irreps, residue_joint_dim: int, geometry_invariant_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.atom_irreps = atom_irreps
        self.atom_invariant_dim = invariant_width(atom_irreps)
        self.geometry_invariant_dim = geometry_invariant_dim
        self.residue_geometry_head = nn.Sequential(
            nn.Linear(residue_joint_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, geometry_invariant_dim)
        )
        bond_input = 2 * self.atom_invariant_dim + 1
        self.bond_distance_head = nn.Sequential(
            nn.Linear(bond_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _default_residue_mask(residue_mask: Tensor) -> Tensor:
        indices = torch.arange(residue_mask.shape[1], device=residue_mask.device)
        selected = residue_mask & (indices[None, :] % 2 == 0)
        for batch in range(residue_mask.shape[0]):
            if residue_mask[batch].any() and not selected[batch].any():
                selected[batch, torch.nonzero(residue_mask[batch], as_tuple=False)[0, 0]] = True
        return selected

    def forward(
        self,
        encoding: HeavyFlowContextEncoding,
        condition: HeavyFlowCondition,
        *,
        target_encoding: Optional[HeavyFlowContextEncoding] = None,
        residue_geometry_mask: Optional[Tensor] = None,
    ) -> AuxiliaryHeadOutput:
        condition.validate()
        target = target_encoding or encoding
        if residue_geometry_mask is None:
            residue_geometry_mask = self._default_residue_mask(condition.residue_mask)
        residue_geometry_mask = residue_geometry_mask.to(device=condition.residue_mask.device, dtype=torch.bool)
        if residue_geometry_mask.shape != condition.residue_mask.shape:
            raise ValueError("residue_geometry_mask must have shape [B, L]")

        # Mask the input token at the selected residue. Its target remains the
        # clean pooled geometry, while unmasked neighboring context is retained.
        masked_joint = encoding.residue_context.joint_scalar.masked_fill(residue_geometry_mask[..., None], 0.0)
        residue_prediction = self.residue_geometry_head(masked_joint)
        residue_prediction = residue_prediction * condition.residue_mask[..., None].to(residue_prediction.dtype)
        residue_target = target.residue_context.geometry_invariant.detach()

        graph = encoding.geometry.graph
        target_graph = target.geometry.graph
        covalent = graph.edge_kind == COVALENT_EDGE
        edge_index = graph.edge_index[:, covalent]
        if edge_index.shape[1]:
            atom_invariant = equivariant_to_invariant(encoding.atom_context.atom_features, encoding.atom_context.irreps)
            left = atom_invariant[edge_index[0]]
            right = atom_invariant[edge_index[1]]
            distance = (
                graph.state.current[edge_index[1]] - graph.state.current[edge_index[0]]
            ).norm(dim=-1, keepdim=True)
            bond_prediction = self.bond_distance_head(torch.cat((left, right, distance), dim=-1)).squeeze(-1)
            target_edge_index = target_graph.edge_index[:, target_graph.edge_kind == COVALENT_EDGE]
            if target_edge_index.shape[1] != edge_index.shape[1]:
                raise ValueError("clean/noised encodings have different covalent edge topology")
            bond_target = (
                target_graph.state.current[target_edge_index[1]]
                - target_graph.state.current[target_edge_index[0]]
            ).norm(dim=-1)
        else:
            bond_prediction = encoding.atom_context.atom_features.new_empty((0,))
            bond_target = bond_prediction.detach().clone()
        return AuxiliaryHeadOutput(
            residue_geometry_prediction=residue_prediction,
            residue_geometry_target=residue_target,
            residue_geometry_mask=residue_geometry_mask & condition.residue_mask,
            bond_distance_prediction=bond_prediction,
            bond_distance_target=bond_target,
            bond_distance_mask=torch.ones_like(bond_prediction, dtype=torch.bool),
        )

    def loss(
        self,
        encoding: HeavyFlowContextEncoding,
        condition: HeavyFlowCondition,
        *,
        target_encoding: Optional[HeavyFlowContextEncoding] = None,
        residue_geometry_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        return self(
            encoding,
            condition,
            target_encoding=target_encoding,
            residue_geometry_mask=residue_geometry_mask,
        ).losses()
