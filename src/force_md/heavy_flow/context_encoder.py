"""Stage 2 heavy-flow context encoder.

This module deliberately composes the Stage 1 geometry encoder with the new
hierarchy.  It does not import or connect any H0-H2, frame-transition,
physics-slot, or teacher components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..data.residue_constants import NUM_ATOM_NAMES
from .atom_context import AtomContextOutput, AtomContextRefiner
from .equivariant_pooling import EquivariantAtomToResiduePool
from .esmc_encoder import ESMCConfig, ESMCEncoder
from .geometry_encoder import HeavyFlowAtomEncoder, HeavyFlowGeometryConfig, HeavyFlowGeometryEncoding
from .seq_geo_fusion import SeqGeoResidueContext, SequenceGeometryFusion, equivariant_to_invariant, invariant_width
from .atom_graph import HeavyFlowGraphConfig
from .types import HeavyFlowCondition

__all__ = ["HeavyFlowContextConfig", "HeavyFlowContextEncoding", "HeavyFlowContextEncoder"]


@dataclass(frozen=True)
class HeavyFlowContextConfig:
    joint_scalar_dim: int = 384
    fusion_blocks: int = 4
    attention_heads: int = 8
    pair_bias: bool = True
    dropout: float = 0.1
    geometry_modality_dropout: float = 0.0
    atom_refine_blocks: int = 2
    atom_refine_lmax: int = 2
    atom_refine_radial_basis: int = 16
    atom_refine_edge_embedding_dim: int = 8
    atom_refine_edge_chunk_size: int = 2048
    atom_refine_activation_checkpoint: bool = True
    attention_hidden_dim: int = 64
    identity_embedding_dim: int = 8


@dataclass(frozen=True)
class HeavyFlowContextEncoding:
    geometry: HeavyFlowGeometryEncoding
    pooling: object
    residue_context: SeqGeoResidueContext
    atom_context: AtomContextOutput

    @property
    def atom_features(self) -> Tensor:
        return self.atom_context.atom_features

    @property
    def padded_atom_features(self) -> Tensor:
        return self.atom_context.padded_features

    @property
    def trainable_parameter_count(self) -> int:
        return 0


class HeavyFlowContextEncoder(nn.Module):
    """Build ``H_local -> G_geo -> C_res -> C_atom``."""

    def __init__(
        self,
        *,
        geometry_encoder: Optional[HeavyFlowAtomEncoder] = None,
        geometry_config: Optional[HeavyFlowGeometryConfig] = None,
        graph_config: Optional[HeavyFlowGraphConfig] = None,
        sequence_encoder: Optional[ESMCEncoder] = None,
        esmc_config: Optional[ESMCConfig] = None,
        context_config: Optional[HeavyFlowContextConfig] = None,
    ):
        super().__init__()
        self.context_config = context_config or HeavyFlowContextConfig()
        self.geometry_encoder = geometry_encoder or HeavyFlowAtomEncoder(
            geometry_config=geometry_config,
            graph_config=graph_config,
            sequence_encoder=sequence_encoder,
            esmc_config=esmc_config,
        )
        self.graph_config = self.geometry_encoder.graph_config
        identity_dim = 3 * self.context_config.identity_embedding_dim + 6
        self.identity_dim = identity_dim
        embed_dim = self.context_config.identity_embedding_dim
        self.atom_type_embedding = nn.Embedding(self.geometry_encoder.geometry_config.max_atomic_number + 1, embed_dim)
        self.atom_name_embedding = nn.Embedding(NUM_ATOM_NAMES, embed_dim)
        self.residue_embedding = nn.Embedding(32, embed_dim)
        self.pool = EquivariantAtomToResiduePool(
            self.geometry_encoder.output_irreps,
            attention_input_dim=invariant_width(self.geometry_encoder.output_irreps) + identity_dim,
            attention_hidden_dim=self.context_config.attention_hidden_dim,
        )
        self.fusion = SequenceGeometryFusion(
            geometry_irreps=self.geometry_encoder.output_irreps,
            joint_scalar_dim=self.context_config.joint_scalar_dim,
            fusion_blocks=self.context_config.fusion_blocks,
            attention_heads=self.context_config.attention_heads,
            pair_bias=self.context_config.pair_bias,
            dropout=self.context_config.dropout,
            spatial_contact_cutoff=self.graph_config.spatial_cutoff_angstrom,
            geometry_modality_dropout=self.context_config.geometry_modality_dropout,
        )
        self.refiner = AtomContextRefiner(
            local_irreps=self.geometry_encoder.output_irreps,
            joint_scalar_dim=self.context_config.joint_scalar_dim,
            identity_dim=identity_dim,
            output_irreps=self.geometry_encoder.output_irreps,
            num_blocks=self.context_config.atom_refine_blocks,
            lmax=self.context_config.atom_refine_lmax,
            radial_basis=self.context_config.atom_refine_radial_basis,
            edge_embedding_dim=self.context_config.atom_refine_edge_embedding_dim,
            spatial_cutoff=self.graph_config.spatial_cutoff_angstrom,
            edge_chunk_size=self.context_config.atom_refine_edge_chunk_size,
            activation_checkpoint=self.context_config.atom_refine_activation_checkpoint,
        )

    @property
    def output_irreps(self) -> o3.Irreps:
        return self.refiner.output_irreps

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _chain_flags(self, condition: HeavyFlowCondition, batch: Tensor, residue: Tensor) -> Tensor:
        before = torch.zeros_like(condition.residue_mask, dtype=torch.bool)
        after = torch.zeros_like(condition.residue_mask, dtype=torch.bool)
        if condition.chain_break is not None and condition.sequence_length > 1:
            after[:, :-1] = condition.chain_break
            before[:, 1:] = condition.chain_break
        return torch.stack((before[batch, residue], after[batch, residue]), dim=-1)

    def _atom_identity_features(self, condition: HeavyFlowCondition, graph) -> Tensor:
        state = graph.state
        batch, local = state.batch, state.local_atom
        residue = condition.atom_to_residue[batch, local]
        atom_type = condition.atom_type[batch, local].clamp(0, self.geometry_encoder.geometry_config.max_atomic_number)
        atom_name = condition.atom_name[batch, local].clamp(0, NUM_ATOM_NAMES - 1)
        residue_type = condition.sequence_tokens[batch, residue].clamp(0, 31)
        if condition.is_backbone is None:
            backbone = torch.zeros_like(atom_type, dtype=torch.bool)
        else:
            backbone = condition.is_backbone[batch, local]
        if condition.is_sidechain is None:
            sidechain = ~backbone
        else:
            sidechain = condition.is_sidechain[batch, local]
        if condition.terminal_residue is None:
            terminal = torch.zeros_like(backbone)
        else:
            terminal = condition.terminal_residue[batch, residue]
        chain = self._chain_flags(condition, batch, residue)
        flags = torch.stack(
            (backbone, sidechain, terminal, chain[:, 0], chain[:, 1], condition.atom_mask[batch, local]), dim=-1
        ).to(self.atom_type_embedding.weight.dtype)
        return torch.cat(
            (
                self.atom_type_embedding(atom_type),
                self.atom_name_embedding(atom_name),
                self.residue_embedding(residue_type),
                flags,
            ),
            dim=-1,
        )

    def forward(
        self,
        condition: HeavyFlowCondition,
        *,
        geometry_dropout: bool = False,
    ) -> HeavyFlowContextEncoding:
        condition.validate()
        if hasattr(self, "required_history") and condition.history_length != self.required_history.length:
            raise ValueError("condition history length differs from upstream config/checkpoint")
        geometry = self.geometry_encoder(condition)
        state = geometry.graph.state
        identity = self._atom_identity_features(condition, geometry.graph)
        atom_invariant = equivariant_to_invariant(geometry.atom_features, geometry.irreps)
        attention_features = torch.cat((atom_invariant, identity), dim=-1)
        pooling = self.pool(
            geometry.atom_features,
            atom_batch=state.batch,
            atom_to_residue=condition.atom_to_residue[state.batch, state.local_atom],
            residue_mask=condition.residue_mask,
            attention_features=attention_features,
            atom_mask=torch.ones(state.current.shape[0], dtype=torch.bool, device=state.current.device),
            atom_positions=state.current,
        )
        residue_context = self.fusion(
            pooling,
            chain_break=condition.chain_break,
            geometry_dropout=geometry_dropout,
        )
        atom_context = self.refiner(
            geometry.atom_features,
            geometry.graph,
            condition,
            residue_context.joint_scalar,
            identity,
        )
        return HeavyFlowContextEncoding(geometry, pooling, residue_context, atom_context)
