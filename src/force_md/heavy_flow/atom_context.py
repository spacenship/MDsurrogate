"""Residue-to-atom broadcast and equivariant atom refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .atom_graph import HeavyFlowAtomGraph
from .geometry_encoder import (
    _EquivariantNormActivation,
    _RadialBasis,
    _tensor_product_aggregate,
)
from .types import HeavyFlowCondition

__all__ = ["AtomContextOutput", "AtomContextRefiner"]


@dataclass(frozen=True)
class AtomContextOutput:
    atom_features: Tensor  # [M, D], packed in Stage 1 graph order
    padded_features: Tensor  # [B, N, D], zero on masked atoms
    irreps: o3.Irreps
    broadcast_residue: Tensor  # [M, joint_scalar_dim]
    atom_mask: Tensor  # [B, N]
    atom_batch: Tensor  # [M]
    local_atom: Tensor  # [M]

    @property
    def parameter_width(self) -> int:
        return self.irreps.dim


class _AtomRefinementBlock(nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        edge_width: int,
        hidden_edge: int,
        *,
        edge_chunk_size: int = 2048,
        activation_checkpoint: bool = True,
    ):
        super().__init__()
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps, sh_irreps, irreps, shared_weights=False, internal_weights=False
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_width, hidden_edge), nn.SiLU(),
            nn.Linear(hidden_edge, self.tp.weight_numel),
        )
        self.node_linear = o3.Linear(irreps, irreps)
        self.message_linear = o3.Linear(irreps, irreps)
        self.norm = _EquivariantNormActivation(irreps)

    def _forward_impl(
        self,
        node: Tensor,
        edge_index: Tensor,
        edge_sh: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        aggregate = _tensor_product_aggregate(
            node,
            edge_index,
            edge_sh,
            edge_features,
            tensor_product=self.tp,
            edge_mlp=self.edge_mlp,
            edge_chunk_size=self.edge_chunk_size,
        )
        degree = torch.bincount(edge_index[1], minlength=node.shape[0]).to(node.dtype).clamp_min(1.0)
        aggregate = aggregate / degree.sqrt()[:, None]
        return self.norm(node + self.node_linear(node) + self.message_linear(aggregate))

    def forward(self, node: Tensor, graph: HeavyFlowAtomGraph, edge_sh: Tensor, edge_features: Tensor) -> Tensor:
        args = (node, graph.edge_index, edge_sh, edge_features)
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, *args, use_reentrant=False)
        return self._forward_impl(*args)


class AtomContextRefiner(nn.Module):
    """Fuse broadcast residue context with local Stage 1 atom geometry."""

    def __init__(
        self,
        *,
        local_irreps: o3.Irreps | str,
        joint_scalar_dim: int,
        identity_dim: int,
        output_irreps: Optional[o3.Irreps | str] = None,
        num_blocks: int = 2,
        lmax: int = 2,
        radial_basis: int = 16,
        edge_embedding_dim: int = 8,
        spatial_cutoff: float = 6.0,
        edge_chunk_size: int = 2048,
        activation_checkpoint: bool = True,
    ):
        super().__init__()
        if num_blocks < 1 or joint_scalar_dim < 1 or identity_dim < 1:
            raise ValueError("num_blocks, joint_scalar_dim, and identity_dim must be positive")
        self.local_irreps = o3.Irreps(local_irreps)
        self.output_irreps = o3.Irreps(output_irreps or local_irreps)
        self.joint_scalar_dim = joint_scalar_dim
        self.identity_dim = identity_dim
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.radial = _RadialBasis(radial_basis, spatial_cutoff)
        self.edge_kind_embedding = nn.Embedding(3, edge_embedding_dim)
        self.bond_type_embedding = nn.Embedding(32, edge_embedding_dim)
        input_irreps = self.local_irreps + o3.Irreps(f"{joint_scalar_dim}x0e") + o3.Irreps(f"{identity_dim}x0e")
        self.node_init = o3.Linear(input_irreps, self.output_irreps)
        self.local_skip = o3.Linear(self.local_irreps, self.output_irreps)
        edge_width = radial_basis + 2 * edge_embedding_dim
        self.blocks = nn.ModuleList(
            _AtomRefinementBlock(
                self.output_irreps,
                self.sh_irreps,
                edge_width,
                max(64, self.output_irreps.dim // 2),
                edge_chunk_size=self.edge_chunk_size,
                activation_checkpoint=self.activation_checkpoint,
            )
            for _ in range(num_blocks)
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _edge_features(self, graph: HeavyFlowAtomGraph, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if graph.edge_index.shape[1] == 0:
            width = self.radial.centers.numel() + 2 * self.edge_kind_embedding.embedding_dim
            return (
                torch.empty((0, width), dtype=dtype, device=graph.state.current.device),
                torch.empty((0, self.sh_irreps.dim), dtype=dtype, device=graph.state.current.device),
            )
        vectors = graph.state.current[graph.edge_index[1]] - graph.state.current[graph.edge_index[0]]
        distances = vectors.norm(dim=-1)
        edge_sh = o3.spherical_harmonics(
            list(range(self.sh_irreps.lmax + 1)), vectors,
            normalize=True, normalization="component",
        ).to(dtype)
        radial = self.radial(distances, graph.edge_kind).to(dtype)
        kind = self.edge_kind_embedding(graph.edge_kind)
        bond = self.bond_type_embedding(graph.bond_type.clamp(0, 31))
        return torch.cat((radial, kind, bond), dim=-1), edge_sh

    def forward(
        self,
        local_atom_features: Tensor,
        graph: HeavyFlowAtomGraph,
        condition: HeavyFlowCondition,
        joint_scalar: Tensor,
        atom_identity_features: Tensor,
    ) -> AtomContextOutput:
        condition.validate()
        state = graph.state
        if local_atom_features.ndim != 2 or local_atom_features.shape[0] != state.current.shape[0]:
            raise ValueError("local_atom_features must align with packed graph atoms")
        if local_atom_features.shape[-1] != self.local_irreps.dim:
            raise ValueError("local_atom_features width does not match local_irreps")
        if joint_scalar.shape[:2] != condition.residue_mask.shape or joint_scalar.shape[-1] != self.joint_scalar_dim:
            raise ValueError("joint_scalar must have shape [B, L, joint_scalar_dim]")
        if atom_identity_features.shape != (state.current.shape[0], self.identity_dim):
            raise ValueError("atom_identity_features has an invalid shape")
        residue = condition.atom_to_residue[state.batch, state.local_atom]
        broadcast = joint_scalar[state.batch, residue]
        node_input = torch.cat((local_atom_features, broadcast, atom_identity_features), dim=-1)
        node = self.node_init(node_input) + self.local_skip(local_atom_features)
        edge_features, edge_sh = self._edge_features(graph, node.dtype)
        for block in self.blocks:
            node = block(node, graph, edge_sh, edge_features)
        padded = node.new_zeros((condition.batch_size, condition.max_atoms, node.shape[-1]))
        for batch in range(condition.batch_size):
            local = torch.nonzero(condition.atom_mask[batch], as_tuple=False).flatten()
            if local.numel():
                packed = graph.state.packed_index[batch, local]
                padded[batch, local] = node[packed]
        return AtomContextOutput(
            atom_features=node,
            padded_features=padded,
            irreps=self.output_irreps,
            broadcast_residue=broadcast,
            atom_mask=condition.atom_mask,
            atom_batch=state.batch,
            local_atom=state.local_atom,
        )
