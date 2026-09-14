"""Stage 3 atom/pair physics predictor.

This module consumes only the Stage 2 ``C_atom`` representation and current
graph geometry.  It has no force-target argument and does not import any of
the legacy ``feasibility_v2`` teacher/oracle modules.  The pair output is a
learned interaction latent, not a unique decomposition of the net atomic
force into pairwise forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from e3nn import o3
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .atom_graph import HeavyFlowAtomGraph
from .context_encoder import HeavyFlowContextEncoding
from .geometry_encoder import (
    _EquivariantNormActivation,
    _RadialBasis,
    _tensor_product_aggregate,
)
from .physics_types import PhysicsState
from .seq_geo_fusion import equivariant_to_invariant, invariant_width
from .types import HeavyFlowCondition

__all__ = [
    "PhysicsPredictorConfig", "PhysicsPredictor", "PhysicsModelOutput",
    "HeavyFlowPhysicsModel",
]


@dataclass(frozen=True)
class PhysicsPredictorConfig:
    """Configuration for the direct atom/pair physics latent.

    The defaults mirror the requested Stage 3 initialization.  ``lmax`` is
    the maximum angular order used by the message-passing edge harmonics;
    output vector channels are kept separate as polar (``1o``) and axial
    (``1e``) fields.
    """

    physics_blocks: int = 3
    lmax: int = 2
    scalar_dim: int = 64
    vector_channels: int = 8
    axial_channels: int = 4
    pair_dim: int = 32
    radial_basis: int = 16
    edge_embedding_dim: int = 8
    spatial_cutoff_angstrom: float = 6.0
    predict_force_mean: bool = True
    predict_isotropic_force_variance: bool = True
    use_pair_latent: bool = True
    edge_chunk_size: int = 2048
    activation_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.physics_blocks < 1 or self.lmax < 0:
            raise ValueError("physics_blocks must be positive and lmax non-negative")
        if min(self.scalar_dim, self.vector_channels, self.axial_channels) < 1:
            raise ValueError("atom latent channel counts must be positive")
        if self.use_pair_latent and self.pair_dim < 1:
            raise ValueError("pair_dim must be positive when use_pair_latent=True")
        if self.radial_basis < 2 or self.edge_embedding_dim < 1:
            raise ValueError("radial_basis must be >=2 and edge embedding positive")
        if self.spatial_cutoff_angstrom <= 0:
            raise ValueError("spatial_cutoff_angstrom must be positive")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")


class _PhysicsMessageBlock(nn.Module):
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


class PhysicsPredictor(nn.Module):
    """Map Stage 2 atom context to an atom/pair physics latent.

    ``forward`` accepts either a :class:`HeavyFlowContextEncoding` or raw
    ``C_atom`` features together with an explicit graph and condition.  Both
    forms have the same boundary: no ground-truth force, future coordinate, or
    teacher latent can enter this module.
    """

    def __init__(
        self,
        atom_irreps: o3.Irreps | str,
        *,
        config: Optional[PhysicsPredictorConfig] = None,
    ):
        super().__init__()
        self.config = config or PhysicsPredictorConfig()
        self.atom_irreps = o3.Irreps(atom_irreps)
        if max((ir.l for _, ir in self.atom_irreps), default=0) < 0:  # pragma: no cover
            raise ValueError("invalid atom irreps")
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.config.lmax)
        self.atom_invariant_dim = invariant_width(self.atom_irreps)
        endpoint_width = min(16, self.atom_invariant_dim)
        self._endpoint_width = endpoint_width
        self.radial = _RadialBasis(
            self.config.radial_basis, self.config.spatial_cutoff_angstrom
        )
        self.edge_kind_embedding = nn.Embedding(3, self.config.edge_embedding_dim)
        self.bond_type_embedding = nn.Embedding(32, self.config.edge_embedding_dim)
        edge_width = (
            self.config.radial_basis
            + 2 * self.config.edge_embedding_dim
            + 2 * endpoint_width
            + (self.config.pair_dim if self.config.use_pair_latent else 0)
        )
        self.blocks = nn.ModuleList(
            _PhysicsMessageBlock(
                self.atom_irreps,
                self.sh_irreps,
                edge_width,
                max(64, self.atom_irreps.dim // 2),
                edge_chunk_size=self.config.edge_chunk_size,
                activation_checkpoint=self.config.activation_checkpoint,
            )
            for _ in range(self.config.physics_blocks)
        )
        self.scalar_projection = o3.Linear(
            self.atom_irreps, o3.Irreps(f"{self.config.scalar_dim}x0e")
        )
        self.vector_projection = o3.Linear(
            self.atom_irreps, o3.Irreps(f"{self.config.vector_channels}x1o")
        )
        self.axial_projection = o3.Linear(
            self.atom_irreps, o3.Irreps(f"{self.config.axial_channels}x1e")
        )
        # Even invariant of axial channels trains their projection through the
        # existing isotropic variance head without introducing a new target.
        self.axial_to_scalar = nn.Linear(self.config.axial_channels, self.config.scalar_dim, bias=False)
        pair_input = self.atom_invariant_dim * 2 + self.config.radial_basis + 2 * self.config.edge_embedding_dim + 1
        if self.config.use_pair_latent:
            self.pair_projection = nn.Sequential(
                nn.Linear(pair_input, max(64, self.config.pair_dim * 2)),
                nn.SiLU(),
                nn.Linear(max(64, self.config.pair_dim * 2), self.config.pair_dim),
            )

    @property
    def output_vector_channels(self) -> int:
        return self.config.vector_channels

    @property
    def output_axial_channels(self) -> int:
        return self.config.axial_channels

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _edge_features(self, graph: HeavyFlowAtomGraph, node: Tensor) -> tuple[Tensor, Tensor]:
        edge_index = graph.edge_index
        edge_width = self.config.radial_basis + 2 * self.config.edge_embedding_dim + 2 * self._endpoint_width
        if edge_index.shape[1] == 0:
            return (
                node.new_empty((0, edge_width)),
                node.new_empty((0, self.sh_irreps.dim)),
            )
        vectors = graph.state.current[edge_index[1]] - graph.state.current[edge_index[0]]
        distances = vectors.norm(dim=-1)
        edge_sh = o3.spherical_harmonics(
            list(range(self.config.lmax + 1)), vectors,
            normalize=True, normalization="component",
        ).to(node.dtype)
        radial = self.radial(distances, graph.edge_kind).to(node.dtype)
        kind = self.edge_kind_embedding(graph.edge_kind)
        bond = self.bond_type_embedding(graph.bond_type.clamp(0, 31))
        invariant = equivariant_to_invariant(node, self.atom_irreps)
        endpoint = torch.cat(
            (
                invariant[edge_index[0], :self._endpoint_width],
                invariant[edge_index[1], :self._endpoint_width],
            ),
            dim=-1,
        )
        return torch.cat((radial, kind, bond, endpoint), dim=-1), edge_sh

    def _pair_latent(self, graph: HeavyFlowAtomGraph, node: Tensor) -> Tensor:
        edges = graph.edge_index
        if not self.config.use_pair_latent:
            return node.new_empty((edges.shape[1], 0))
        if edges.shape[1] == 0:
            return node.new_empty((0, self.config.pair_dim))
        vectors = graph.state.current[edges[1]] - graph.state.current[edges[0]]
        distances = vectors.norm(dim=-1, keepdim=True)
        radial = self.radial(distances[:, 0], graph.edge_kind).to(node.dtype)
        kind = self.edge_kind_embedding(graph.edge_kind)
        bond = self.bond_type_embedding(graph.bond_type.clamp(0, 31))
        invariant = equivariant_to_invariant(node, self.atom_irreps)
        pair_input = torch.cat(
            (invariant[edges[0]], invariant[edges[1]], radial, kind, bond, distances), dim=-1
        )
        # The pair channels are intentionally an interaction representation,
        # not a unique pairwise-force decomposition of the net atom force.
        return self.pair_projection(pair_input)

    @staticmethod
    def _context_parts(
        context: Union[HeavyFlowContextEncoding, Tensor],
        graph: Optional[HeavyFlowAtomGraph],
        condition: Optional[HeavyFlowCondition],
    ) -> tuple[Tensor, HeavyFlowAtomGraph, Tensor, Tensor, Tensor, int, int]:
        if isinstance(context, HeavyFlowContextEncoding):
            features = context.atom_context.atom_features
            graph = context.geometry.graph
            atom_batch = context.atom_context.atom_batch
            atom_local = context.atom_context.local_atom
            atom_mask = context.atom_context.atom_mask
            batch_size = int(atom_mask.shape[0])
            max_atoms = int(atom_mask.shape[1])
        else:
            features = context
            if graph is None or condition is None:
                raise ValueError("raw C_atom input requires graph and condition")
            atom_batch = graph.state.batch
            atom_local = graph.state.local_atom
            atom_mask = condition.atom_mask
            batch_size = condition.batch_size
            max_atoms = condition.max_atoms
        if condition is not None:
            condition.validate()
            if condition.atom_mask.shape != atom_mask.shape:
                raise ValueError("condition and context atom masks disagree")
        if graph is None:
            raise ValueError("a HeavyFlowAtomGraph is required")
        return features, graph, atom_batch, atom_local, atom_mask, batch_size, max_atoms

    def forward(
        self,
        context: Union[HeavyFlowContextEncoding, Tensor],
        graph: Optional[HeavyFlowAtomGraph] = None,
        condition: Optional[HeavyFlowCondition] = None,
    ) -> PhysicsState:
        """Predict atom/pair latent fields without access to GT force."""
        features, graph, atom_batch, atom_local, atom_mask, batch_size, max_atoms = self._context_parts(
            context, graph, condition
        )
        if features.ndim != 2 or features.shape[-1] != self.atom_irreps.dim:
            raise ValueError("C_atom must be packed [M, atom_irreps.dim]")
        if features.shape[0] != graph.num_nodes:
            raise ValueError("C_atom and graph node count disagree")
        node = features
        edge_features, edge_sh = self._edge_features(graph, node)
        pair = self._pair_latent(graph, node)
        edge_features = torch.cat((edge_features, pair), dim=-1)
        for block in self.blocks:
            node = block(node, graph, edge_sh, edge_features)

        atom_scalar = self.scalar_projection(node)
        vector = self.vector_projection(node).reshape(node.shape[0], self.config.vector_channels, 3)
        axial = self.axial_projection(node).reshape(node.shape[0], self.config.axial_channels, 3)
        axial_invariant = equivariant_to_invariant(axial.flatten(1), f"{self.config.axial_channels}x1e")
        atom_scalar = atom_scalar + self.axial_to_scalar(axial_invariant)
        state = PhysicsState(
            atom_scalar=atom_scalar,
            atom_vector=vector,
            atom_axial=axial,
            edge_scalar=pair,
            atom_batch=atom_batch,
            atom_local=atom_local,
            atom_mask=atom_mask,
            edge_index=graph.edge_index,
            edge_kind=graph.edge_kind,
            bond_type=graph.bond_type,
            edge_mask=torch.ones(graph.num_edges, dtype=torch.bool, device=features.device),
            batch_size=batch_size,
            max_atoms=max_atoms,
        )
        state.validate()
        return state


@dataclass(frozen=True)
class PhysicsModelOutput:
    """Output of the composed Stage 3 context/predictor/head path."""

    context: HeavyFlowContextEncoding
    physics_state: PhysicsState
    force_mean: Tensor
    force_logvar: Tensor

    @property
    def force_distribution(self):
        from .force_head import ForceDistribution

        return ForceDistribution(self.force_mean, self.force_logvar, self.physics_state)


class HeavyFlowPhysicsModel(nn.Module):
    """Compose Stage 2 context, Stage 3 predictor, and atom force head."""

    def __init__(self, context_encoder: nn.Module, predictor: PhysicsPredictor, force_head: nn.Module):
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.force_head = force_head

    def forward(self, condition: HeavyFlowCondition) -> PhysicsModelOutput:
        # Deliberately only condition is accepted: targets stay in the loss.
        context = self.context_encoder(condition)
        state = self.predictor(context)
        prediction = self.force_head(state)
        enriched = state.with_force(prediction.force_mean, prediction.force_logvar)
        return PhysicsModelOutput(context, enriched, prediction.force_mean, prediction.force_logvar)
