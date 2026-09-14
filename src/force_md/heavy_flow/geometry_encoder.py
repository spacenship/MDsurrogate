"""Generic e3nn atom-level message passing for heavy-flow Stage 1.

This is a small, explicit tensor-product network. It is not MACE, NequIP,
Allegro, IPA, or a force head. It consumes only ``HeavyFlowCondition`` and
returns an atom-level irreps feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..data.residue_constants import NUM_ATOM_NAMES
from .atom_graph import HeavyFlowAtomGraph, HeavyFlowGraphConfig, build_atom_graph
from .esmc_encoder import ESMCConfig, ESMCEncoder, ESMCEncoding
from .types import HeavyFlowCondition

__all__ = [
    "HeavyFlowGeometryConfig",
    "HeavyFlowGeometryEncoding",
    "HeavyFlowAtomEncoder",
    "HeavyFlowConditionEncoder",
]


@dataclass(frozen=True)
class HeavyFlowGeometryConfig:
    num_blocks: int = 4
    lmax: int = 2
    hidden_irreps: str = "96x0e + 24x1o + 8x1e + 8x2e"
    radial_basis: int = 16
    radial_width: float = 1.0
    history_scale_angstrom: float = 6.0
    max_atomic_number: int = 118
    scalar_embedding_dim: int = 16
    edge_embedding_dim: int = 8
    edge_chunk_size: int = 2048
    activation_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.num_blocks < 1 or self.lmax < 0:
            raise ValueError("num_blocks must be positive and lmax non-negative")
        if self.radial_basis < 2 or self.history_scale_angstrom <= 0:
            raise ValueError("radial_basis must be >=2 and history scale positive")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        irreps = o3.Irreps(self.hidden_irreps)
        if max((ir.l for _, ir in irreps), default=0) > self.lmax:
            raise ValueError("hidden_irreps contains l greater than configured lmax")


@dataclass(frozen=True)
class HeavyFlowGeometryEncoding:
    atom_features: Tensor  # [M, irreps.dim], valid atoms in batch-local order
    padded_features: Tensor  # [B,N,irreps.dim], zero on padding
    irreps: o3.Irreps
    graph: HeavyFlowAtomGraph
    sequence: ESMCEncoding

    @property
    def parameter_width(self) -> int:
        return self.irreps.dim


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    result = values.new_zeros((size, values.shape[-1]))
    if values.numel():
        result.index_add_(0, index, values)
    return result


def _tensor_product_aggregate(
    node: Tensor,
    edge_index: Tensor,
    edge_sh: Tensor,
    edge_features: Tensor,
    *,
    tensor_product: nn.Module,
    edge_mlp: nn.Module,
    edge_chunk_size: int,
) -> Tensor:
    """Accumulate tensor-product messages in bounded edge batches.

    The full graph edge list is retained for the model, but the expensive
    per-edge MLP output and tensor-product message are materialized only for
    one chunk at a time.  The caller may wrap the enclosing block in an
    activation checkpoint so autograd does not retain every chunk's internal
    activations until backward.
    """

    edge_count = int(edge_index.shape[1])
    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    if edge_count == 0:
        return node.new_zeros(node.shape)

    sources = edge_index[0]
    targets = edge_index[1]
    aggregate = node.new_zeros(node.shape)
    for start in range(0, edge_count, edge_chunk_size):
        stop = min(start + edge_chunk_size, edge_count)
        weights = edge_mlp(edge_features[start:stop])
        messages = tensor_product(
            node[sources[start:stop]],
            edge_sh[start:stop],
            weights,
        )
        # The accumulator is fresh and index_add's backward does not need its
        # previous contents. Avoid a full [nodes, channels] zero + add per tile.
        aggregate.index_add_(0, targets[start:stop], messages)
    return aggregate


def _bounded_vector(values: Tensor, scale: float) -> Tensor:
    norm = values.norm(dim=-1, keepdim=True)
    return values / norm.clamp_min(1e-8) * torch.tanh(norm / scale)


class _EquivariantNormActivation(nn.Module):
    """Parameter-free norm nonlinearity for arbitrary e3nn irreps.

    e3nn 0.6 does not expose ``NormActivation`` under ``o3``. Applying a
    scalar function to each irrep multiplicity's norm commutes with every
    orthogonal representation matrix, including odd/even vector and rank-two
    channels.
    """

    def __init__(self, irreps: o3.Irreps, *, norm_epsilon: float = 1e-4):
        super().__init__()
        self.irreps = irreps
        if norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        self.norm_epsilon = float(norm_epsilon)

    def forward(self, values: Tensor) -> Tensor:
        parts: list[Tensor] = []
        offset = 0
        for multiplicity, irrep in self.irreps:
            width = multiplicity * irrep.dim
            chunk = values[:, offset:offset + width]
            offset += width
            shaped = chunk.reshape(chunk.shape[0], multiplicity, irrep.dim)
            if irrep.l == 0:
                parts.append(torch.nn.functional.silu(shaped).reshape(chunk.shape[0], width))
                continue
            # Norm activations can receive exactly zero disallowed/parity
            # channels (for example 1e channels projected from scalar/1o
            # inputs).  ``sqrt(x).backward()`` at x == 0 produces NaN even
            # though the forward value is finite.  Keep the e3nn convention
            # used here (mean over the irrep components), but do the norm
            # arithmetic in fp32 and regularize the square root itself.
            h32 = shaped.float()
            sq_norm = h32.square().mean(dim=-1, keepdim=True)
            safe_norm = torch.sqrt(sq_norm + self.norm_epsilon ** 2)
            normalized = h32 / safe_norm
            activated = normalized * torch.tanh(safe_norm)
            parts.append(activated.to(dtype=chunk.dtype).reshape(chunk.shape[0], width))
        return torch.cat(parts, dim=-1) if parts else values


class _RadialBasis(nn.Module):
    def __init__(self, count: int, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("centers", torch.linspace(0.0, self.cutoff, count), persistent=False)
        self.width = self.cutoff / max(count - 1, 1)

    def forward(self, distance: Tensor, edge_kind: Tensor) -> Tensor:
        # Covalent edges are retained outside the spatial cutoff. Their radial
        # basis is therefore not hard-zeroed; only radius-selected edges get
        # the smooth cutoff envelope.
        centers = self.centers.to(device=distance.device, dtype=distance.dtype)
        value = torch.exp(-((distance[:, None] - centers[None, :]) / self.width) ** 2)
        spatial = edge_kind == 1
        envelope = torch.ones_like(distance)
        normalized = (distance / self.cutoff).clamp_min(0.0)
        envelope[spatial] = 0.5 * (1.0 + torch.cos(torch.pi * normalized[spatial]))
        envelope[spatial & (distance >= self.cutoff)] = 0.0
        return value * envelope[:, None]


class _TensorProductBlock(nn.Module):
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

    def forward(
        self,
        node: Tensor,
        edge_index: Tensor,
        edge_sh: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                node,
                edge_index,
                edge_sh,
                edge_features,
                use_reentrant=False,
            )
        return self._forward_impl(node, edge_index, edge_sh, edge_features)


class HeavyFlowAtomEncoder(nn.Module):
    """Sequence-conditioned, history-aware atom feature encoder."""

    def __init__(
        self,
        *,
        geometry_config: Optional[HeavyFlowGeometryConfig] = None,
        graph_config: Optional[HeavyFlowGraphConfig] = None,
        sequence_encoder: Optional[ESMCEncoder] = None,
        esmc_config: Optional[ESMCConfig] = None,
    ):
        super().__init__()
        self.geometry_config = geometry_config or HeavyFlowGeometryConfig()
        self.graph_config = graph_config or HeavyFlowGraphConfig()
        self.sequence_encoder = sequence_encoder or ESMCEncoder(esmc_config or ESMCConfig())
        p = self.sequence_encoder.config.projected_dim
        d = self.geometry_config.scalar_embedding_dim
        self.atom_type_embedding = nn.Embedding(self.geometry_config.max_atomic_number + 1, d)
        self.atom_name_embedding = nn.Embedding(NUM_ATOM_NAMES, d)
        self.residue_embedding = nn.Embedding(32, d)
        self.bond_type_embedding = nn.Embedding(32, self.geometry_config.edge_embedding_dim)
        # Atom/name/residue embeddings, ESM-C projection, six binary flags,
        # four history scalars, three temperature scalars, and one node bond
        # summary. All are invariant scalar channels.
        self.scalar_width = 3 * d + p + 6 + 4 + 3 + self.geometry_config.edge_embedding_dim
        input_irreps = o3.Irreps(f"{self.scalar_width}x0e + 2x1o")
        self.hidden_irreps = o3.Irreps(self.geometry_config.hidden_irreps)
        self.node_init = o3.Linear(input_irreps, self.hidden_irreps)
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.geometry_config.lmax)
        endpoint_width = min(16, self.scalar_width)
        edge_width = self.geometry_config.radial_basis + 2 * self.geometry_config.edge_embedding_dim + 2 * endpoint_width
        self.radial = _RadialBasis(self.geometry_config.radial_basis, self.graph_config.spatial_cutoff_angstrom)
        self.edge_kind_embedding = nn.Embedding(3, self.geometry_config.edge_embedding_dim)
        self.blocks = nn.ModuleList(
            _TensorProductBlock(
                self.hidden_irreps,
                self.sh_irreps,
                edge_width,
                max(64, self.hidden_irreps.dim // 2),
                edge_chunk_size=self.geometry_config.edge_chunk_size,
                activation_checkpoint=self.geometry_config.activation_checkpoint,
            )
            for _ in range(self.geometry_config.num_blocks)
        )
        self._endpoint_width = endpoint_width

    @property
    def output_irreps(self) -> o3.Irreps:
        return self.hidden_irreps

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

    def _node_scalars(self, condition: HeavyFlowCondition, graph: HeavyFlowAtomGraph, sequence: ESMCEncoding) -> tuple[Tensor, Tensor]:
        state = graph.state
        batch, local = state.batch, state.local_atom
        residue = condition.atom_to_residue[batch, local]
        atom_type = condition.atom_type[batch, local].clamp(0, self.geometry_config.max_atomic_number)
        atom_name = condition.atom_name[batch, local].clamp(0, NUM_ATOM_NAMES - 1)
        residue_type = condition.sequence_tokens[batch, residue].clamp(0, 31)
        esm = sequence.projected_embedding[batch, residue]
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
        chain_flags = self._chain_flags(condition, batch, residue)
        flags = torch.stack((backbone, sidechain, terminal, chain_flags[:, 0], chain_flags[:, 1], condition.atom_mask[batch, local]), dim=-1).to(esm.dtype)

        history = state.history
        relative = history - state.current[:, None, :]
        relative_norm = relative.norm(dim=-1)
        if history.shape[1] > 1:
            frame_delta = history[:, 1:] - history[:, :-1]
            delta_norm = frame_delta.norm(dim=-1)
            mean_delta = frame_delta.mean(dim=1)
            delta_mean = delta_norm.mean(dim=1)
        else:
            mean_delta = torch.zeros_like(state.current)
            delta_mean = torch.zeros_like(relative_norm[:, 0])
        mean_relative = relative.mean(dim=1)
        history_scalar = torch.stack((
            relative_norm.mean(dim=1), relative_norm.max(dim=1).values,
            relative_norm[:, -1], delta_mean,
        ), dim=-1)
        temp = condition.temperature[batch].to(esm.dtype)
        time_scalar = torch.stack((
            temp / 400.0, torch.sin(temp * 0.01), torch.cos(temp * 0.01),
        ), dim=-1)
        if graph.edge_index.shape[1]:
            incoming = graph.edge_index[1]
            edge_bond = graph.bond_type
            node_bond = _scatter_sum(
                self.bond_type_embedding(edge_bond.clamp(0, 31)), incoming, state.current.shape[0]
            )
            degree = torch.bincount(incoming, minlength=state.current.shape[0]).to(esm.dtype).clamp_min(1.0)
            node_bond = node_bond / degree[:, None]
        else:
            node_bond = esm.new_zeros((state.current.shape[0], self.geometry_config.edge_embedding_dim))
        scalar = torch.cat((
            self.atom_type_embedding(atom_type), self.atom_name_embedding(atom_name),
            self.residue_embedding(residue_type), esm, flags, history_scalar, time_scalar, node_bond,
        ), dim=-1)
        vectors = torch.cat((
            _bounded_vector(mean_relative, self.geometry_config.history_scale_angstrom),
            _bounded_vector(mean_delta, self.geometry_config.history_scale_angstrom),
        ), dim=-1)
        return scalar, vectors

    def _edge_features(self, condition: HeavyFlowCondition, graph: HeavyFlowAtomGraph, scalar: Tensor) -> tuple[Tensor, Tensor]:
        edge_index = graph.edge_index
        if edge_index.shape[1] == 0:
            return scalar.new_empty((0, self.radial.centers.numel() + 2 * self.geometry_config.edge_embedding_dim + 2 * self._endpoint_width)), scalar.new_empty((0, self.sh_irreps.dim))
        vectors = graph.state.current[edge_index[1]] - graph.state.current[edge_index[0]]
        distances = vectors.norm(dim=-1)
        sh = o3.spherical_harmonics(
            list(range(self.geometry_config.lmax + 1)), vectors,
            normalize=True, normalization="component",
        ).to(scalar.dtype)
        radial = self.radial(distances, graph.edge_kind).to(scalar.dtype)
        kind = self.edge_kind_embedding(graph.edge_kind)
        bond = self.bond_type_embedding(graph.bond_type.clamp(0, 31))
        endpoint = torch.cat((scalar[edge_index[0], :self._endpoint_width], scalar[edge_index[1], :self._endpoint_width]), dim=-1)
        return torch.cat((radial, kind, bond, endpoint), dim=-1), sh

    def forward(self, condition: HeavyFlowCondition) -> HeavyFlowGeometryEncoding:
        condition.validate()
        graph = build_atom_graph(condition, self.graph_config)
        sequence = self.sequence_encoder(condition)
        scalar, vectors = self._node_scalars(condition, graph, sequence)
        node = self.node_init(torch.cat((scalar, vectors), dim=-1))
        edge_features, edge_sh = self._edge_features(condition, graph, scalar)
        for block in self.blocks:
            node = block(node, graph.edge_index, edge_sh, edge_features)
        padded = node.new_zeros((condition.batch_size, condition.max_atoms, node.shape[-1]))
        for batch in range(condition.batch_size):
            local = torch.nonzero(condition.atom_mask[batch], as_tuple=False).flatten()
            if local.numel():
                packed = graph.state.packed_index[batch, local]
                padded[batch, local] = node[packed]
        return HeavyFlowGeometryEncoding(node, padded, self.hidden_irreps, graph, sequence)


HeavyFlowConditionEncoder = HeavyFlowAtomEncoder
