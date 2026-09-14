"""E(3)-equivariant conditional Cartesian velocity decoder.

The decoder is intentionally a new path.  It does not import H0, H1a, H1b,
H2, residue-frame transition heads, or backmapping code.  Atom sequence/
geometry context, residue joint context, physics latents, predicted force
moments, and invariant scalar conditions enter through separate adapters and
gates at multiple decoder blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from e3nn import o3
from torch import Tensor, nn

from ..data.residue_constants import NUM_ATOM_NAMES
from .atom_graph import HeavyFlowAtomGraph
from .geometry_encoder import _EquivariantNormActivation, _RadialBasis, _scatter_sum
from .physics_types import PhysicsState
from .physics_edges import attach_physics_edges
from .seq_geo_fusion import SeqGeoResidueContext, equivariant_to_invariant, invariant_width
from .types import HeavyFlowCondition
from .flow_types import FlowAtomTopology, FlowConditionBundle, coerce_flow_topology

__all__ = [
    "FlowDecoderConfig",
    "FlowDecoder",
    "HeavyAtomFlowDecoder",
]


@dataclass(frozen=True)
class FlowDecoderConfig:
    """Initial bounded Stage 4 decoder configuration."""

    flow_blocks: int = 6
    lmax: int = 2
    hidden_irreps: str = "64x0e + 16x1o + 4x1e + 4x2e"
    conditioning_layers: tuple[int, ...] = (0, 2, 4)
    radial_basis: int = 16
    edge_embedding_dim: int = 8
    spatial_cutoff_angstrom: float = 6.0
    max_spatial_neighbors: int = 32
    spatial_recompute_every: int = 1
    time_embedding_dim: int = 32
    atom_identity_dim: int = 8
    max_atomic_number: int = 118

    def __post_init__(self) -> None:
        if self.flow_blocks < 1 or self.lmax < 0:
            raise ValueError("flow_blocks must be positive and lmax non-negative")
        if self.radial_basis < 2 or self.edge_embedding_dim < 1:
            raise ValueError("radial_basis and edge_embedding_dim must be positive")
        if self.spatial_cutoff_angstrom <= 0 or self.max_spatial_neighbors < 1:
            raise ValueError("spatial graph settings must be positive")
        if self.spatial_recompute_every < 1 or self.time_embedding_dim < 1:
            raise ValueError("spatial_recompute_every and time_embedding_dim must be positive")
        hidden = o3.Irreps(self.hidden_irreps)
        if max((ir.l for _, ir in hidden), default=0) > self.lmax:
            raise ValueError("hidden_irreps contains l greater than configured lmax")
        layers = tuple(int(layer) for layer in self.conditioning_layers)
        if any(layer < 0 or layer >= self.flow_blocks for layer in layers):
            raise ValueError("conditioning_layers must refer to decoder blocks")
        object.__setattr__(self, "conditioning_layers", layers)


def _packed_from_dense_or_packed(value: Tensor, topology: FlowAtomTopology, width: Optional[int] = None) -> Tensor:
    if value.ndim == 2:
        if value.shape[0] != topology.atom_batch.shape[0]:
            raise ValueError("packed atom feature count disagrees with topology")
        result = value
    elif value.ndim == 3 and value.shape[:2] == (topology.batch_size, topology.max_atoms):
        result = value[topology.atom_batch, topology.atom_local]
    else:
        raise ValueError("atom feature must be packed [M,D] or dense [B,N,D]")
    if width is not None and result.shape[-1] != width:
        raise ValueError(f"atom feature width {result.shape[-1]} disagrees with expected {width}")
    return result


def _packed_vector(
    value: Tensor,
    topology: FlowAtomTopology,
    channels: Optional[int] = None,
    *,
    dense: Optional[bool] = None,
) -> Tensor:
    """Normalize a polar/axial vector field to packed ``[M,V,3]``.

    A one-protein dense tensor has ``B == M`` in common tiny fixtures.  The
    explicit ``dense`` hint therefore avoids mistaking dense ``[B,N,3]`` for
    packed ``[M,V,3]`` when ``N`` happens to equal a channel count.
    """
    if dense is True:
        if value.ndim == 3 and value.shape[:2] == (topology.batch_size, topology.max_atoms) and value.shape[-1] == 3:
            result = value[topology.atom_batch, topology.atom_local][:, None, :]
        elif value.ndim == 4 and value.shape[:2] == (topology.batch_size, topology.max_atoms) and value.shape[-1] == 3:
            result = value[topology.atom_batch, topology.atom_local]
        else:
            raise ValueError("dense vector field must have shape [B,N,3] or [B,N,V,3]")
    else:
        if value.ndim == 2 and value.shape[-1] == 3:
            result = value[:, None, :]
        elif value.ndim == 3 and value.shape[0] == topology.atom_batch.shape[0] and value.shape[-1] == 3:
            result = value
        else:
            raise ValueError("packed vector field must have shape [M,3] or [M,V,3]")
    if channels is not None and result.shape[1] != channels:
        raise ValueError("vector channel count disagrees with physics configuration")
    return result


def _packed_force(value: Optional[Tensor], topology: FlowAtomTopology, *, axial: bool = False) -> Tensor:
    if value is None:
        return topology.atom_mask.new_zeros((topology.atom_batch.shape[0], 3), dtype=torch.float32)
    if value.ndim == 2 and value.shape[-1] == 3:
        if value.shape[0] == topology.atom_batch.shape[0]:
            return value
    if value.ndim == 3 and value.shape[:2] == (topology.batch_size, topology.max_atoms):
        return value[topology.atom_batch, topology.atom_local]
    raise ValueError("force mean must be packed [M,3] or dense [B,N,3]")


def _packed_logvar(value: Optional[Tensor], topology: FlowAtomTopology) -> Tensor:
    if value is None:
        return topology.atom_mask.new_zeros((topology.atom_batch.shape[0], 1), dtype=torch.float32)
    if value.ndim == 2 and value.shape[0] == topology.atom_batch.shape[0]:
        return value if value.shape[-1] == 1 else value.mean(dim=-1, keepdim=True)
    if value.ndim == 3 and value.shape[:2] == (topology.batch_size, topology.max_atoms):
        value = value[topology.atom_batch, topology.atom_local]
        return value if value.shape[-1] == 1 else value.mean(dim=-1, keepdim=True)
    raise ValueError("force logvar must be packed/dense with a scalar or 3-vector channel")


def _centred_positions(x_s: Tensor, topology: FlowAtomTopology) -> Tensor:
    dense = x_s
    weights = topology.atom_mask.to(dense.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (dense * weights[..., None]).sum(dim=1, keepdim=True) / count[..., None]
    return (dense - center) * weights[..., None]


def _update_graph_positions(graph: HeavyFlowAtomGraph, positions: Tensor) -> HeavyFlowAtomGraph:
    state = graph.state
    return HeavyFlowAtomGraph(
        graph.edge_index,
        graph.edge_kind,
        graph.bond_type,
        graph.edge_batch,
        type(state)(
            current=positions,
            history=state.history,
            batch=state.batch,
            local_atom=state.local_atom,
            packed_index=state.packed_index,
        ),
        graph.stats,
    )


class _FlowMessageBlock(nn.Module):
    def __init__(self, irreps: o3.Irreps, sh_irreps: o3.Irreps, edge_width: int, hidden_edge: int):
        super().__init__()
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

    def forward(self, node: Tensor, graph: HeavyFlowAtomGraph, edge_sh: Tensor, edge_features: Tensor) -> Tensor:
        if graph.edge_index.shape[1] == 0:
            aggregate = node.new_zeros(node.shape)
        else:
            weights = self.edge_mlp(edge_features)
            messages = self.tp(node[graph.edge_index[0]], edge_sh, weights)
            aggregate = _scatter_sum(messages, graph.edge_index[1], node.shape[0])
            degree = torch.bincount(graph.edge_index[1], minlength=node.shape[0]).to(node.dtype).clamp_min(1.0)
            aggregate = aggregate / degree.sqrt()[:, None]
        return self.norm(node + self.node_linear(node) + self.message_linear(aggregate))


class _IrrepGate:
    def __init__(self, irreps: o3.Irreps):
        self.irreps = irreps

    @property
    def width(self) -> int:
        return len(self.irreps)

    def __call__(self, values: Tensor, logits: Tensor) -> Tensor:
        if logits.shape[-1] != self.width:
            raise ValueError("gate width disagrees with hidden irreps")
        parts = []
        offset = 0
        gate = torch.sigmoid(logits)
        for index, (multiplicity, irrep) in enumerate(self.irreps):
            width = multiplicity * irrep.dim
            chunk = values[:, offset:offset + width].reshape(values.shape[0], multiplicity, irrep.dim)
            parts.append((chunk * gate[:, index, None, None]).reshape(values.shape[0], width))
            offset += width
        return torch.cat(parts, dim=-1)


class FlowDecoder(nn.Module):
    """Predict dense ``[B,N,3]`` polar velocity from a flow state.

    The public argument order mirrors the Stage 4 contract exactly.  ``x_s``
    is the only coordinate seen by this module; ``x_future`` is not an
    argument, so target construction cannot leak into conditioning.
    """

    def __init__(
        self,
        atom_context_irreps: o3.Irreps | str,
        residue_context_dim: int,
        physics_scalar_dim: int,
        physics_vector_channels: int = 1,
        physics_axial_channels: int = 1,
        *,
        physics_pair_dim: int = 32,
        config: Optional[FlowDecoderConfig] = None,
    ):
        super().__init__()
        self.config = config or FlowDecoderConfig()
        self.atom_context_irreps = o3.Irreps(atom_context_irreps)
        self.residue_context_dim = int(residue_context_dim)
        self.physics_scalar_dim = int(physics_scalar_dim)
        self.physics_vector_channels = int(physics_vector_channels)
        self.physics_axial_channels = int(physics_axial_channels)
        self.physics_pair_dim = int(physics_pair_dim)
        if min(self.residue_context_dim, self.physics_scalar_dim, self.physics_vector_channels, self.physics_axial_channels) < 1:
            raise ValueError("context and physics widths must be positive")
        self.hidden_irreps = o3.Irreps(self.config.hidden_irreps)
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.config.lmax)
        self.gates = _IrrepGate(self.hidden_irreps)
        self.context_invariant_dim = invariant_width(self.atom_context_irreps)
        self.physics_irreps = o3.Irreps(
            f"{self.physics_vector_channels}x1o + {self.physics_axial_channels}x1e"
        )
        edge_width = self.config.radial_basis + 2 * self.config.edge_embedding_dim + self.physics_pair_dim + 1
        self.radial = _RadialBasis(self.config.radial_basis, self.config.spatial_cutoff_angstrom)
        self.edge_kind_embedding = nn.Embedding(3, self.config.edge_embedding_dim)
        self.bond_type_embedding = nn.Embedding(32, self.config.edge_embedding_dim)
        self.blocks = nn.ModuleList(
            _FlowMessageBlock(self.hidden_irreps, self.sh_irreps, edge_width, max(64, self.hidden_irreps.dim // 2))
            for _ in range(self.config.flow_blocks)
        )

        # Independent modality adapters.  The physics tensor product is an
        # explicit e3nn product with a scalar one, preserving 1o/1e parity.
        self.position_adapter = o3.Linear(o3.Irreps("1x1o"), self.hidden_irreps)
        self.atom_context_adapter = o3.Linear(self.atom_context_irreps, self.hidden_irreps)
        self.residue_context_adapter = o3.Linear(o3.Irreps(f"{self.residue_context_dim}x0e"), self.hidden_irreps)
        self.physics_tensor_product = o3.FullyConnectedTensorProduct(
            self.physics_irreps, o3.Irreps("1x0e"), self.hidden_irreps,
            shared_weights=True, internal_weights=True,
        )
        self.force_adapter = o3.Linear(o3.Irreps("1x1o"), self.hidden_irreps)
        self.time_embedding = nn.Sequential(
            nn.Linear(6, self.config.time_embedding_dim), nn.SiLU(),
            nn.Linear(self.config.time_embedding_dim, self.config.time_embedding_dim), nn.SiLU(),
        )
        self.time_adapter = o3.Linear(o3.Irreps(f"{self.config.time_embedding_dim}x0e"), self.hidden_irreps)
        self.atom_type_embedding = nn.Embedding(self.config.max_atomic_number + 1, self.config.atom_identity_dim)
        self.atom_name_embedding = nn.Embedding(NUM_ATOM_NAMES, self.config.atom_identity_dim)
        self.identity_adapter = o3.Linear(
            o3.Irreps(f"{2 * self.config.atom_identity_dim}x0e"), self.hidden_irreps
        )
        self.output = o3.Linear(self.hidden_irreps, o3.Irreps("1x1o"))

        gate_width = self.gates.width
        self.context_gate = nn.Sequential(
            nn.Linear(self.context_invariant_dim + self.residue_context_dim + self.config.time_embedding_dim, max(64, gate_width * 4)),
            nn.SiLU(), nn.Linear(max(64, gate_width * 4), gate_width),
        )
        self.residue_gate = nn.Sequential(
            nn.Linear(self.residue_context_dim + self.config.time_embedding_dim, max(64, gate_width * 2)),
            nn.SiLU(), nn.Linear(max(64, gate_width * 2), gate_width),
        )
        self.physics_gate = nn.Sequential(
            nn.Linear(self.physics_scalar_dim + 2 + self.config.time_embedding_dim, max(64, gate_width * 4)),
            nn.SiLU(), nn.Linear(max(64, gate_width * 4), gate_width),
        )
        self.force_gate = nn.Sequential(
            nn.Linear(2 + self.config.time_embedding_dim, max(64, gate_width * 2)),
            nn.SiLU(), nn.Linear(max(64, gate_width * 2), gate_width),
        )
        self.time_gate = nn.Sequential(
            nn.Linear(self.config.time_embedding_dim, max(64, gate_width * 2)),
            nn.SiLU(), nn.Linear(max(64, gate_width * 2), gate_width),
        )
        self._cached_graph: Optional[HeavyFlowAtomGraph] = None
        self.spatial_graph_recomputations = 0

    def reset_graph_cache(self) -> None:
        self._cached_graph = None
        self.spatial_graph_recomputations = 0

    def _dynamic_graph(self, topology: FlowAtomTopology, x_s: Tensor, recompute_spatial: bool) -> HeavyFlowAtomGraph:
        same_shape = self._cached_graph is not None and self._cached_graph.state.current.shape[0] == topology.atom_batch.shape[0]
        if recompute_spatial or not same_shape:
            graph = topology.graph_for(x_s, rebuild_spatial=True)
            self._cached_graph = graph
            self.spatial_graph_recomputations += 1
            return graph
        # Hold the neighbour list for a bounded interval but always update the
        # geometry used by radial features and spherical harmonics.
        graph = _update_graph_positions(self._cached_graph, topology.packed_positions(x_s))
        self._cached_graph = graph
        return graph

    def _edge_features(self, graph: HeavyFlowAtomGraph, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if graph.edge_index.shape[1] == 0:
            width = self.config.radial_basis + 2 * self.config.edge_embedding_dim
            return (
                graph.state.current.new_empty((0, width), dtype=dtype),
                graph.state.current.new_empty((0, self.sh_irreps.dim), dtype=dtype),
            )
        vectors = graph.state.current[graph.edge_index[1]] - graph.state.current[graph.edge_index[0]]
        distance = vectors.norm(dim=-1)
        edge_sh = o3.spherical_harmonics(
            list(range(self.config.lmax + 1)), vectors,
            normalize=True, normalization="component",
        ).to(dtype)
        radial = self.radial(distance, graph.edge_kind).to(dtype)
        kind = self.edge_kind_embedding(graph.edge_kind).to(dtype)
        bond = self.bond_type_embedding(graph.bond_type.clamp(0, 31)).to(dtype)
        return torch.cat((radial, kind, bond), dim=-1), edge_sh

    def _atom_context(self, atom_context: Union[Tensor, object], topology: FlowAtomTopology) -> Tensor:
        if hasattr(atom_context, "atom_features"):
            atom_context = atom_context.atom_features
        elif hasattr(atom_context, "atom_context") and hasattr(atom_context.atom_context, "atom_features"):
            atom_context = atom_context.atom_context.atom_features
        if not isinstance(atom_context, Tensor):
            raise TypeError("atom_context must be a tensor or Stage 2 atom context object")
        return _packed_from_dense_or_packed(atom_context, topology, self.atom_context_irreps.dim)

    def _residue_context(self, residue_context: Union[Tensor, SeqGeoResidueContext], topology: FlowAtomTopology) -> Tensor:
        if isinstance(residue_context, SeqGeoResidueContext):
            residue_context = residue_context.joint_scalar
        if not isinstance(residue_context, Tensor) or residue_context.ndim != 3:
            raise TypeError("residue_context must be [B,L,D] or SeqGeoResidueContext")
        if residue_context.shape[0] != topology.batch_size or residue_context.shape[-1] != self.residue_context_dim:
            raise ValueError("residue_context shape disagrees with decoder")
        if topology.atom_to_residue is None:
            return residue_context.new_zeros((topology.atom_batch.shape[0], self.residue_context_dim))
        residue = topology.atom_to_residue[topology.atom_batch, topology.atom_local]
        return residue_context[topology.atom_batch, residue]

    def _physics_fields(self, state: PhysicsState, topology: FlowAtomTopology) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state.validate()
        scalar = _packed_from_dense_or_packed(state.atom_scalar, topology, self.physics_scalar_dim)
        dense_state = state.atom_scalar.ndim == 3
        vector = _packed_vector(state.atom_vector, topology, self.physics_vector_channels, dense=dense_state)
        axial = _packed_vector(state.atom_axial, topology, self.physics_axial_channels, dense=dense_state)
        mean = _packed_force(state.force_mean, topology).to(scalar.dtype)
        logvar = _packed_logvar(state.force_logvar, topology).to(scalar.dtype)
        physics = torch.cat((vector.reshape(vector.shape[0], -1), axial.reshape(axial.shape[0], -1)), dim=-1)
        return scalar, physics, mean, logvar, torch.cat((mean.norm(dim=-1, keepdim=True), logvar), dim=-1)

    def _time_features(self, flow_time: Tensor, temperature: Tensor, lag: Tensor, atom_batch: Tensor, dtype: torch.dtype) -> Tensor:
        if flow_time.ndim == 0:
            flow_time = flow_time.expand(int(atom_batch.max().item()) + 1 if atom_batch.numel() else temperature.shape[0])
        if flow_time.shape != temperature.shape or lag.shape != temperature.shape:
            raise ValueError("flow_time, temperature, and lag must all be [B]")
        s = flow_time.to(dtype)[atom_batch]
        temp = temperature.to(dtype)[atom_batch] / 300.0
        log_lag = lag.to(dtype).clamp_min(1e-6).log1p()[atom_batch]
        raw = torch.stack((s, s.square(), torch.sin(torch.pi * s), torch.cos(torch.pi * s), temp, log_lag), dim=-1)
        return self.time_embedding(raw)

    def forward(
        self,
        x_s: Tensor,
        flow_time: Tensor,
        atom_context: Union[Tensor, object],
        residue_context: Union[Tensor, SeqGeoResidueContext],
        physics_state: PhysicsState,
        atom_topology: FlowAtomTopology | HeavyFlowAtomGraph,
        temperature: Tensor,
        lag: Tensor,
        *,
        recompute_spatial: bool = True,
    ) -> Tensor:
        """Return dense polar velocity ``[B,N,3]`` without future/GT inputs."""
        if x_s.ndim != 3 or x_s.shape[-1] != 3:
            raise ValueError("x_s must have shape [B,N,3]")
        if temperature.ndim != 1 or lag.ndim != 1 or temperature.shape != lag.shape:
            raise ValueError("temperature and lag must be [B]")
        topology = coerce_flow_topology(
            atom_topology,
            atom_mask=torch.ones(x_s.shape[:2], dtype=torch.bool, device=x_s.device)
            if isinstance(atom_topology, HeavyFlowAtomGraph) and atom_topology.state.local_atom.numel() == x_s.shape[1]
            else None,
            spatial_cutoff_angstrom=self.config.spatial_cutoff_angstrom,
            max_spatial_neighbors=self.config.max_spatial_neighbors,
        )
        if topology.batch_size != x_s.shape[0] or topology.max_atoms != x_s.shape[1]:
            raise ValueError("x_s dimensions disagree with atom topology")
        topology.validate()
        graph = self._dynamic_graph(topology, x_s, recompute_spatial)
        if physics_state.atom_mask is not None and not torch.equal(physics_state.atom_mask, topology.atom_mask):
            raise ValueError("physics atom mask differs from decoder topology")
        graph, pair_features = attach_physics_edges(graph, physics_state, self.physics_pair_dim)
        atom = self._atom_context(atom_context, topology)
        residue = self._residue_context(residue_context, topology)
        physics_scalar, physics, force_mean, force_logvar, force_summary = self._physics_fields(physics_state, topology)
        dtype = atom.dtype
        time = self._time_features(flow_time, temperature, lag, topology.atom_batch, dtype)
        atom_invariant = equivariant_to_invariant(atom, self.atom_context_irreps)

        if topology.atom_type is None:
            atom_type = torch.zeros((atom.shape[0],), dtype=torch.int64, device=atom.device)
        else:
            atom_type = topology.atom_type[topology.atom_batch, topology.atom_local].clamp(0, self.config.max_atomic_number)
        if topology.atom_name is None:
            atom_name = torch.zeros((atom.shape[0],), dtype=torch.int64, device=atom.device)
        else:
            atom_name = topology.atom_name[topology.atom_batch, topology.atom_local].clamp(0, NUM_ATOM_NAMES - 1)
        identity = torch.cat((self.atom_type_embedding(atom_type), self.atom_name_embedding(atom_name)), dim=-1)
        centered = _centred_positions(x_s, topology)[topology.atom_batch, topology.atom_local]
        node = self.position_adapter(centered.to(dtype)) + self.identity_adapter(identity.to(dtype))

        context_value = self.atom_context_adapter(atom)
        residue_value = self.residue_context_adapter(residue.to(dtype))
        physics_value = self.physics_tensor_product(physics, torch.ones((physics.shape[0], 1), dtype=dtype, device=physics.device))
        force_value = self.force_adapter(force_mean.to(dtype))
        time_value = self.time_adapter(time)
        context_gate = self.context_gate(torch.cat((atom_invariant.to(dtype), residue.to(dtype), time), dim=-1))
        residue_gate = self.residue_gate(torch.cat((residue.to(dtype), time), dim=-1))
        physics_gate = self.physics_gate(torch.cat((physics_scalar.to(dtype), force_summary.to(dtype), time), dim=-1))
        force_gate = self.force_gate(torch.cat((force_summary.to(dtype), time), dim=-1))
        time_gate = self.time_gate(time)
        edge_features, edge_sh = self._edge_features(graph, dtype)
        edge_features = torch.cat((edge_features, pair_features.to(dtype)), dim=-1)
        for index, block in enumerate(self.blocks):
            node = block(node, graph, edge_sh, edge_features)
            if index in self.config.conditioning_layers:
                # These are separate additions and separate gates; no early
                # concatenation of sequence-geometry and physics is performed.
                node = node + self.gates(context_value, context_gate)
                node = node + self.gates(residue_value, residue_gate)
                node = node + self.gates(physics_value, physics_gate)
                node = node + self.gates(force_value, force_gate)
                node = node + self.gates(time_value, time_gate)
        velocity = self.output(node)
        dense = velocity.new_zeros((topology.batch_size, topology.max_atoms, 3))
        dense[topology.atom_batch, topology.atom_local] = velocity
        return dense * topology.atom_mask[..., None].to(dense.dtype)


HeavyAtomFlowDecoder = FlowDecoder
