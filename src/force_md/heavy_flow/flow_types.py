"""Typed contracts for the direct heavy-atom rectified-flow decoder.

Stage 4 keeps the upstream modalities separate until the decoder's adapters:

``C_atom`` + ``C_res`` + ``PhysicsState`` + predicted force distribution
    -> conditional Cartesian velocity field.

The topology object in this module stores fixed covalent/chain edges and
rebuilds only spatial edges from the current integration coordinates.  It is
deliberately independent of the legacy H0--H2 and residue-frame modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from .atom_graph import (
    COVALENT_EDGE,
    SPATIAL_EDGE,
    HeavyFlowAtomGraph,
    HeavyFlowGraphStats,
    PackedAtomState,
)
from .physics_types import PhysicsState

__all__ = [
    "FlowAtomTopology",
    "FlowConditionBundle",
    "FlowCondition",
    "HeavyAtomFlowCondition",
    "build_flow_graph",
    "coerce_flow_topology",
]


def _as_dense_positions(positions: Tensor, topology: "FlowAtomTopology") -> Tensor:
    if positions.ndim == 3:
        if positions.shape[:2] != (topology.batch_size, topology.max_atoms) or positions.shape[-1] != 3:
            raise ValueError("dense positions must have shape [B,N,3]")
        return positions
    if positions.ndim == 2 and positions.shape == (topology.atom_batch.shape[0], 3):
        dense = positions.new_zeros((topology.batch_size, topology.max_atoms, 3))
        dense[topology.atom_batch, topology.atom_local] = positions
        return dense
    raise ValueError("positions must be [B,N,3] or packed [M,3]")


def _pack_positions(positions: Tensor, topology: "FlowAtomTopology") -> Tensor:
    if positions.ndim == 2:
        if positions.shape != (topology.atom_batch.shape[0], 3):
            raise ValueError("packed positions do not match topology atom count")
        return positions
    dense = _as_dense_positions(positions, topology)
    return dense[topology.atom_batch, topology.atom_local]


@dataclass(frozen=True)
class FlowAtomTopology:
    """Fixed atom identity plus dynamic spatial-neighbour construction.

    ``fixed_edge_*`` contains the graph's non-spatial edges in packed atom
    order.  Spatial edges are never copied from the input graph: they are
    selected from the supplied ``x_s`` at each requested refresh.  This makes
    the bond constraint explicit and prevents the ODE from accidentally using
    a graph built from future coordinates.
    """

    fixed_edge_index: Tensor  # [2,E_fixed], packed atom indices
    fixed_edge_kind: Tensor  # [E_fixed], covalent or chain-adjacent
    fixed_bond_type: Tensor  # [E_fixed]
    fixed_edge_batch: Tensor  # [E_fixed]
    atom_batch: Tensor  # [M]
    atom_local: Tensor  # [M]
    atom_mask: Tensor  # [B,N]
    batch_size: int
    max_atoms: int
    atom_to_residue: Optional[Tensor] = None  # [B,N]
    atom_type: Optional[Tensor] = None  # [B,N]
    atom_name: Optional[Tensor] = None  # [B,N]
    residue_mask: Optional[Tensor] = None  # [B,L]
    chain_break: Optional[Tensor] = None  # [B,L-1]
    spatial_cutoff_angstrom: float = 6.0
    max_spatial_neighbors: int = 32

    @classmethod
    def from_graph(
        cls,
        graph: HeavyFlowAtomGraph,
        *,
        atom_mask: Optional[Tensor] = None,
        atom_to_residue: Optional[Tensor] = None,
        atom_type: Optional[Tensor] = None,
        atom_name: Optional[Tensor] = None,
        residue_mask: Optional[Tensor] = None,
        chain_break: Optional[Tensor] = None,
        spatial_cutoff_angstrom: float = 6.0,
        max_spatial_neighbors: int = 32,
    ) -> "FlowAtomTopology":
        """Create a flow topology from the Stage 1/2 graph output."""
        if not isinstance(graph, HeavyFlowAtomGraph):
            raise TypeError("graph must be a HeavyFlowAtomGraph")
        if atom_mask is not None:
            if atom_mask.ndim != 2 or atom_mask.dtype != torch.bool:
                raise ValueError("atom_mask must be a dense [B,N] bool tensor")
            batch_size, max_atoms = map(int, atom_mask.shape)
        else:
            batch_size = int(graph.state.batch.max().item()) + 1 if graph.state.batch.numel() else 0
            max_atoms = int(graph.state.local_atom.max().item()) + 1 if graph.state.local_atom.numel() else 0
            atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=graph.state.current.device)
            if graph.state.batch.numel():
                atom_mask[graph.state.batch, graph.state.local_atom] = True
        if graph.state.batch.numel() and (int(graph.state.batch.max()) >= batch_size or int(graph.state.local_atom.max()) >= max_atoms):
            raise ValueError("atom_mask is smaller than the graph's packed atom mapping")
        fixed = graph.edge_kind != SPATIAL_EDGE
        return cls(
            fixed_edge_index=graph.edge_index[:, fixed],
            fixed_edge_kind=graph.edge_kind[fixed],
            fixed_bond_type=graph.bond_type[fixed],
            fixed_edge_batch=graph.edge_batch[fixed],
            atom_batch=graph.state.batch,
            atom_local=graph.state.local_atom,
            atom_mask=atom_mask,
            batch_size=batch_size,
            max_atoms=max_atoms,
            atom_to_residue=atom_to_residue,
            atom_type=atom_type,
            atom_name=atom_name,
            residue_mask=residue_mask,
            chain_break=chain_break,
            spatial_cutoff_angstrom=spatial_cutoff_angstrom,
            max_spatial_neighbors=max_spatial_neighbors,
        )

    def validate(self) -> None:
        if self.fixed_edge_index.ndim != 2 or self.fixed_edge_index.shape[0] != 2:
            raise ValueError("fixed_edge_index must be [2,E]")
        edge_count = self.fixed_edge_index.shape[1]
        for name, value in (
            ("fixed_edge_kind", self.fixed_edge_kind),
            ("fixed_bond_type", self.fixed_bond_type),
            ("fixed_edge_batch", self.fixed_edge_batch),
        ):
            if value.ndim != 1 or value.shape[0] != edge_count:
                raise ValueError(f"{name} must have shape [E_fixed]")
        if self.atom_batch.ndim != 1 or self.atom_local.ndim != 1 or self.atom_batch.shape != self.atom_local.shape:
            raise ValueError("atom_batch and atom_local must be [M]")
        if self.atom_mask.shape != (self.batch_size, self.max_atoms) or self.atom_mask.dtype != torch.bool:
            raise ValueError("atom_mask disagrees with topology dimensions")
        if self.atom_to_residue is not None and self.atom_to_residue.shape != self.atom_mask.shape:
            raise ValueError("atom_to_residue must match atom_mask")
        if self.atom_type is not None and self.atom_type.shape != self.atom_mask.shape:
            raise ValueError("atom_type must match atom_mask")
        if self.atom_name is not None and self.atom_name.shape != self.atom_mask.shape:
            raise ValueError("atom_name must match atom_mask")
        if self.spatial_cutoff_angstrom <= 0 or self.max_spatial_neighbors < 1:
            raise ValueError("spatial cutoff and max neighbors must be positive")

    def to(self, device: torch.device | str) -> "FlowAtomTopology":
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        return FlowAtomTopology(**values)

    def packed_positions(self, positions: Tensor) -> Tensor:
        return _pack_positions(positions, self)

    def graph_for(self, positions: Tensor, *, rebuild_spatial: bool = True) -> HeavyFlowAtomGraph:
        """Build a graph whose fixed edges are retained and spatial edges use ``positions``."""
        return build_flow_graph(self, positions, rebuild_spatial=rebuild_spatial)


@dataclass(frozen=True)
class FlowConditionBundle:
    """All inference-time conditioning supplied to the Stage 4 decoder.

    The fields are intentionally not fused here.  ``FlowDecoder`` owns the
    separate context, residue, physics, force, and scalar adapters.
    """

    atom_context: Tensor  # packed [M,D_atom] or dense [B,N,D_atom]
    residue_context: Tensor  # [B,L,D_residue]
    physics_state: PhysicsState
    force_mean: Tensor  # [B,N,3], predicted Stage 3 mean
    force_logvar: Tensor  # [B,N,1], predicted Stage 3 uncertainty
    atom_topology: FlowAtomTopology
    temperature: Tensor  # [B]
    lag: Tensor  # [B]
    current_positions: Tensor  # [B,N,3]

    @property
    def atom_mask(self) -> Tensor:
        return self.atom_topology.atom_mask

    @property
    def batch_size(self) -> int:
        return self.atom_topology.batch_size

    @property
    def max_atoms(self) -> int:
        return self.atom_topology.max_atoms

    def validate(self) -> None:
        topology = self.atom_topology
        topology.validate()
        if not isinstance(self.atom_context, Tensor):
            raise ValueError("atom_context must be a tensor")
        packed_shape = self.atom_context.ndim == 2 and self.atom_context.shape[0] == topology.atom_batch.shape[0]
        dense_shape = self.atom_context.ndim == 3 and self.atom_context.shape[:2] == (self.batch_size, self.max_atoms)
        if not (packed_shape or dense_shape):
            raise ValueError("atom_context must be packed [M,D] or dense [B,N,D]")
        if self.residue_context.ndim != 3 or self.residue_context.shape[0] != self.batch_size:
            raise ValueError("residue_context must have shape [B,L,D]")
        if self.force_mean.shape != (self.batch_size, self.max_atoms, 3):
            raise ValueError("force_mean must be [B,N,3]")
        if self.force_logvar.ndim != 3 or self.force_logvar.shape[:2] != (self.batch_size, self.max_atoms):
            raise ValueError("force_logvar must be [B,N,1] or [B,N,3]")
        if self.current_positions.shape != (self.batch_size, self.max_atoms, 3):
            raise ValueError("current_positions must be [B,N,3]")
        if self.temperature.shape != (self.batch_size,) or self.lag.shape != (self.batch_size,):
            raise ValueError("temperature and lag must be [B]")
        if not (self.force_mean.isfinite().all() and self.force_logvar.isfinite().all() and self.current_positions.isfinite().all()):
            raise ValueError("flow conditioning contains non-finite values")
        self.physics_state.validate()

    def to(self, device: torch.device | str) -> "FlowConditionBundle":
        values = {
            name: (getattr(self, name).to(device) if hasattr(getattr(self, name), "to") else getattr(self, name))
            for name in self.__dataclass_fields__
        }
        result = FlowConditionBundle(**values)
        result.validate()
        return result


FlowCondition = FlowConditionBundle
HeavyAtomFlowCondition = FlowConditionBundle


def _spatial_edges(
    positions: Tensor,
    atom_batch: Tensor,
    *,
    cutoff: float,
    max_neighbors: int,
) -> tuple[list[tuple[int, int, int]], int, int, int]:
    rows: list[tuple[int, int, int]] = []
    candidates = 0
    truncated = 0
    maximum = 0
    for batch in range(int(atom_batch.max().item()) + 1 if atom_batch.numel() else 0):
        ids = torch.nonzero(atom_batch == batch, as_tuple=False).flatten()
        if ids.numel() <= 1:
            continue
        local = positions[ids]
        distance = torch.cdist(local, local)
        radius = (distance <= cutoff) & ~torch.eye(ids.numel(), dtype=torch.bool, device=ids.device)
        candidates += int(radius.sum())
        for destination in range(ids.numel()):
            sources = torch.nonzero(radius[:, destination], as_tuple=False).flatten().tolist()
            sources.sort(key=lambda source: (float(distance[source, destination]), source))
            if len(sources) > max_neighbors:
                truncated += 1
                sources = sources[:max_neighbors]
            maximum = max(maximum, len(sources))
            rows.extend((int(ids[source]), int(ids[destination]), batch) for source in sources)
    return rows, candidates, truncated, maximum


def build_flow_graph(
    topology: FlowAtomTopology,
    positions: Tensor,
    *,
    rebuild_spatial: bool = True,
) -> HeavyFlowAtomGraph:
    """Create a packed graph for ``positions`` while preserving fixed bonds."""
    topology.validate()
    packed = topology.packed_positions(positions)
    if topology.fixed_edge_index.shape[1]:
        rows = [
            (
                int(topology.fixed_edge_index[0, i]),
                int(topology.fixed_edge_index[1, i]),
                int(topology.fixed_edge_kind[i]),
                int(topology.fixed_bond_type[i]),
                int(topology.fixed_edge_batch[i]),
            )
            for i in range(topology.fixed_edge_index.shape[1])
        ]
    else:
        rows = []
    candidates = truncated = maximum = 0
    if rebuild_spatial:
        spatial, candidates, truncated, maximum = _spatial_edges(
            packed,
            topology.atom_batch,
            cutoff=topology.spatial_cutoff_angstrom,
            max_neighbors=topology.max_spatial_neighbors,
        )
        rows.extend((source, destination, SPATIAL_EDGE, 0, batch) for source, destination, batch in spatial)
    rows.sort(key=lambda row: (row[4], row[2], row[1], row[0], row[3]))
    device = packed.device
    if rows:
        edge_index = torch.tensor([(row[0], row[1]) for row in rows], dtype=torch.int64, device=device).t()
        edge_kind = torch.tensor([row[2] for row in rows], dtype=torch.int64, device=device)
        bond_type = torch.tensor([row[3] for row in rows], dtype=torch.int64, device=device)
        edge_batch = torch.tensor([row[4] for row in rows], dtype=torch.int64, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64, device=device)
        edge_kind = torch.empty((0,), dtype=torch.int64, device=device)
        bond_type = torch.empty((0,), dtype=torch.int64, device=device)
        edge_batch = torch.empty((0,), dtype=torch.int64, device=device)
    fixed_covalent = int((topology.fixed_edge_kind == COVALENT_EDGE).sum())
    fixed_chain = int((topology.fixed_edge_kind != COVALENT_EDGE).sum())
    stats = HeavyFlowGraphStats(
        num_nodes=int(packed.shape[0]),
        num_edges=int(edge_index.shape[1]),
        num_covalent_edges=fixed_covalent,
        num_spatial_edges=int((edge_kind == SPATIAL_EDGE).sum()),
        num_chain_adjacent_edges=fixed_chain,
        spatial_candidates=candidates,
        spatial_truncated_destinations=truncated,
        maximum_spatial_degree=maximum,
    )
    state = PackedAtomState(
        current=packed,
        history=packed.new_empty((packed.shape[0], 0, 3)),
        batch=topology.atom_batch,
        local_atom=topology.atom_local,
        packed_index=_packed_index(topology),
    )
    return HeavyFlowAtomGraph(edge_index, edge_kind, bond_type, edge_batch, state, stats)


def _packed_index(topology: FlowAtomTopology) -> Tensor:
    result = torch.full((topology.batch_size, topology.max_atoms), -1, dtype=torch.int64, device=topology.atom_mask.device)
    if topology.atom_batch.numel():
        result[topology.atom_batch, topology.atom_local] = torch.arange(
            topology.atom_batch.shape[0], dtype=torch.int64, device=topology.atom_batch.device
        )
    return result


def coerce_flow_topology(
    topology: FlowAtomTopology | HeavyFlowAtomGraph,
    *,
    atom_mask: Optional[Tensor] = None,
    atom_to_residue: Optional[Tensor] = None,
    atom_type: Optional[Tensor] = None,
    atom_name: Optional[Tensor] = None,
    residue_mask: Optional[Tensor] = None,
    chain_break: Optional[Tensor] = None,
    spatial_cutoff_angstrom: float = 6.0,
    max_spatial_neighbors: int = 32,
) -> FlowAtomTopology:
    if isinstance(topology, FlowAtomTopology):
        return topology
    return FlowAtomTopology.from_graph(
        topology,
        atom_mask=atom_mask,
        atom_to_residue=atom_to_residue,
        atom_type=atom_type,
        atom_name=atom_name,
        residue_mask=residue_mask,
        chain_break=chain_break,
        spatial_cutoff_angstrom=spatial_cutoff_angstrom,
        max_spatial_neighbors=max_spatial_neighbors,
    )
