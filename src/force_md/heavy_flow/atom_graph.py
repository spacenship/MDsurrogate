"""Deterministic atom graph construction for heavy-flow conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data.residue_constants import ATOM_NAME_TO_ID
from .types import HeavyFlowCondition

__all__ = [
    "COVALENT_EDGE", "SPATIAL_EDGE", "CHAIN_ADJACENT_EDGE",
    "HeavyFlowGraphConfig", "HeavyFlowGraphStats", "PackedAtomState",
    "HeavyFlowAtomGraph", "pack_atom_state", "build_atom_graph",
]

COVALENT_EDGE = 0
SPATIAL_EDGE = 1
CHAIN_ADJACENT_EDGE = 2


@dataclass(frozen=True)
class HeavyFlowGraphConfig:
    spatial_cutoff_angstrom: float = 6.0
    max_spatial_neighbors: int = 32
    include_covalent_edges: bool = True
    include_chain_adjacent_edges: bool = False

    def __post_init__(self) -> None:
        if self.spatial_cutoff_angstrom <= 0:
            raise ValueError("spatial_cutoff_angstrom must be positive")
        if self.max_spatial_neighbors <= 0:
            raise ValueError("max_spatial_neighbors must be positive")


@dataclass(frozen=True)
class HeavyFlowGraphStats:
    num_nodes: int
    num_edges: int
    num_covalent_edges: int
    num_spatial_edges: int
    num_chain_adjacent_edges: int
    spatial_candidates: int
    spatial_truncated_destinations: int
    maximum_spatial_degree: int

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PackedAtomState:
    current: Tensor
    history: Tensor
    batch: Tensor
    local_atom: Tensor
    packed_index: Tensor


@dataclass(frozen=True)
class HeavyFlowAtomGraph:
    edge_index: Tensor
    edge_kind: Tensor
    bond_type: Tensor
    edge_batch: Tensor
    state: PackedAtomState
    stats: HeavyFlowGraphStats

    @property
    def num_nodes(self) -> int:
        return int(self.state.current.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "HeavyFlowAtomGraph":
        return HeavyFlowAtomGraph(
            self.edge_index.to(device), self.edge_kind.to(device), self.bond_type.to(device),
            self.edge_batch.to(device), PackedAtomState(
                self.state.current.to(device), self.state.history.to(device),
                self.state.batch.to(device), self.state.local_atom.to(device),
                self.state.packed_index.to(device),
            ), self.stats,
        )


def pack_atom_state(condition: HeavyFlowCondition) -> PackedAtomState:
    condition.validate()
    b, n = condition.batch_size, condition.max_atoms
    packed_index = torch.full((b, n), -1, dtype=torch.int64, device=condition.atom_mask.device)
    batch_parts: list[Tensor] = []
    local_parts: list[Tensor] = []
    current_parts: list[Tensor] = []
    history_parts: list[Tensor] = []
    offset = 0
    for batch in range(b):
        local = torch.nonzero(condition.atom_mask[batch], as_tuple=False).flatten()
        count = int(local.numel())
        packed_index[batch, local] = torch.arange(offset, offset + count, device=local.device)
        batch_parts.append(torch.full((count,), batch, dtype=torch.int64, device=local.device))
        local_parts.append(local)
        current_parts.append(condition.x_history[batch, -1, local])
        history_parts.append(condition.x_history[batch, :, local].transpose(0, 1))
        offset += count
    if not batch_parts:
        empty = condition.x_history.new_empty((0,))
        return PackedAtomState(
            empty.reshape(0, 3), empty.reshape(0, condition.history_length, 3),
            torch.empty(0, dtype=torch.int64, device=condition.atom_mask.device),
            torch.empty(0, dtype=torch.int64, device=condition.atom_mask.device), packed_index,
        )
    return PackedAtomState(
        torch.cat(current_parts, dim=0), torch.cat(history_parts, dim=0),
        torch.cat(batch_parts, dim=0), torch.cat(local_parts, dim=0), packed_index,
    )


def _top_k_spatial(positions: Tensor, *, cutoff: float, max_neighbors: int) -> tuple[list[tuple[int, int]], int, int, int]:
    """Return directed ``(source,destination)`` edges with deterministic ties."""
    count = int(positions.shape[0])
    if count <= 1:
        return [], 0, 0, 0
    # Neighbor selection is discrete: do not retain its autograd graph. Tile
    # destinations to bound scratch memory; stable sorting preserves source-ID
    # tie breaking without thousands of GPU scalar reads/Python sorts.
    edges: list[tuple[int, int]] = []
    degrees = []
    edge_parts = []
    source_ids = torch.arange(count, device=positions.device)[:, None]
    mode = "use_mm_for_euclid_dist" if count > 25 else "donot_use_mm_for_euclid_dist"
    with torch.no_grad():
        for start in range(0, count, 256):
            destinations = torch.arange(start, min(start + 256, count), device=positions.device)
            distances = torch.cdist(positions.detach(), positions.detach()[destinations], compute_mode=mode)
            valid = (distances <= cutoff) & (source_ids != destinations[None, :])
            degrees.append(valid.sum(dim=0))
            ranked = distances.masked_fill(~valid, float("inf")).argsort(dim=0, stable=True)[:max_neighbors]
            selected = valid.gather(0, ranked).t()
            pairs = torch.stack((ranked.t(), destinations[:, None].expand_as(ranked.t())), dim=-1)
            edge_parts.append(pairs[selected])
        degree = torch.cat(degrees)
        stats = torch.stack((degree.sum(), (degree > max_neighbors).sum(), degree.clamp_max(max_neighbors).max())).cpu().tolist()
        edges = [tuple(pair) for pair in torch.cat(edge_parts).cpu().tolist()]
    return edges, *map(int, stats)


def build_atom_graph(condition: HeavyFlowCondition, config: Optional[HeavyFlowGraphConfig] = None) -> HeavyFlowAtomGraph:
    """Build covalent + radius/kNN edges while preserving PSF covalent edges."""
    config = config or HeavyFlowGraphConfig()
    condition.validate()
    state = pack_atom_state(condition)
    # Topology is discrete metadata. Transfer in bulk, never one CUDA scalar
    # per bond/edge. Differentiable positions/history remain on their device.
    atom_mask = condition.atom_mask.cpu()
    packed_index = state.packed_index.cpu()
    bond_index = condition.bond_index.cpu()
    bond_types = condition.bond_type.cpu()
    edge_rows: list[tuple[int, int, int, int, int]] = []
    covalent_seen: set[tuple[int, int]] = set()
    covalent_count = 0
    chain_count = 0
    b, e = condition.batch_size, condition.bond_index.shape[-1]
    if config.include_covalent_edges:
        for batch in range(b):
            for slot in range(e):
                a = int(bond_index[batch, 0, slot])
                z = int(bond_index[batch, 1, slot])
                if a < 0 or z < 0 or not bool(atom_mask[batch, a]) or not bool(atom_mask[batch, z]):
                    continue
                source = int(packed_index[batch, a])
                destination = int(packed_index[batch, z])
                key = (min(source, destination), max(source, destination))
                if key in covalent_seen:
                    continue
                covalent_seen.add(key)
                bond = int(bond_types[batch, slot])
                edge_rows.extend(((source, destination, COVALENT_EDGE, bond, batch),
                                  (destination, source, COVALENT_EDGE, bond, batch)))
                covalent_count += 2

    spatial_count = 0
    spatial_candidates = 0
    spatial_truncated = 0
    maximum_degree = 0
    for batch in range(b):
        local = torch.nonzero(condition.atom_mask[batch], as_tuple=False).flatten()
        spatial, candidates, truncated, maximum = _top_k_spatial(
            condition.current_positions[batch, local],
            cutoff=config.spatial_cutoff_angstrom,
            max_neighbors=config.max_spatial_neighbors,
        )
        spatial_candidates += candidates
        spatial_truncated += truncated
        maximum_degree = max(maximum_degree, maximum)
        packed_local = packed_index[batch, local.cpu()].tolist()
        for source_local, destination_local in spatial:
            source = packed_local[source_local]
            destination = packed_local[destination_local]
            edge_rows.append((source, destination, SPATIAL_EDGE, 0, batch))
            spatial_count += 1

        if config.include_chain_adjacent_edges and local.numel():
            c_id, n_id = ATOM_NAME_TO_ID.get("C", -1), ATOM_NAME_TO_ID.get("N", -1)
            for residue in range(condition.sequence_length - 1):
                if condition.chain_break is not None and bool(condition.chain_break[batch, residue]):
                    continue
                left = torch.nonzero(condition.atom_mask[batch] &
                                     (condition.atom_to_residue[batch] == residue) &
                                     (condition.atom_name[batch] == c_id), as_tuple=False).flatten()
                right = torch.nonzero(condition.atom_mask[batch] &
                                      (condition.atom_to_residue[batch] == residue + 1) &
                                      (condition.atom_name[batch] == n_id), as_tuple=False).flatten()
                if left.numel() and right.numel():
                    source = int(state.packed_index[batch, int(left[0])])
                    destination = int(state.packed_index[batch, int(right[0])])
                    edge_rows.extend(((source, destination, CHAIN_ADJACENT_EDGE, 0, batch),
                                      (destination, source, CHAIN_ADJACENT_EDGE, 0, batch)))
                    chain_count += 2

    if edge_rows:
        edge_rows.sort(key=lambda row: (row[4], row[2], row[1], row[0], row[3]))
        edge_index = torch.tensor([(r[0], r[1]) for r in edge_rows], dtype=torch.int64, device=state.current.device).t()
        edge_kind = torch.tensor([r[2] for r in edge_rows], dtype=torch.int64, device=state.current.device)
        bond_type = torch.tensor([r[3] for r in edge_rows], dtype=torch.int64, device=state.current.device)
        edge_batch = torch.tensor([r[4] for r in edge_rows], dtype=torch.int64, device=state.current.device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64, device=state.current.device)
        edge_kind = torch.empty((0,), dtype=torch.int64, device=state.current.device)
        bond_type = torch.empty((0,), dtype=torch.int64, device=state.current.device)
        edge_batch = torch.empty((0,), dtype=torch.int64, device=state.current.device)
    stats = HeavyFlowGraphStats(
        num_nodes=state.current.shape[0], num_edges=edge_index.shape[1],
        num_covalent_edges=covalent_count, num_spatial_edges=spatial_count,
        num_chain_adjacent_edges=chain_count, spatial_candidates=spatial_candidates,
        spatial_truncated_destinations=spatial_truncated, maximum_spatial_degree=maximum_degree,
    )
    return HeavyFlowAtomGraph(edge_index, edge_kind, bond_type, edge_batch, state, stats)
