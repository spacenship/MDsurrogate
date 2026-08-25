"""Assembly of the full three-level hierarchical graph.

Six relations, in the order information flows through them::

    atom__bonded__atom            A - A   covalent
    atom__spatial__atom           A - A   within cutoff, intra/inter-residue
    residue__contains__atom       R -> A  (+ reverse: atom -> residue pooling)
    backbone__owns__residue       B -> R  (+ reverse)
    backbone__sequence__backbone  B - B   +-1, +-2, chain-break aware
    backbone__spatial__backbone   B - B   kNN on CA

The vertical relations are stored once and reversed on demand, so the
parent/child mapping can never disagree with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data.contracts import HierarchicalProteinBatch
from .edges import (
    EdgeSet,
    build_covalent_bonds,
    build_knn_edges,
    build_radius_edges,
    build_sequence_edges,
    build_vertical_edges,
)

__all__ = ["GraphConfig", "HierarchicalGraph", "build_hierarchical_graph"]


@dataclass(frozen=True)
class GraphConfig:
    """Topology hyper-parameters.

    Defaults come from an audit of six real mdCATH domains (1a0rP01, 1a9xB01,
    1aamA02, 1ad3A02, 1adnA00, 1ahsA00), not from a guess:

    ============  =====================  =====================
    atom cutoff   heavy-atom neighbours  all-atom neighbours
    ============  =====================  =====================
    4.5 A         15.2 mean / 23 p95     30.0 mean / 43 p95
    **5.0 A**     **21.4 mean / 32 p95** 41.2 mean / 58 p95
    6.0 A         35.4 mean / 50 p95     68.1 mean / 97 p95
    ============  =====================  =====================

    5.0 A is the heavy-atom default: it spans the first two coordination shells
    at ~21 neighbours per atom, which is the range MACE-style models use, while
    6.0 A costs 65% more edges for interactions that the backbone level already
    carries. For ``all_atom`` mode 4.5 A gives a comparable edge budget.

    ``residue_knn = 16`` corresponds to a mean CA radius of 10.2 A (p95 13.0 A)
    on the same audit, so it adapts to packing density instead of fixing a radius.
    """

    sequence_max_offset: int = 2
    require_contiguous_resid: bool = True
    residue_knn: int = 16
    residue_cutoff: Optional[float] = None
    atom_cutoff: float = 5.0
    atom_include_inter_residue: bool = True
    bond_tolerance: float = 1.3
    lmax: int = 2


@dataclass
class HierarchicalGraph:
    """All six relations for one batch."""

    backbone_sequence: EdgeSet
    backbone_spatial: EdgeSet
    backbone_owns_residue: EdgeSet
    residue_contains_atom: EdgeSet
    atom_bonded: EdgeSet
    atom_spatial: EdgeSet
    config: GraphConfig

    # -- reverse views (derived, never stored separately) ------------------
    @property
    def residue_owned_by_backbone(self) -> EdgeSet:
        return self.backbone_owns_residue.reverse("residue__owned_by__backbone")

    @property
    def atom_belongs_to_residue(self) -> EdgeSet:
        return self.residue_contains_atom.reverse("atom__belongs_to__residue")

    def relations(self) -> dict[str, EdgeSet]:
        return {
            e.relation: e
            for e in (
                self.backbone_sequence, self.backbone_spatial,
                self.backbone_owns_residue, self.residue_contains_atom,
                self.atom_bonded, self.atom_spatial,
            )
        }

    def edge_counts(self) -> dict[str, int]:
        return {name: e.num_edges for name, e in self.relations().items()}

    def to(self, device) -> "HierarchicalGraph":
        import dataclasses
        return dataclasses.replace(
            self,
            **{f.name: getattr(self, f.name).to(device)
               for f in dataclasses.fields(self) if f.name != "config"},
        )

    def validate(self, batch: HierarchicalProteinBatch) -> None:
        """Check ranges, vertical cardinality and absence of batch leakage."""
        n_atom, n_res = batch.num_atoms, batch.num_residues
        sizes = {
            "backbone__sequence__backbone": (n_res, n_res),
            "backbone__spatial__backbone": (n_res, n_res),
            "backbone__owns__residue": (n_res, n_res),
            "residue__contains__atom": (n_res, n_atom),
            "atom__bonded__atom": (n_atom, n_atom),
            "atom__spatial__atom": (n_atom, n_atom),
        }
        for name, e in self.relations().items():
            ns, nd = sizes[name]
            e.validate(num_src_nodes=ns, num_dst_nodes=nd)

        # vertical cardinality: exactly one parent per child
        if self.residue_contains_atom.num_edges != n_atom:
            raise ValueError(
                f"residue__contains__atom has {self.residue_contains_atom.num_edges} "
                f"edges but there are {n_atom} atoms; containment must be 1 parent "
                "per child"
            )
        if self.backbone_owns_residue.num_edges != n_res:
            raise ValueError(
                f"backbone__owns__residue has {self.backbone_owns_residue.num_edges} "
                f"edges but there are {n_res} residues"
            )

        # no relation may connect two graphs
        atom_g, res_g = batch.atoms.batch_index, batch.residues.batch_index
        for name, e in self.relations().items():
            if e.num_edges == 0:
                continue
            gs = atom_g if name.startswith("atom") else res_g
            gd = atom_g if name.endswith("atom") else res_g
            if not bool(torch.equal(gs[e.src], gd[e.dst])):
                raise ValueError(f"{name}: edge crosses a graph boundary (batch leakage)")


def build_hierarchical_graph(
    batch: HierarchicalProteinBatch,
    config: GraphConfig = GraphConfig(),
    *,
    explicit_bonds: Optional[Tensor] = None,
) -> HierarchicalGraph:
    """Build all six relations for ``batch``.

    Args:
        batch: validated hierarchical state.
        config: topology hyper-parameters.
        explicit_bonds: optional ``[2, E]`` authoritative bond list (e.g. parsed
            from the mdCATH PSF). When given it replaces the distance heuristic;
            it is assumed to be within-graph and is symmetrised here.

    Returns:
        :class:`HierarchicalGraph`. The topology is a *discrete* function of the
        coordinates: gradients flow through edge features, never through the
        neighbour list itself. Rebuild the graph when coordinates move enough to
        change neighbours; do not claim the neighbour list is differentiable.
    """
    atoms, residues, bb = batch.atoms, batch.residues, batch.backbone

    sequence = build_sequence_edges(
        bb.batch_index, residues.chain_index, residues.resid_original,
        max_offset=config.sequence_max_offset,
        require_contiguous_resid=config.require_contiguous_resid,
    )
    spatial_bb = build_knn_edges(
        bb.ca_positions, bb.batch_index,
        k=config.residue_knn, cutoff=config.residue_cutoff,
    )
    owns = build_vertical_edges(bb.residue_to_backbone, "backbone__owns__residue")
    contains = build_vertical_edges(atoms.atom_to_residue, "residue__contains__atom")

    if explicit_bonds is not None:
        src, dst = explicit_bonds[0], explicit_bonds[1]
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])  # symmetrise
        typ = (atoms.atom_to_residue[src] != atoms.atom_to_residue[dst]).to(torch.int64)
        bonded = EdgeSet(src, dst, "atom__bonded__atom", typ, 2)
    else:
        bonded = build_covalent_bonds(
            atoms.positions, atoms.atomic_number, atoms.atom_to_residue,
            atoms.batch_index, tolerance=config.bond_tolerance,
        )

    spatial_atom = build_radius_edges(
        atoms.positions, atoms.atom_to_residue, atoms.batch_index,
        cutoff=config.atom_cutoff,
        include_inter_residue=config.atom_include_inter_residue,
    )
    return HierarchicalGraph(
        backbone_sequence=sequence,
        backbone_spatial=spatial_bb,
        backbone_owns_residue=owns,
        residue_contains_atom=contains,
        atom_bonded=bonded,
        atom_spatial=spatial_atom,
        config=config,
    )
