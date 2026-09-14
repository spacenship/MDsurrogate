"""The hierarchical physics encoder: ``A -> R -> B -> R -> A``, twice.

One cycle is::

    atom interaction          local chemistry, 3-body, within r_cut
    atom -> residue pool       equivariant, size-normalised
    residue -> backbone        pooled irreps + PLM/temperature scalars
    backbone interaction       sequence +-1/+-2 and CA kNN: long-range context
    backbone -> residue        gated global context back down
    residue -> atom broadcast  gated, not a raw copy

Phase 1 small runs two cycles. Phase 2 loads these same modules and only adds
temporal/transition machinery on top of ``physics_latent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..conditioning.esm2 import ESM2_EMBED_DIM
from ..conditioning.residue import ResidueConditioner
from ..data.contracts import HierarchicalProteinBatch
from ..data.residue_constants import NUM_ATOM_NAMES
from ..geometry.frames import ResidueFrames, atom_local_coordinates
from ..graph.edges import edge_geometry, edge_spherical_harmonics, merge_edge_sets
from ..graph.hierarchy import HierarchicalGraph
from .blocks import AtomInteractionBlock, BackboneInteractionBlock
from .irreps import IrrepsConfig
from .vertical import (
    AtomToResiduePool,
    BackboneToResidue,
    ResidueToAtomBroadcast,
    ResidueToBackboneInjection,
)

__all__ = [
    "EncoderConfig",
    "EncoderOutput",
    "PairIntermediates",
    "AtomEmbedding",
    "HierarchicalPhysicsEncoder",
]

_MAX_Z = 20  # covers H, C, N, O, S with room to spare


@dataclass(frozen=True)
class EncoderConfig:
    """Phase 1 small configuration.

    Every field is a width or a depth. Phase 2 changes these numbers; it does not
    change which classes are used, so a Phase 1 checkpoint remains loadable.
    """

    irreps: IrrepsConfig = field(default_factory=IrrepsConfig)
    num_cycles: int = 2
    num_radial_basis: int = 8
    atom_cutoff: float = 5.0
    backbone_cutoff: float = 13.0
    atom_avg_neighbors: float = 21.4
    backbone_avg_neighbors: float = 20.0
    plm_dim: int = ESM2_EMBED_DIM
    plm_projection_dim: int = 128
    residue_scalar_channels: int = 64
    element_embedding_dim: int = 16
    atom_name_embedding_dim: int = 16
    # ablation hooks (Checkpoint 8)
    use_plm: bool = True
    use_temperature: bool = True
    use_atom_branch: bool = True
    use_backbone_branch: bool = True
    use_body_order_3: bool = True


@dataclass
class PairIntermediates:
    """Per-edge messages of the residue-pair (backbone) level, **global frame**.

    This is the tensor a single message pass actually computes and then throws
    away: ``m_ij = TP(h_j, Y(r_ij); w(d_ij, type, h_i, h_j))`` for the edge
    ``src=j -> dst=i``, before ``scatter_sum`` collapses it onto the receiving
    node. It is the closest thing Phase 1 has to a *pair interaction latent*, and
    it exists only because :meth:`HierarchicalPhysicsEncoder.forward` was asked
    for it -- the default path is unchanged.

    Taken from the **last backbone block of the last cycle**, so the node features
    feeding it have been through the whole hierarchy.

    Args:
        src / dst: ``[E]`` residue indices. Messages flow ``src -> dst``.
        edge_type: ``[E]`` relation sub-type (sequence offset bucket or spatial).
        num_types: size of that vocabulary, for embedding tables downstream.
        message: ``[E, irreps.dim]`` the message itself, carrying ``irreps``.
        irreps: message irreps, e.g. ``88x0e+104x1o+96x2e`` at the Phase 1 widths.
            **Not** the node irreps: a ``uvu`` product produces its own set.
        distance: ``[E]`` ``|r_ij|`` in Angstrom.
        unit_vector: ``[E, 3]`` direction from ``src`` to ``dst``, global frame.

    There is no triplet or higher-order edge object here and none is invented.
    Phase 1's body-order-3 term is a *node*-level ``o3.TensorSquare``, so that
    information already lives in ``residue_features``.
    """

    src: Tensor
    dst: Tensor
    edge_type: Tensor
    num_types: int
    message: Tensor
    irreps: o3.Irreps
    distance: Tensor
    unit_vector: Tensor

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    def _map(self, fn) -> "PairIntermediates":
        from dataclasses import replace as _replace

        return _replace(
            self,
            src=fn(self.src),
            dst=fn(self.dst),
            edge_type=fn(self.edge_type),
            message=fn(self.message),
            distance=fn(self.distance),
            unit_vector=fn(self.unit_vector),
        )

    def to(self, device) -> "PairIntermediates":
        return self._map(lambda t: t.to(device))

    def detach(self) -> "PairIntermediates":
        return self._map(lambda t: t.detach())


@dataclass
class EncoderOutput:
    """Per-level equivariant features, all in the **global frame**.

    Args:
        atom_features: ``[N_atom, D]``.
        residue_features: ``[N_res, D]`` -- this is ``physics_latent``, the
            tensor Phase 2 conditions on. Its irreps and row order (aligned with
            ``batch.residues``) are a frozen contract.
        backbone_features: ``[N_res, D]``.
        irreps: the irreps every one of the above carries.
        pair: residue-pair messages, only when they were asked for. ``None`` on
            the default path, so nothing that existed before pays for them.
    """

    atom_features: Tensor
    residue_features: Tensor
    backbone_features: Tensor
    irreps: o3.Irreps
    pair: Optional[PairIntermediates] = None

    @property
    def physics_latent(self) -> Tensor:
        return self.residue_features


class AtomEmbedding(nn.Module):
    """Initial atom features: invariant scalars only, ``l>0`` starts at zero.

    The scalars include the residue-local coordinate ``y_ia`` and its norm. That
    is where **chirality** enters the network: ``y_ia`` is built with the residue
    frame whose third axis is a cross product, so a mirrored structure produces
    different scalars. Without it the encoder -- built from spherical harmonics of
    relative positions -- would be accidentally E(3)-equivariant and unable to
    tell an L-protein from its D mirror image.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.irreps = config.irreps.node_irreps()
        first_mul, first_ir = self.irreps[0]
        if first_ir.l != 0:
            raise ValueError("node irreps must start with the scalar block")
        self.num_scalars = first_mul

        self.element_embedding = nn.Embedding(_MAX_Z + 1, config.element_embedding_dim)
        self.name_embedding = nn.Embedding(NUM_ATOM_NAMES, config.atom_name_embedding_dim)
        in_dim = (
            config.element_embedding_dim
            + config.atom_name_embedding_dim
            + 2   # is_backbone, is_cap
            + 4   # y_ia (3) and |y_ia|
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.num_scalars),
            nn.SiLU(),
            nn.Linear(self.num_scalars, self.num_scalars),
        )

    def forward(self, batch: HierarchicalProteinBatch, local_coords: Tensor) -> Tensor:
        atoms = batch.atoms
        scalars = self.mlp(
            torch.cat(
                [
                    self.element_embedding(atoms.atomic_number.clamp(max=_MAX_Z)),
                    self.name_embedding(atoms.atom_name_id),
                    atoms.is_backbone.unsqueeze(-1).to(local_coords.dtype),
                    atoms.is_cap.unsqueeze(-1).to(local_coords.dtype),
                    local_coords,
                    torch.linalg.norm(local_coords, dim=-1, keepdim=True),
                ],
                dim=-1,
            )
        )
        features = local_coords.new_zeros((atoms.num_atoms, self.irreps.dim))
        features[:, : self.num_scalars] = scalars
        return features


class HierarchicalPhysicsEncoder(nn.Module):
    """Two-cycle ``A -> R -> B -> R -> A`` equivariant encoder at ``l_max=2``.

    Shape:
        ``(batch, graph) -> EncoderOutput`` with all features in
        ``64x0e + 16x1o + 8x2e`` at the default config.
    """

    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config
        self.irreps = config.irreps.node_irreps()
        self.irreps_sh = config.irreps.sh_irreps()

        self.atom_embedding = AtomEmbedding(config)
        self.residue_conditioner = ResidueConditioner(
            plm_dim=config.plm_dim,
            plm_projection_dim=config.plm_projection_dim,
            out_channels=config.residue_scalar_channels,
            use_plm=config.use_plm,
            use_temperature=config.use_temperature,
        )
        self.residue_init = o3.Linear(
            o3.Irreps(f"{config.residue_scalar_channels}x0e"), self.irreps
        )

        n_cycles = config.num_cycles
        # atom relations: bonded(2) + spatial(2); backbone: sequence(5) + spatial(1)
        self.atom_blocks = nn.ModuleList(
            AtomInteractionBlock(
                self.irreps, self.irreps_sh, num_relation_types=4,
                r_cut=config.atom_cutoff, avg_num_neighbors=config.atom_avg_neighbors,
                num_radial_basis=config.num_radial_basis,
                use_body_order_3=config.use_body_order_3,
            )
            for _ in range(n_cycles)
        )
        self.backbone_blocks = nn.ModuleList(
            BackboneInteractionBlock(
                self.irreps, self.irreps_sh, num_relation_types=6,
                r_cut=config.backbone_cutoff,
                avg_num_neighbors=config.backbone_avg_neighbors,
                num_radial_basis=config.num_radial_basis,
                use_body_order_3=config.use_body_order_3,
            )
            for _ in range(n_cycles)
        )
        self.pools = nn.ModuleList(
            AtomToResiduePool(self.irreps) for _ in range(n_cycles)
        )
        self.injections = nn.ModuleList(
            ResidueToBackboneInjection(self.irreps, config.residue_scalar_channels)
            for _ in range(n_cycles)
        )
        self.backbone_to_residue = nn.ModuleList(
            BackboneToResidue(self.irreps) for _ in range(n_cycles)
        )
        self.residue_to_atom = nn.ModuleList(
            ResidueToAtomBroadcast(self.irreps) for _ in range(n_cycles)
        )

    def forward(
        self,
        batch: HierarchicalProteinBatch,
        graph: HierarchicalGraph,
        frames: Optional[ResidueFrames] = None,
        *,
        return_pair_messages: bool = False,
    ) -> EncoderOutput:
        """Encode one batch.

        Args:
            return_pair_messages: also return the last backbone block's per-edge
                messages as :class:`PairIntermediates`. Default off: the Phase 1
                training path must keep the memory profile it was tuned at.
        """
        local_coords, frames = atom_local_coordinates(batch, frames)

        atom_edges = merge_edge_sets(
            [graph.atom_bonded, graph.atom_spatial], "atom__message__atom"
        )
        bb_edges = merge_edge_sets(
            [graph.backbone_sequence, graph.backbone_spatial], "backbone__message__backbone"
        )
        atom_geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, atom_edges)
        bb_geom = edge_geometry(
            batch.backbone.ca_positions, batch.backbone.ca_positions, bb_edges
        )
        atom_sh = edge_spherical_harmonics(atom_geom.unit_vector, self.config.irreps.lmax)
        bb_sh = edge_spherical_harmonics(bb_geom.unit_vector, self.config.irreps.lmax)

        atom_features = self.atom_embedding(batch, local_coords)
        residue_scalars = self.residue_conditioner(batch)
        residue_features = self.residue_init(residue_scalars)
        backbone_features = residue_features.new_zeros(
            (batch.num_residues, self.irreps.dim)
        )

        pair_messages: Optional[Tensor] = None
        last_cycle = self.config.num_cycles - 1
        for c in range(self.config.num_cycles):
            if self.config.use_atom_branch:
                atom_features = self.atom_blocks[c](
                    atom_features, atom_edges, atom_sh, atom_geom.distance
                )
            pooled = self.pools[c](
                atom_features, batch.atoms.atom_to_residue, batch.num_residues
            )
            backbone_features = self.injections[c](
                backbone_features, pooled, residue_scalars
            )
            if self.config.use_backbone_branch:
                want = return_pair_messages and c == last_cycle
                result = self.backbone_blocks[c](
                    backbone_features, bb_edges, bb_sh, bb_geom.distance,
                    return_messages=want,
                )
                if want:
                    backbone_features, pair_messages = result
                else:
                    backbone_features = result
                residue_features = self.backbone_to_residue[c](
                    residue_features, backbone_features, batch.backbone.residue_to_backbone
                )
            else:
                residue_features = residue_features + pooled
            atom_features = self.residue_to_atom[c](
                atom_features, residue_features, batch.atoms.atom_to_residue
            )

        pair = None
        if return_pair_messages:
            if not self.config.use_backbone_branch:
                raise ValueError(
                    "pair messages were requested but use_backbone_branch is False: "
                    "there is no residue-pair message pass to read them from"
                )
            block = self.backbone_blocks[last_cycle]
            if pair_messages is None:
                # An empty residue graph is legal (one-residue protein); an empty
                # message tensor of the right width keeps every consumer's shape
                # arithmetic valid instead of making them special-case None.
                pair_messages = residue_features.new_zeros((0, block.irreps_message.dim))
            pair = PairIntermediates(
                src=bb_edges.src,
                dst=bb_edges.dst,
                edge_type=bb_edges.edge_type,
                num_types=bb_edges.num_types,
                message=pair_messages,
                irreps=block.irreps_message,
                distance=bb_geom.distance,
                unit_vector=bb_geom.unit_vector,
            )

        return EncoderOutput(
            atom_features=atom_features,
            residue_features=residue_features,
            backbone_features=backbone_features,
            irreps=self.irreps,
            pair=pair,
        )
