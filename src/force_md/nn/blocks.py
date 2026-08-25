"""Equivariant message-passing blocks.

One implementation, :class:`EquivariantMessageBlock`, serves both levels;
:class:`AtomInteractionBlock` and :class:`BackboneInteractionBlock` are thin
configurations of it. Phase 2 grows depth and channels by changing the config,
never by substituting a different class, so a Phase 1 checkpoint keeps loading.

**Body order.** A single message pass is a 2-body interaction: each edge mixes
one neighbour into one node. After aggregation the block applies an explicit
``o3.TensorSquare`` to the updated node feature and mixes the result back. A
square of a sum over neighbours contains pair-of-neighbour cross terms, so one
block reaches **correlation order 3** (3-body: the centre plus two neighbours),
and the 2-cycle Phase 1 encoder reaches order 5 in the receptive field's sense.
This is the MACE idea -- raise body order inside a layer instead of stacking more
2-body layers -- implemented directly here rather than by importing MACE.
"""

from __future__ import annotations

from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..graph.edges import EdgeSet
from .irreps import GatedLinear, scatter_sum
from .radial import BesselBasis, RadialMLP

__all__ = [
    "EquivariantMessageBlock",
    "AtomInteractionBlock",
    "BackboneInteractionBlock",
    "extract_scalars",
]


def extract_scalars(x: Tensor, irreps: o3.Irreps) -> Tensor:
    """Slice out the ``l=0`` block of an irreps feature.

    These are the only components that may be fed to an ordinary MLP; feeding an
    ``l>0`` component to one would break equivariance silently.
    """
    out = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l == 0:
            out.append(x[:, offset : offset + width])
        offset += width
    if not out:
        return x.new_zeros((x.shape[0], 0))
    return torch.cat(out, dim=-1)


def scalar_dim(irreps: o3.Irreps) -> int:
    return sum(mul for mul, ir in irreps if ir.l == 0)


def _build_uvu_tensor_product(
    irreps_node: o3.Irreps, irreps_sh: o3.Irreps
) -> tuple[o3.TensorProduct, o3.Irreps]:
    """Depthwise (``uvu``) tensor product, the NequIP/MACE convolution kernel.

    A fully-connected (``uvw``) product needs one weight per
    (in-channel, sh-channel, out-channel) triple: 8064 weights **per edge** at
    the Phase 1 widths. With ~21 neighbours per atom that is 1.35 GB of weights
    alone for a two-protein batch, which is what made the first real training run
    run out of memory on an 80 GB card.

    ``uvu`` instead keeps the input multiplicity and weights each path once: 288
    weights per edge, a 28x reduction, with the same set of irreps reachable.
    Output irreps are restricted to the parities the node features actually
    carry (``0e``, ``1o``, ``2e``); the ``1e``/``2o`` paths a general product
    would also produce are unused by the node representation.

    Returns:
        ``(tensor_product, output_irreps)``. The output irreps are **sorted but
        not simplified** -- simplifying merges entries and would silently
        invalidate the instruction indices that point at them.
    """
    wanted = {(ir.l, ir.p) for _, ir in irreps_node}
    out_list: list[tuple[int, o3.Irrep]] = []
    instructions: list[tuple] = []
    for i, (mul, ir_in) in enumerate(irreps_node):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                if (ir_out.l, ir_out.p) in wanted:
                    instructions.append((i, j, len(out_list), "uvu", True))
                    out_list.append((mul, ir_out))
    irreps_out = o3.Irreps(out_list)
    irreps_out, permutation, _ = irreps_out.sort()
    instructions = [
        (i, j, permutation[k], mode, trainable)
        for (i, j, k, mode, trainable) in instructions
    ]
    tp = o3.TensorProduct(
        irreps_node, irreps_sh, irreps_out, instructions,
        shared_weights=False, internal_weights=False,
    )
    return tp, irreps_out


class EquivariantMessageBlock(nn.Module):
    """Relation-conditioned equivariant convolution with a 3-body path.

    Args:
        irreps_node: node feature irreps, in the global frame.
        irreps_sh: spherical-harmonic irreps of the edge direction.
        num_relation_types: size of the edge sub-type vocabulary. Distinct
            relations (bond vs. spatial, sequence offset +1 vs. +2) keep distinct
            ids, so the block conditions on relation identity rather than
            collapsing them into "an edge".
        r_cut: cutoff used by the radial basis; must match the graph's cutoff or
            the envelope will not reach zero at the neighbour-list boundary.
        avg_num_neighbors: aggregated messages are divided by its square root so
            activations do not scale with coordination number. Defaults are
            measured on real mdCATH, not guessed.
        extra_scalar_dim: width of extra invariant per-node conditioning
            (e.g. residue context) concatenated into the radial MLP input.

    Shape:
        ``node_features [N, irreps_node.dim] -> [N, irreps_node.dim]`` (residual).
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        *,
        num_relation_types: int,
        r_cut: float,
        num_radial_basis: int = 8,
        avg_num_neighbors: float = 20.0,
        relation_embedding_dim: int = 16,
        radial_hidden: int = 64,
        use_body_order_3: bool = True,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.use_body_order_3 = use_body_order_3

        self.radial_basis = BesselBasis(r_cut, num_radial_basis)
        self.relation_embedding = nn.Embedding(num_relation_types, relation_embedding_dim)

        self.tp, tp_out_irreps = _build_uvu_tensor_product(
            self.irreps_node, self.irreps_sh
        )
        # Linear is applied *after* the scatter, not per edge: a linear map
        # commutes with summation, and doing it on N nodes instead of E edges is
        # where most of the activation memory is saved.
        self.post_message = o3.Linear(tp_out_irreps, self.irreps_node)
        n_scalar = scalar_dim(self.irreps_node)
        edge_scalar_dim = num_radial_basis + relation_embedding_dim + 2 * n_scalar
        self.radial_mlp = RadialMLP(edge_scalar_dim, self.tp.weight_numel,
                                    hidden=radial_hidden)

        self.self_interaction = o3.Linear(self.irreps_node, self.irreps_node)
        if use_body_order_3:
            self.tensor_square = o3.TensorSquare(self.irreps_node,
                                                 irreps_out=self.irreps_node)
            self.square_mix = o3.Linear(self.irreps_node, self.irreps_node)
        self.output = GatedLinear(self.irreps_node, self.irreps_node)

    def forward(
        self,
        node_features: Tensor,
        edges: EdgeSet,
        edge_sh: Tensor,
        edge_distance: Tensor,
    ) -> Tensor:
        """One residual update of ``node_features``.

        An empty relation is a no-op, not an error: ``scatter_sum`` returns exact
        zeros, so a protein whose cutoff graph is empty still produces finite
        output and finite gradients.
        """
        n = node_features.shape[0]
        h = self.self_interaction(node_features)

        if edges.num_edges > 0:
            scalars = extract_scalars(node_features, self.irreps_node)
            edge_input = torch.cat(
                [
                    self.radial_basis(edge_distance),
                    self.relation_embedding(edges.edge_type),
                    scalars[edges.src],
                    scalars[edges.dst],
                ],
                dim=-1,
            )
            weights = self.radial_mlp(edge_input)
            messages = self.tp(node_features[edges.src], edge_sh, weights)
            pooled = scatter_sum(messages, edges.dst, n) / (self.avg_num_neighbors**0.5)
            h = h + self.post_message(pooled)

        if self.use_body_order_3:
            h = h + self.square_mix(self.tensor_square(h))

        return node_features + self.output(h)


class AtomInteractionBlock(EquivariantMessageBlock):
    """Atom-level block.

    ``avg_num_neighbors=21.4`` is the measured mean heavy-atom neighbour count at
    the 5.0 A default cutoff across six real mdCATH domains.
    """

    def __init__(self, irreps_node, irreps_sh, *, num_relation_types: int,
                 r_cut: float = 5.0, avg_num_neighbors: float = 21.4, **kwargs):
        super().__init__(irreps_node, irreps_sh,
                         num_relation_types=num_relation_types, r_cut=r_cut,
                         avg_num_neighbors=avg_num_neighbors, **kwargs)


class BackboneInteractionBlock(EquivariantMessageBlock):
    """Backbone-level block over sequence and spatial relations.

    ``r_cut=13.0`` A covers the measured p95 radius of the 16 nearest CA
    neighbours (10.2 A mean), so the envelope does not clip real neighbours.
    ``avg_num_neighbors=20`` = 16 spatial + ~4 sequence.
    """

    def __init__(self, irreps_node, irreps_sh, *, num_relation_types: int,
                 r_cut: float = 13.0, avg_num_neighbors: float = 20.0, **kwargs):
        super().__init__(irreps_node, irreps_sh,
                         num_relation_types=num_relation_types, r_cut=r_cut,
                         avg_num_neighbors=avg_num_neighbors, **kwargs)
