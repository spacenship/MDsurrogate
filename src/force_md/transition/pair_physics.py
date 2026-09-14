"""Residue-**pair** conditioners: the Phase 1.6 arms P0-P4.

Phase 1.5 asked whether Phase 1's *node* representation helps predict a 1-4 ns
transition. It does, by under 1%, and the oracle arm showed that is the ceiling
for instantaneous force at those lags. Phase 1.6 asks a different question with
the same machinery: is the **interaction** between two residues -- the thing a
message pass computes and then throws away -- worth more than the node summary it
was collapsed into?

The tensor this reads is real and is not new machinery. Inside every
:class:`~force_md.nn.blocks.EquivariantMessageBlock`::

    m_ij = TP(h_j, Y(r_ij); w(d_ij, type, h_i, h_j))     [E, 880]
    h_i <- ... + post_message(sum_j m_ij / sqrt(<n>))

Phase 1 keeps only the sum. ``m_ij`` is the pair latent, taken from the last
backbone block of the last cycle, where the residue-pair graph is sequence +-1/+-2
plus CA-kNN(16).

**The control is the point of the experiment.** ``P1`` beating ``S0`` would prove
nothing on its own -- it has more parameters and an extra pooling stage. So
``P0`` is the identical architecture with the pair latent replaced by a
geometry-only edge embedding *of a deliberately matched parameter count*: see
:func:`matched_hidden_width`. If ``P1`` beats ``P0`` the difference is what Phase 1
learned from force labels, because that is the only thing left.

**Everything a conditioner emits is an SE(3) invariant.** Edge messages are
rotated into the receiving residue's frame before they meet an MLP
(:class:`~force_md.transition.local_frame.IrrepsLocalFrame`), the edge direction
enters as ``R_i^T r_hat_ij``, and the relative orientation as ``R_i^T R_j``. A
conditioner that flattened a global-frame tensor would make the whole probe
rotation-dependent through the back door.

**Nothing here reads a label.** The pair message is a function of the current
structure computed by frozen weights; the moments in ``P2`` are moments of
Phase 1's *predicted* atom forces. Ground-truth forces reach only the oracle arm,
and :func:`~force_md.transition.phase1_features.assert_production_safe` is called
on the way into every conditioner below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..conditioning.lag import LAG_FEATURE_DIM, lag_features
from ..conditioning.temperature import TEMPERATURE_FEATURE_DIM, temperature_features
from ..nn.irreps import scatter_sum
from .conditioners import (
    ConditionerConfig,
    TransitionConditioner,
    _Adapter,
    precision_weights,
    register_conditioner,
)
from .local_frame import IrrepsLocalFrame
from .moments import MOMENT_FEATURE_DIM, ResidueShape, force_moments, residue_shape
from .phase1_features import FeatureBundle

__all__ = [
    "ConditionerContext",
    "EdgeGeometryFeatures",
    "PairSource",
    "PhysicsPairSource",
    "GeometryPairSource",
    "PairMessageMLP",
    "PairInteractionConditioner",
    "PairGeometryControlConditioner",
    "PairPhysicsConditioner",
    "PairPhysicsMomentsConditioner",
    "PairPhysicsUncertaintyConditioner",
    "PairPhysicsFutureConditioner",
    "edge_geometry_features",
    "EDGE_GEOMETRY_DIM",
    "matched_hidden_width",
]

#: Gaussian basis functions over the edge distance.
_DISTANCE_BASIS = 8

#: Width of :func:`edge_geometry_features` **excluding** the relation embedding,
#: which is added by the conditioner: distance basis + saturating distance (9),
#: ``R_i^T r_hat_ij`` (3), ``R_i^T R_j`` (9).
EDGE_GEOMETRY_DIM = _DISTANCE_BASIS + 1 + 3 + 9


@dataclass
class ConditionerContext:
    """Per-residue thermodynamic context, for arms that gate on it.

    Args:
        temperature_kelvin: ``[N_res]`` the graph temperature, broadcast to rows.
        lag_ps: ``[N_res]`` the transition lag, broadcast to rows.

    Raw quantities, not features: the conditioner featurises them itself with
    :mod:`force_md.conditioning.temperature` and :mod:`force_md.conditioning.lag`,
    so their encodings live in one place and this class cannot go stale against
    them. Only arms whose ``wants_context`` is True are handed one, which is why
    the five Phase 1.5 arms keep their exact forward signature.
    """

    temperature_kelvin: Tensor
    lag_ps: Tensor

    def features(self) -> Tensor:
        """``[N_res, TEMPERATURE_FEATURE_DIM + LAG_FEATURE_DIM]`` invariants."""
        return torch.cat(
            [
                temperature_features(self.temperature_kelvin),
                lag_features(self.lag_ps),
            ],
            dim=-1,
        )

    @staticmethod
    def feature_dim() -> int:
        return TEMPERATURE_FEATURE_DIM + LAG_FEATURE_DIM


@dataclass
class EdgeGeometryFeatures:
    """Invariant geometry of one directed residue pair, plus its validity."""

    features: Tensor        # [E, EDGE_GEOMETRY_DIM]
    valid: Tensor           # [E] bool -- both endpoints usable
    src: Tensor
    dst: Tensor
    edge_type: Tensor


def edge_geometry_features(
    bundle: FeatureBundle, *, cutoff: float
) -> EdgeGeometryFeatures:
    """Current-frame pair geometry, in the **receiving** residue's frame.

    Args:
        bundle: must carry pair messages; only its edge index and geometry are
            read here, never the message itself.
        cutoff: the distance the basis spans. Pass the graph's own cutoff so the
            basis does not stop short of the neighbour list.

    Returns:
        :class:`EdgeGeometryFeatures` with ``[E, EDGE_GEOMETRY_DIM]`` invariants:

        * a Gaussian basis over ``d_ij`` plus ``tanh(d_ij / cutoff)`` -- bounded,
          for the reason ``displacement_features`` documents: these features feed
          blocks with a body-order-3 term that squares its input;
        * ``R_i^T r_hat_ij``, the direction of the neighbour in ``i``'s frame;
        * ``R_i^T R_j``, how the neighbour is oriented relative to ``i``.

    **Sequence separation is not recomputed here.** It is already carried, exactly
    as Phase 1 carries it, by ``edge_type``: the backbone relation vocabulary is
    the sequence offset buckets ``-2, -1, +1, +2`` and "spatial". The conditioner
    embeds that id rather than re-deriving a separation from residue numbering
    that the edge builder has already bucketed.

    Only the current frame is read. There is no path from a future coordinate to
    this function: the bundle is built from the state at ``t`` alone.
    """
    pair = bundle.require_pair()
    rotation = bundle.frames.rotation
    dst_rotation = rotation[pair.dst]            # receiving residue's frame
    inverse = dst_rotation.transpose(-1, -2)

    distance = pair.distance
    centres = torch.linspace(
        0.0, float(cutoff), _DISTANCE_BASIS, dtype=distance.dtype, device=distance.device
    )
    width = float(cutoff) / max(_DISTANCE_BASIS - 1, 1)
    basis = torch.exp(-(((distance[:, None] - centres[None, :]) / width) ** 2))
    saturating = torch.tanh(distance / float(cutoff)).unsqueeze(-1)

    direction = torch.einsum("eij,ej->ei", inverse, pair.unit_vector)
    relative = torch.einsum("eij,ejk->eik", inverse, rotation[pair.src]).reshape(-1, 9)

    valid = bundle.residue_valid[pair.src] & bundle.residue_valid[pair.dst]
    return EdgeGeometryFeatures(
        features=torch.cat([basis, saturating, direction, relative], dim=-1),
        valid=valid,
        src=pair.src,
        dst=pair.dst,
        edge_type=pair.edge_type,
    )


def matched_hidden_width(
    *, physics_in: int, geometry_in: int, d_pair: int, minimum: int = 8
) -> int:
    """Hidden width that gives :class:`GeometryPairSource` the parameter count of
    :class:`PhysicsPairSource`.

    The physics source is ``LayerNorm(P) -> Linear(P, d)``, so it costs
    ``2P + Pd + d``. The geometry source is
    ``LayerNorm(G) -> Linear(G, h) -> SiLU -> Linear(h, d)``, costing
    ``2G + Gh + h + hd + d``. Solving the two for ``h``:

        h = (2P + Pd - 2G) / (G + 1 + d)

    ``P`` is 880 and ``G`` is small, so ``h`` comes out in the hundreds and the
    control ends up **wider** than the arm it controls, not narrower. That is the
    safe direction: if the geometry arm still loses, it did not lose for want of
    capacity.

    This exists so §7 of the plan -- "adjust the projection width or report the
    difference" -- is satisfied by construction and re-derived whenever a width
    changes, instead of being a number someone typed once.
    """
    numerator = 2 * physics_in + physics_in * d_pair - 2 * geometry_in
    denominator = geometry_in + 1 + d_pair
    return max(int(round(numerator / denominator)), minimum)


class PairSource(nn.Module):
    """Produces ``[E, d_pair]`` invariant features for each directed residue pair.

    The single point of difference between ``P0`` and ``P1``. Everything
    downstream -- the message MLP, the pooling, the adapter, the backbone -- is
    the same module with the same shape.
    """

    def __init__(self, d_pair: int):
        super().__init__()
        self.d_pair = int(d_pair)

    def forward(self, bundle: FeatureBundle, geometry: EdgeGeometryFeatures) -> Tensor:
        raise NotImplementedError


class PhysicsPairSource(PairSource):
    """``P1``: Phase 1's pair message, rotated into the receiving residue's frame.

    ``IrrepsLocalFrame`` has no parameters -- it is a change of basis -- so every
    trainable weight here is in the projection, which is what
    :func:`matched_hidden_width` matches against.
    """

    def __init__(self, message_irreps: str, d_pair: int):
        super().__init__(d_pair)
        self.projection = IrrepsLocalFrame(message_irreps)
        self.in_dim = self.projection.dim
        self.norm = nn.LayerNorm(self.in_dim)
        self.linear = nn.Linear(self.in_dim, self.d_pair)

    def forward(self, bundle: FeatureBundle, geometry: EdgeGeometryFeatures) -> Tensor:
        pair = bundle.require_pair()
        local = self.projection(pair.message, bundle.frames.rotation[pair.dst])
        return self.linear(self.norm(local))


class GeometryPairSource(PairSource):
    """``P0``: the same width from **current pair geometry only**.

    No physics latent, no force, no future anything -- the same edge geometry the
    physics arm also receives, put through a matched-capacity MLP. If the physics
    arm wins over this one, capacity is not the explanation.
    """

    def __init__(self, geometry_dim: int, d_pair: int, hidden: int):
        super().__init__(d_pair)
        self.in_dim = int(geometry_dim)
        self.norm = nn.LayerNorm(self.in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.d_pair),
        )

    def forward(self, bundle: FeatureBundle, geometry: EdgeGeometryFeatures) -> Tensor:
        return self.net(self.norm(geometry.features))


class PairMessageMLP(nn.Module):
    """Edge features -> edge message. Identical shape in every pair arm."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = int(out_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def pool_edges(
    messages: Tensor, dst: Tensor, num_residues: int
) -> Tensor:
    """Degree-stable pooling of edge messages onto their receiving residue.

    Returns ``[N_res, 2 * d_msg + 1]``:

    * ``sum / sqrt(deg)`` -- extensive, but with the ``sqrt`` normalisation Phase 1
      uses on its own aggregations, so a 24-neighbour residue does not arrive with
      five times the activation of a 1-neighbour one;
    * ``sum / deg`` -- intensive, "what are this residue's interactions like"
      independent of how many there are;
    * ``tanh(deg / 20)`` -- the coordination number itself, bounded, because the
      two normalised channels above have deliberately removed it and a buried
      residue differs from an exposed one.

    An isolated residue (no edges) gets exact zeros and a degree channel of zero,
    not a division by zero.
    """
    total = scatter_sum(messages, dst, num_residues)
    degree = scatter_sum(
        torch.ones_like(messages[:, :1]), dst, num_residues
    ).clamp(min=1.0)
    return torch.cat(
        [total / degree.sqrt(), total / degree, torch.tanh(degree / 20.0)], dim=-1
    )


class PairInteractionConditioner(TransitionConditioner):
    """Shared body of every pair arm; subclasses choose the source and extras.

    Args:
        config: the shared conditioner shape. ``d_cond`` is identical to every
            other arm's, so the transition backbone is unchanged.
        irreps: Phase 1's **node** latent irreps -- accepted for signature
            compatibility with :func:`build_conditioner` and used only by the
            moment arms.
        message_irreps: Phase 1's **pair message** irreps, from
            ``LocalPhysicsModel.pair_contract()``.
        cutoff: the residue graph's cutoff, for the distance basis.
    """

    arm = "pair_abstract"
    takes_latent_irreps = True
    requires_pair_features = True
    #: Subclass switch: read Phase 1's pair message, or only the geometry.
    uses_physics_pair: bool = True
    #: Subclass switch: add the predicted-force moments and residue shape.
    uses_moments: bool = False

    def __init__(
        self,
        config: ConditionerConfig,
        irreps: str,
        *,
        message_irreps: str,
        cutoff: float = 13.0,
    ):
        super().__init__(config)
        self.cutoff = float(cutoff)
        self.latent_irreps = str(irreps)
        self.message_irreps = str(message_irreps)

        self.relation_embedding = nn.Embedding(
            config.pair_relation_types, config.element_embedding_dim
        )
        geometry_dim = EDGE_GEOMETRY_DIM + config.element_embedding_dim

        if self.uses_physics_pair:
            self.source: PairSource = PhysicsPairSource(message_irreps, config.d_pair)
        else:
            hidden = matched_hidden_width(
                physics_in=IrrepsLocalFrame(message_irreps).dim,
                geometry_in=geometry_dim,
                d_pair=config.d_pair,
            )
            self.source = GeometryPairSource(geometry_dim, config.d_pair, hidden)

        self.message = PairMessageMLP(
            config.d_pair + geometry_dim, config.pair_hidden, config.pair_message_dim
        )
        pooled_dim = 2 * config.pair_message_dim + 1
        raw_dim = pooled_dim
        if self.uses_moments:
            raw_dim += MOMENT_FEATURE_DIM + ResidueShape.feature_dim()
        self.pooled_dim = pooled_dim
        self.adapter = _Adapter(raw_dim, config)

    # -- pieces ------------------------------------------------------------

    def _pooled_pair(self, b: FeatureBundle) -> Tensor:
        geometry = edge_geometry_features(b, cutoff=self.cutoff)
        edge_features = torch.cat(
            [geometry.features, self.relation_embedding(geometry.edge_type)], dim=-1
        )
        source = self.source(b, EdgeGeometryFeatures(
            features=edge_features, valid=geometry.valid, src=geometry.src,
            dst=geometry.dst, edge_type=geometry.edge_type,
        ))
        messages = self.message(torch.cat([source, edge_features], dim=-1))
        # An edge touching an unusable residue contributes nothing. Zeroing the
        # message rather than dropping the row keeps the pooling's degree count
        # honest about the graph the model actually saw.
        messages = messages * geometry.valid.unsqueeze(-1).to(messages.dtype)
        return pool_edges(messages, geometry.dst, b.num_residues)

    def _moments(self, b: FeatureBundle) -> Tensor:
        """Moments of Phase 1's **predicted** atom forces, reused from Phase 1.5.

        Identical call to the one ``force_pattern_shape`` makes, so ``P2`` differs
        from Phase 1.5's arm D in what it pairs the moments *with*, not in how the
        moments are computed.
        """
        rotation = b.frames.rotation[b.atom_to_residue]
        local_force = torch.einsum(
            "nij,nj->ni", rotation.transpose(-1, -2), b.atom_force_mean
        )
        y = b.atom_local_coordinates
        weights = precision_weights(b.atom_force_logvar, self.config, valid=b.atom_valid)
        moments = force_moments(
            y, local_force, b.atom_to_residue, b.num_residues, weights=weights
        )
        shape = residue_shape(y, b.atom_to_residue, b.num_residues)
        return torch.cat([moments.as_features(), shape.as_features()], dim=-1)

    def _assemble(self, b: FeatureBundle, pooled: Tensor) -> Tensor:
        parts = [pooled]
        if self.uses_moments:
            parts.append(self._moments(b))
        return self._mask(self.adapter(torch.cat(parts, dim=-1)), b)

    def forward(self, bundle) -> Tensor:
        b = self._production(bundle)
        return self._assemble(b, self._pooled_pair(b))


@register_conditioner
class PairGeometryControlConditioner(PairInteractionConditioner):
    """``P0_pair_geometry_control`` -- the capacity-matched control.

    Same adapter, same pooling, same ``d_cond``, same edge set, matched parameter
    count. Sees the current pair geometry and nothing else. No force, no physics
    latent, no future contact.
    """

    arm = "pair_geometry"
    uses_physics_pair = False


@register_conditioner
class PairPhysicsConditioner(PairInteractionConditioner):
    """``P1_pair_physics_frozen`` -- Phase 1's pair message, and geometry."""

    arm = "pair_physics"


@register_conditioner
class PairPhysicsMomentsConditioner(PairInteractionConditioner):
    """``P2_pair_physics_moments`` -- ``P1`` plus the residue force moments.

    ``F_i``, ``tau_i`` and the symmetric traceless ``S_i`` of the predicted atom
    forces, exactly as :func:`force_md.transition.moments.force_moments` defines
    them: ``S_i`` is handled as the five independent components of an ``l = 2``
    object in the residue frame, never as nine flattened global Cartesian numbers.
    """

    arm = "pair_physics_moments"
    uses_moments = True


@register_conditioner
class PairPhysicsUncertaintyConditioner(PairPhysicsMomentsConditioner):
    """``P3_pair_physics_uncertainty`` -- ``P2`` with an uncertainty gate.

    ``g_i = sigmoid(MLP([log sigma_i, T, lag]))`` multiplies the physics-derived
    block, so the probe can learn to *stop* leaning on Phase 1 where Phase 1 says
    it is unsure -- and, because temperature and lag are inputs to the gate, to do
    that differently at 450 K and at 4 ns than at 320 K and 1 ns.

    The gate reads Phase 1's **predicted** log-variance. There is no ground-truth
    uncertainty in mdCATH and none is invented.
    """

    arm = "pair_physics_uncertainty"
    wants_context = True

    #: mean atom-force logvar (3) + residue force logvar (3) + torque logvar (3)
    UNCERTAINTY_DIM = 9

    def __init__(self, config: ConditionerConfig, irreps: str, **kwargs):
        super().__init__(config, irreps, **kwargs)
        self.gate = nn.Sequential(
            nn.Linear(
                self.UNCERTAINTY_DIM + ConditionerContext.feature_dim(),
                config.gate_hidden,
            ),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, 1),
        )
        # Start at g = 0.5 rather than at an arbitrary point: the arm begins as a
        # uniformly attenuated P2 and has to learn where to open up, instead of
        # starting somewhere the initialisation happened to put it.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _uncertainty(self, b: FeatureBundle) -> Tensor:
        atom_logvar = scatter_sum(
            b.atom_force_logvar * b.atom_valid.unsqueeze(-1).to(b.atom_force_logvar.dtype),
            b.atom_to_residue,
            b.num_residues,
        )
        counts = scatter_sum(
            b.atom_valid.unsqueeze(-1).to(atom_logvar.dtype),
            b.atom_to_residue,
            b.num_residues,
        ).clamp(min=1.0)
        return torch.cat(
            [atom_logvar / counts, b.residue_force_logvar, b.residue_torque_logvar],
            dim=-1,
        )

    def forward(self, bundle, *, context: Optional[ConditionerContext] = None) -> Tensor:
        # Production check first: a ground-truth label reaching a production arm
        # is the more serious of the two errors, and checking the context first
        # would report a missing keyword argument for what is actually a leak.
        b = self._production(bundle)
        if context is None:
            raise ValueError(
                f"arm {self.arm!r} gates on temperature and lag and was called "
                "without a ConditionerContext. Without it the gate would be a "
                "function of uncertainty alone, which is a different arm."
            )
        gate = torch.sigmoid(
            self.gate(torch.cat([self._uncertainty(b), context.features()], dim=-1))
        )
        pooled = self._pooled_pair(b) * gate
        parts = [pooled, self._moments(b) * gate]
        return self._mask(self.adapter(torch.cat(parts, dim=-1)), b)


@register_conditioner
class PairPhysicsFutureConditioner(PairPhysicsMomentsConditioner):
    """``P4_future_physics_consistency`` -- ``P2``'s conditioner, unchanged.

    P4 differs from P2 **only** in an auxiliary loss, which lives on the probe
    (:class:`force_md.transition.future_physics.FuturePhysicsHead`) and in the
    trainer. Giving it its own arm name keeps the two apart in every provenance
    record and results row; giving it the same conditioner keeps the comparison
    honest, because a conditioner difference would confound the auxiliary loss
    with a representation change.
    """

    arm = "pair_physics_future"
