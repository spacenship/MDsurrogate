"""The A-E conditioners: five ways to summarise physics for one residue.

Every arm of the Phase 1.5 ablation uses the **same** transition backbone, the
same targets, the same manifest and the same budget. The only thing that changes
is which of these produces the ``[N_res, d_cond]`` conditioning vector:

======  =====================  =====================================================
arm     conditioner            what it may look at
======  =====================  =====================================================
A       ``ZeroConditioner``     nothing -- structure-only control
B       ``ResidueForceTorque``  Phase 1's predicted residue net force and torque
C       ``PhysicsLatent``       Phase 1's 152-d residue latent
D       ``ForcePatternShape``   latent + predicted atom-force pattern + shape
E       ``OracleAtomicForce``   ground-truth atom forces -- diagnostic only
======  =====================  =====================================================

**Every conditioner emits invariant scalars.** Global-frame quantities are rotated
into the residue's own frame first (:mod:`force_md.transition.local_frame` for the
irreps latent, ``R^T v`` for plain vectors), never flattened as they are. A
conditioner whose output changed when the protein was rotated would make the whole
probe non-equivariant through the back door.

**Fairness is enforced by construction.** All five share one adapter shape
(``LayerNorm -> Linear -> SiLU -> Linear``) with the same hidden width and the
same ``d_cond``, so the arms differ in what they *see*, not in how much machinery
they get to see it with. The raw input widths differ -- that is the point of the
experiment -- so trainable parameter counts differ slightly and
:meth:`TransitionConditioner.parameter_count` is recorded per arm in the results
table rather than being hidden.

**C is not new information.** ``z_phys = g(q_t, sequence, temperature)`` is a
function of inputs arm A already has. If C beats A that is evidence about
*force-supervised pretraining as an inductive bias*, not about extra observations.
The module says so here because the sentence belongs next to the code.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..data.residue_constants import NUM_ATOM_NAMES
from ..nn.irreps import scatter_sum
from .local_frame import IrrepsLocalFrame
from .moments import (
    MOMENT_FEATURE_DIM,
    ResidueShape,
    force_moments,
    residue_shape,
)
from .phase1_features import FeatureBundle, OracleFeatureBundle, assert_production_safe

__all__ = [
    "ConditionerConfig",
    "TransitionConditioner",
    "ZeroConditioner",
    "ResidueForceTorqueConditioner",
    "PhysicsLatentConditioner",
    "ForcePatternShapeConditioner",
    "OracleAtomicForceConditioner",
    "CONDITIONER_ARMS",
    "build_conditioner",
]

#: Elements mdCATH contains, plus headroom. Matches the encoder's ``_MAX_Z``.
_MAX_Z = 20


@dataclass(frozen=True)
class ConditionerConfig:
    """Shared shape of every arm's adapter, so the comparison stays fair.

    Args:
        d_cond: output width. Identical for every arm, including the zero one, so
            the transition backbone is byte-identical across the ablation.
        hidden: adapter hidden width.
        atom_message_dim: per-atom message width inside the learned set pooling.
        element_embedding_dim / name_embedding_dim: chemistry embeddings, used in
            place of an invented radius table.
        uncertainty_gating: weight each atom's force contribution by its predicted
            precision. See :func:`precision_weights`.
        logvar_reference: the log-variance at which the gate is 1.0. Phase 1 clamps
            log-variance to ``[-8, 8]``; the reference sits at 0 so a residue with
            typical uncertainty is unweighted and the gate spans a useful range in
            both directions.
        detach_conditioner: keep the conditioner's gradient from reaching Phase 1.
            Phase 1 is frozen anyway, so this is belt and braces for a later
            joint fine-tune.
    """

    d_cond: int = 64
    hidden: int = 128
    atom_message_dim: int = 64
    element_embedding_dim: int = 16
    name_embedding_dim: int = 16
    uncertainty_gating: bool = True
    logvar_reference: float = 0.0
    max_precision_weight: float = 4.0


def precision_weights(
    logvar: Tensor, config: ConditionerConfig, *, valid: Optional[Tensor] = None
) -> Tensor:
    """``[N_atom]`` weight in ``(0, max]`` falling as predicted uncertainty rises.

    ``w = exp(-(mean logvar - reference) / 2)``, i.e. one over the predicted
    standard deviation. A force Phase 1 is unsure about should push the
    conditioner around less than one it is confident in, and this is the
    monotone, parameter-free way to say so. The cap keeps a confidently-predicted
    atom from dominating a residue outright.
    """
    if not config.uncertainty_gating:
        weights = torch.ones_like(logvar[:, 0])
    else:
        scaled = (logvar.mean(dim=-1) - config.logvar_reference) * -0.5
        weights = torch.exp(scaled.clamp(max=float(torch.log(
            torch.tensor(config.max_precision_weight)
        ))))
    if valid is not None:
        weights = weights * valid.to(weights.dtype)
    return weights


class _Adapter(nn.Module):
    """The one adapter shape every arm gets, so arms differ only in their input."""

    def __init__(self, in_dim: int, config: ConditionerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.d_cond),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class TransitionConditioner(nn.Module, abc.ABC):
    """Maps a frozen-Phase-1 feature bundle to ``[N_res, d_cond]`` invariants."""

    #: True only for the diagnostic oracle arm.
    requires_oracle: bool = False
    #: Short name used in configs, run records and the results table.
    arm: str = "abstract"

    def __init__(self, config: ConditionerConfig):
        super().__init__()
        self.config = config

    @property
    def d_cond(self) -> int:
        return self.config.d_cond

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abc.abstractmethod
    def forward(self, bundle) -> Tensor:
        ...

    @staticmethod
    def _production(bundle) -> FeatureBundle:
        return assert_production_safe(bundle)

    def _mask(self, features: Tensor, bundle: FeatureBundle) -> Tensor:
        """Zero the rows of residues that are not usable."""
        return features * bundle.residue_valid.unsqueeze(-1).to(features.dtype)


class ZeroConditioner(TransitionConditioner):
    """Arm A: a fixed-size zero condition -- the structure-only control.

    Not "no conditioning": the backbone still receives a ``d_cond`` block, so arms
    A-E are the same network fed different content. Making A a narrower model
    would confound representation with capacity, which is the confound this whole
    experiment exists to avoid.
    """

    arm = "structure_only"

    def forward(self, bundle) -> Tensor:
        production = self._production(bundle)
        return torch.zeros(
            production.num_residues,
            self.d_cond,
            dtype=production.physics_latent.dtype,
            device=production.physics_latent.device,
        )


class ResidueForceTorqueConditioner(TransitionConditioner):
    """Arm B: Phase 1's predicted residue net force and torque, in the local frame.

    The arm that tests the plan's stated suspicion directly: a residue's net force
    is the right observable for translation but a poor conformational descriptor,
    because the internal forces cancel in the sum. If B fails while C or D
    succeeds, that suspicion is what the result is about.
    """

    arm = "force_torque"
    #: F(3) + tau(3) + |F| + |tau| + force logvar(3) + torque logvar(3)
    RAW_DIM = 3 + 3 + 1 + 1 + 3 + 3

    def __init__(self, config: ConditionerConfig):
        super().__init__(config)
        self.adapter = _Adapter(self.RAW_DIM, config)

    def forward(self, bundle) -> Tensor:
        b = self._production(bundle)
        inverse = b.frames.rotation.transpose(-1, -2)
        force = torch.einsum("nij,nj->ni", inverse, b.residue_force_mean)
        torque = torch.einsum("nij,nj->ni", inverse, b.residue_torque_mean)
        raw = torch.cat(
            [
                force,
                torque,
                force.norm(dim=-1, keepdim=True),
                torque.norm(dim=-1, keepdim=True),
                b.residue_force_logvar,
                b.residue_torque_logvar,
            ],
            dim=-1,
        )
        return self._mask(self.adapter(raw), b)


class PhysicsLatentConditioner(TransitionConditioner):
    """Arm C: Phase 1's 152-d residue latent, rotated into the residue frame.

    The latent is not flattened as it stands: its ``1o`` and ``2e`` channels are
    global-frame tensors and would make the conditioner rotation-dependent. See
    :mod:`force_md.transition.local_frame`.
    """

    arm = "physics_latent"

    def __init__(self, config: ConditionerConfig, irreps: str):
        super().__init__(config)
        self.projection = IrrepsLocalFrame(irreps)
        self.adapter = _Adapter(self.projection.dim, config)

    def forward(self, bundle) -> Tensor:
        b = self._production(bundle)
        local = self.projection(b.physics_latent, b.frames.rotation)
        return self._mask(self.adapter(local), b)


class _AtomSetPooling(nn.Module):
    """Permutation-invariant pooling of a residue's atoms.

    A per-atom **nonlinear** message is formed first and only then summed, which
    is the whole point: summing raw forces first and passing the total to an MLP
    would throw away the internal pattern before the network ever saw it, and
    would make this module an expensive way to recompute arm B.

    Sum and mean are both kept. Sum is extensive and carries "how much force is in
    this residue"; mean is intensive and carries "what is it like", which stops a
    tryptophan from outranking a glycine for having more atoms.
    """

    def __init__(self, in_dim: int, config: ConditionerConfig):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(in_dim, config.atom_message_dim),
            nn.SiLU(),
            nn.Linear(config.atom_message_dim, config.atom_message_dim),
        )
        self.out_dim = 2 * config.atom_message_dim

    def forward(
        self,
        atom_features: Tensor,
        atom_to_residue: Tensor,
        num_residues: int,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        messages = self.message(atom_features)
        if weights is not None:
            messages = messages * weights.unsqueeze(-1)
        total = scatter_sum(messages, atom_to_residue, num_residues)
        counts = scatter_sum(
            torch.ones_like(messages[:, :1]), atom_to_residue, num_residues
        ).clamp(min=1.0)
        return torch.cat([total, total / counts], dim=-1)


class ForcePatternShapeConditioner(TransitionConditioner):
    """Arm D: latent + the atom-force pattern + explicit moments + residue shape.

    Three sources, all invariant, all in the residue's local frame:

    1. the Phase 1 latent, as in arm C;
    2. **explicit** moments of the predicted atom forces -- net force, torque,
       isotropic compression, symmetric traceless stress -- so the quantities that
       survive Newton's third law are handed over rather than hoped for;
    3. a **learned** permutation-invariant pooling over the residue's atoms, which
       can represent patterns no fixed moment captures.

    Ground-truth forces never enter: ``f_ia`` here is Phase 1's predicted mean.
    """

    arm = "force_pattern_shape"

    def __init__(self, config: ConditionerConfig, irreps: str):
        super().__init__(config)
        self.projection = IrrepsLocalFrame(irreps)
        self.element_embedding = nn.Embedding(_MAX_Z + 1, config.element_embedding_dim)
        self.name_embedding = nn.Embedding(NUM_ATOM_NAMES, config.name_embedding_dim)
        atom_dim = (
            config.element_embedding_dim
            + config.name_embedding_dim
            + 3  # y_ia
            + 1  # |y_ia|
            + 3  # f_local
            + 1  # |f_local|
            + 3  # logvar
            + 2  # backbone / heavy flags
        )
        self.pooling = _AtomSetPooling(atom_dim, config)
        raw_dim = (
            self.projection.dim
            + MOMENT_FEATURE_DIM
            + ResidueShape.feature_dim()
            + self.pooling.out_dim
        )
        self.adapter = _Adapter(raw_dim, config)

    def _atom_forces(self, bundle: FeatureBundle) -> Tensor:
        """Predicted atomic force. Overridden by the oracle arm."""
        return bundle.atom_force_mean

    def forward(self, bundle) -> Tensor:
        b = bundle.production if isinstance(bundle, OracleFeatureBundle) else bundle
        if not self.requires_oracle:
            b = self._production(bundle)

        rotation = b.frames.rotation[b.atom_to_residue]
        local_force = torch.einsum(
            "nij,nj->ni", rotation.transpose(-1, -2), self._atom_forces(bundle)
        )
        y = b.atom_local_coordinates
        weights = precision_weights(b.atom_force_logvar, self.config, valid=b.atom_valid)

        moments = force_moments(
            y, local_force, b.atom_to_residue, b.num_residues, weights=weights
        )
        shape = residue_shape(y, b.atom_to_residue, b.num_residues)

        atom_features = torch.cat(
            [
                self.element_embedding(b.atomic_number.clamp(max=_MAX_Z)),
                self.name_embedding(b.atom_name_id),
                y,
                y.norm(dim=-1, keepdim=True),
                local_force,
                local_force.norm(dim=-1, keepdim=True),
                b.atom_force_logvar,
                b.atom_is_backbone.unsqueeze(-1).to(y.dtype),
                b.atom_is_heavy.unsqueeze(-1).to(y.dtype),
            ],
            dim=-1,
        )
        pooled = self.pooling(
            atom_features, b.atom_to_residue, b.num_residues, weights=weights
        )
        raw = torch.cat(
            [
                self.projection(b.physics_latent, b.frames.rotation),
                moments.as_features(),
                shape.as_features(),
                pooled,
            ],
            dim=-1,
        )
        return self._mask(self.adapter(raw), b)


class OracleAtomicForceConditioner(ForcePatternShapeConditioner):
    """Arm E: arm D fed **ground-truth** atom forces. Diagnostic only.

    Isolates one question from the other: if E does not beat arm A either, then
    instantaneous force is of little use for a 1-4 ns transition and no amount of
    improving Phase 1's force prediction would change that. If E beats A while D
    does not, the bottleneck is Phase 1's prediction or its representation, not
    the physics.

    It reads a label, so it is never a production arm and never a deployable
    model. It takes an :class:`OracleFeatureBundle` and refuses a production one,
    which is the mirror image of the guard on the other four.
    """

    arm = "oracle_force"
    requires_oracle = True

    def _atom_forces(self, bundle) -> Tensor:
        if not isinstance(bundle, OracleFeatureBundle):
            raise TypeError(
                "the oracle arm needs an OracleFeatureBundle carrying the force "
                "labels; it was given a production bundle. Silently falling back "
                "to the predicted forces would make arm E a copy of arm D."
            )
        return bundle.atom_force


#: Arm name -> class. Configs and the ablation runner name arms through this, so
#: no arm is ever a copy-pasted variant of the model.
CONDITIONER_ARMS: dict[str, type[TransitionConditioner]] = {
    ZeroConditioner.arm: ZeroConditioner,
    ResidueForceTorqueConditioner.arm: ResidueForceTorqueConditioner,
    PhysicsLatentConditioner.arm: PhysicsLatentConditioner,
    ForcePatternShapeConditioner.arm: ForcePatternShapeConditioner,
    OracleAtomicForceConditioner.arm: OracleAtomicForceConditioner,
}


def build_conditioner(
    arm: str, config: ConditionerConfig, *, irreps: str
) -> TransitionConditioner:
    """Construct one arm by name.

    Args:
        arm: a key of :data:`CONDITIONER_ARMS`.
        irreps: Phase 1's ``physics_latent_irreps``, read from the checkpoint
            contract rather than hard-coded -- Phase 1.5 must not assume 152.
    """
    if arm not in CONDITIONER_ARMS:
        raise ValueError(
            f"unknown arm {arm!r}; available: {sorted(CONDITIONER_ARMS)}"
        )
    cls = CONDITIONER_ARMS[arm]
    if cls in (PhysicsLatentConditioner, ForcePatternShapeConditioner,
               OracleAtomicForceConditioner):
        return cls(config, irreps=irreps)
    return cls(config)
