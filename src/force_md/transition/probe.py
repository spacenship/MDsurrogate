"""The Phase 1.5 transition probe: a small deterministic residue-frame model.

This is **not** a stochastic flow model and must not be described as one. It
predicts a single rigid update per residue::

    (q_{t-1}, q_t, sequence, temperature, lag)  ->  (delta_r_local_i, R_rel_i)

That is deliberately less than Phase 2 needs. The question this probe exists to
answer is narrow -- does Phase 1's force-supervised representation help predict a
1-4 ns future structure, against a structure-only control of the same size? -- and
a full flow model would answer it more slowly and less clearly.

**One model, five arms.** Every arm shares this class, this backbone, this head
and this configuration; ``config.arm`` selects which conditioner produces the
``d_cond`` block. There is no per-arm subclass, because two copies of a model
diverge and the divergence is then reported as a result.

**Equivariance comes from the architecture, not from a loss term.** The backbone
is Phase 1's own :class:`~force_md.nn.blocks.BackboneInteractionBlock` over
global-frame irreps, so its features rotate with the protein. Both outputs are
then expressed in the current residue's frame::

    delta_r_local = R_cur^T v          v from a 1x1o head
    R_rel         = R_cur^T R_pred

and both are invariant under a global rigid motion, matching the targets built in
:mod:`force_md.transition.targets` exactly. No input is ever Kabsch-aligned.

**The model starts at the identity baseline.** Both heads are zero-initialised and
the rotation head predicts a *correction* to the current frame's own axes, so at
step 0 the probe predicts "nothing moves" exactly. Since that baseline is strong
at these lags (measured 1 ns Ca RMSD ~1.1-2.9 A at 320 K), starting there makes
"has it learned anything" unambiguous rather than something to be inferred from a
loss curve.

**Rotation is emitted through the 6D chart**, not axis-angle: it is continuous and
surjective onto SO(3), and Checkpoint 2 measured residue frame rotations with a
p95 of 125 degrees at 4 ns, which is close enough to the axis-angle discontinuity
at 180 degrees to matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from e3nn import o3
from torch import Tensor, nn

from ..conditioning.residue import ResidueConditioner
from ..conditioning.temperature import TEMPERATURE_FEATURE_DIM, temperature_features
from ..data.contracts import FrameGeometry, HierarchicalProteinBatch
from ..geometry.frames import ResidueFrames, build_residue_frames, link_backbone_to_atom_positions
from ..geometry.so3 import relative_rotation, rotation_from_6d
from ..graph.edges import (
    build_knn_edges,
    build_sequence_edges,
    edge_geometry,
    edge_spherical_harmonics,
    merge_edge_sets,
)
from ..nn.blocks import BackboneInteractionBlock
from ..nn.irreps import IrrepsConfig
from .conditioners import ConditionerConfig, TransitionConditioner, build_conditioner
from .phase1_features import FeatureBundle, OracleFeatureBundle
from .targets import TransitionPrediction

__all__ = [
    "TransitionProbeConfig",
    "TransitionProbe",
    "lag_features",
    "LAG_FEATURE_DIM",
    "displacement_features",
    "DISPLACEMENT_FEATURE_DIM",
    "HISTORY_FEATURE_DIM",
]

#: Lag encoding width: Gaussian basis over the lag range, plus log-lag.
LAG_BASIS = 8
LAG_FEATURE_DIM = LAG_BASIS + 1

#: Range the lag basis spans, in picoseconds. Covers the Phase 1.5 lags (1 and
#: 4 ns) with room for the 2/8/16 ns Phase 2 wants, so the encoding does not have
#: to change when they are added.
_LAG_MIN_PS = 1000.0
_LAG_MAX_PS = 16000.0


def lag_features(lag_ps: Tensor, num_basis: int = LAG_BASIS) -> Tensor:
    """Featurise the physical lag as invariant scalars, ``[B, LAG_FEATURE_DIM]``.

    Continuous, not a two-way one-hot over {1 ns, 4 ns}, for the same reason
    temperature is continuous in :mod:`force_md.conditioning.temperature`: the
    probe should place 4 ns relative to 1 ns rather than memorise two buckets, and
    Phase 2's 2/8/16 ns lags then need no new representation.

    The basis is spread over ``log`` lag because the lags are geometric (1, 2, 4,
    8, 16 ns); linear centres would put six of eight basis functions above 8 ns.
    """
    lag = lag_ps.reshape(-1).to(torch.get_default_dtype())
    log_lag = torch.log(lag.clamp(min=1.0))
    lo, hi = torch.log(torch.tensor(_LAG_MIN_PS)), torch.log(torch.tensor(_LAG_MAX_PS))
    centres = torch.linspace(float(lo), float(hi), num_basis, dtype=lag.dtype, device=lag.device)
    width = (float(hi) - float(lo)) / max(num_basis - 1, 1)
    basis = torch.exp(-(((log_lag[:, None] - centres[None, :]) / width) ** 2))
    normalised = ((log_lag - float(lo)) / (float(hi) - float(lo))).unsqueeze(-1)
    return torch.cat([basis, normalised], dim=-1)


#: History displacement encoding: Gaussian basis over |dr|, plus a saturating scale.
DISPLACEMENT_BASIS = 8
DISPLACEMENT_FEATURE_DIM = DISPLACEMENT_BASIS + 1

#: Per history frame: 6D relative rotation + the displacement magnitude encoding.
HISTORY_FEATURE_DIM = 6 + DISPLACEMENT_FEATURE_DIM

#: Range the displacement basis spans, in Angstrom. The measured mean CA
#: displacement over a 1 ns step runs 1.36 A at 320 K to 5.61 A at 450 K, with a
#: per-residue maximum of 44 A over a 630-pair sample; 20 A covers the bulk and
#: the saturating channel carries whatever lies beyond it.
_DISPLACEMENT_MAX_ANGSTROM = 20.0

#: Below this the displacement direction is not defined; the seed vector fades
#: linearly to zero instead of becoming arbitrary.
_DISPLACEMENT_EPS = 1e-3


def displacement_features(
    magnitude: Tensor, num_basis: int = DISPLACEMENT_BASIS
) -> Tensor:
    """Featurise a displacement magnitude as **bounded** invariant scalars.

    Args:
        magnitude: ``[N, 1]`` displacement magnitudes in Angstrom.

    Returns:
        ``[N, DISPLACEMENT_FEATURE_DIM]``: a Gaussian basis over ``[0, 20] A`` and
        ``tanh(|dr| / 20)``. Every channel is in ``[0, 1]``.

    Why this is not simply ``|dr|``. Each :class:`BackboneInteractionBlock`
    contains a body-order-3 term ``h <- h + square_mix(h (x) h)``, so feature
    magnitude is squared once per block and ``num_blocks`` blocks compound it to
    roughly ``|h|**(2**num_blocks)``. Feeding a raw Angstrom magnitude -- 1.4 A at
    320 K, up to 44 A at 450 K -- makes that a divergence rather than a non-
    linearity. Measured over 60 steps at ``num_blocks=3``: raw magnitude gives a
    median gradient norm of 1.7e6 and a peak of 1.4e11; removing the raw seed
    entirely gives 0.01 and 0.74. Phase 1 never hits this because it interleaves
    its two backbone blocks with pooling and injection layers rather than stacking
    them.
    """
    m = magnitude.reshape(-1).to(torch.get_default_dtype())
    centres = torch.linspace(
        0.0, _DISPLACEMENT_MAX_ANGSTROM, num_basis, dtype=m.dtype, device=m.device
    )
    width = _DISPLACEMENT_MAX_ANGSTROM / max(num_basis - 1, 1)
    basis = torch.exp(-(((m[:, None] - centres[None, :]) / width) ** 2))
    saturating = torch.tanh(m / _DISPLACEMENT_MAX_ANGSTROM).unsqueeze(-1)
    return torch.cat([basis, saturating], dim=-1)


@dataclass(frozen=True)
class TransitionProbeConfig:
    """Probe hyper-parameters. Identical across arms except for ``arm``.

    Args:
        arm: which conditioner to use -- a key of
            :data:`~force_md.transition.conditioners.CONDITIONER_ARMS`.
        conditioner: shared conditioner shape, including ``d_cond``.
        irreps: node feature width. Smaller than Phase 1's: this is a probe, and
            a wide transition model would confound capacity with representation.
        num_blocks: residue-level interaction blocks. The plan's 2-4.
        history_length: frames of state the model sees, counting the current one.
            2 is the ablation default; 1 is a diagnostic.
        residue_knn / sequence_max_offset / backbone_cutoff: the **same** edge
            semantics Phase 1's backbone level uses, reused rather than redefined.
        use_plm / use_temperature: conditioning ablation hooks, shared by all arms.
        predict_translation / predict_rotation: for diagnostics; both on by default.
    """

    arm: str = "structure_only"
    conditioner: ConditionerConfig = field(default_factory=ConditionerConfig)
    irreps: IrrepsConfig = field(
        default_factory=lambda: IrrepsConfig(
            scalar_channels=32, vector_channels=8, tensor_channels=4
        )
    )
    num_blocks: int = 3
    num_radial_basis: int = 8
    history_length: int = 2
    residue_knn: int = 16
    sequence_max_offset: int = 2
    backbone_cutoff: float = 13.0
    backbone_avg_neighbors: float = 20.0
    residue_scalar_channels: int = 64
    plm_dim: int = 1280
    plm_projection_dim: int = 128
    use_plm: bool = True
    use_temperature: bool = True
    use_body_order_3: bool = True

    @property
    def past_frames(self) -> int:
        return self.history_length - 1


class TransitionProbe(nn.Module):
    """Predicts one rigid update per residue. Same class for every arm.

    Args:
        config: see :class:`TransitionProbeConfig`.
        latent_irreps: Phase 1's ``physics_latent_irreps``, read from the frozen
            checkpoint's contract. Never hard-coded: Phase 1.5 must not assume 152.
    """

    def __init__(self, config: TransitionProbeConfig, *, latent_irreps: str):
        super().__init__()
        self.config = config
        self.irreps = config.irreps.node_irreps()
        self.irreps_sh = config.irreps.sh_irreps()

        self.conditioner: TransitionConditioner = build_conditioner(
            config.arm, config.conditioner, irreps=latent_irreps
        )
        self.residue_conditioner = ResidueConditioner(
            plm_dim=config.plm_dim,
            plm_projection_dim=config.plm_projection_dim,
            out_channels=config.residue_scalar_channels,
            use_plm=config.use_plm,
            use_temperature=config.use_temperature,
        )

        # invariant scalars entering the node features
        scalar_dim = (
            config.residue_scalar_channels
            + config.conditioner.d_cond
            + LAG_FEATURE_DIM
            + HISTORY_FEATURE_DIM * config.past_frames
        )
        # equivariant seeds: one global displacement vector per history frame
        seed_irreps = o3.Irreps(f"{scalar_dim}x0e")
        if config.past_frames:
            seed_irreps = seed_irreps + o3.Irreps(f"{config.past_frames}x1o")
        self.node_init = o3.Linear(seed_irreps, self.irreps)
        self.seed_irreps = seed_irreps

        self.blocks = nn.ModuleList(
            BackboneInteractionBlock(
                self.irreps,
                self.irreps_sh,
                # sequence offsets (-2,-1,+1,+2 -> 5 buckets) + spatial
                num_relation_types=2 * config.sequence_max_offset + 2,
                r_cut=config.backbone_cutoff,
                avg_num_neighbors=config.backbone_avg_neighbors,
                num_radial_basis=config.num_radial_basis,
                use_body_order_3=config.use_body_order_3,
            )
            for _ in range(config.num_blocks)
        )

        # Heads are zero-initialised so the probe starts at the identity baseline.
        self.translation_head = o3.Linear(self.irreps, o3.Irreps("1x1o"))
        self.rotation_head = o3.Linear(self.irreps, o3.Irreps("2x1o"))
        for head in (self.translation_head, self.rotation_head):
            for parameter in head.parameters():
                nn.init.zeros_(parameter)

    # -- parameter accounting ---------------------------------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-part counts, so an arm's total is explainable rather than quoted."""
        return {
            "conditioner": self.conditioner.parameter_count(),
            "residue_conditioner": sum(
                p.numel() for p in self.residue_conditioner.parameters()
            ),
            "node_init": sum(p.numel() for p in self.node_init.parameters()),
            "blocks": sum(p.numel() for p in self.blocks.parameters()),
            "heads": sum(
                p.numel()
                for head in (self.translation_head, self.rotation_head)
                for p in head.parameters()
            ),
            "total": self.parameter_count(),
        }

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        batch: HierarchicalProteinBatch,
        bundle: FeatureBundle | OracleFeatureBundle,
        *,
        history: Sequence[FrameGeometry] = (),
        lag_ps: Optional[Tensor] = None,
    ) -> TransitionPrediction:
        """Predict the transition.

        Args:
            batch: the state at ``t``. Its ``atoms.forces`` are not read.
            bundle: frozen Phase 1 features for the same state. An
                :class:`OracleFeatureBundle` only for the oracle arm; the
                conditioners enforce that.
            history: frames before ``t``, oldest first. Must be
                ``config.past_frames`` long.
            lag_ps: ``[B]`` physical lag. Required: a probe that did not condition
                on the lag would be asked to predict 1 ns and 4 ns with one answer.

        Returns:
            :class:`TransitionPrediction` in the current local frame.
        """
        if len(history) != self.config.past_frames:
            raise ValueError(
                f"history_length={self.config.history_length} needs "
                f"{self.config.past_frames} past frame(s), got {len(history)}"
            )
        if lag_ps is None:
            raise ValueError(
                "lag_ps is required: without it the probe would have to give the "
                "same answer for 1 ns and 4 ns"
            )

        linked = link_backbone_to_atom_positions(batch)
        frames = build_residue_frames(
            linked.backbone.n_positions,
            linked.backbone.ca_positions,
            linked.backbone.c_positions,
            prior_valid=linked.backbone.frame_valid,
        )
        scalars, vectors = self._inputs(linked, bundle, history, lag_ps, frames)

        seed = scalars if not vectors else torch.cat([scalars, *vectors], dim=-1)
        node = self.node_init(seed)

        edges, edge_sh, distance = self._graph(linked)
        for block in self.blocks:
            node = block(node, edges, edge_sh, distance)

        return self._heads(node, frames)

    def _inputs(
        self,
        batch: HierarchicalProteinBatch,
        bundle,
        history: Sequence[FrameGeometry],
        lag_ps: Tensor,
        frames: ResidueFrames,
    ) -> tuple[Tensor, list[Tensor]]:
        """Invariant scalars and equivariant seed vectors for every residue."""
        parts = [self.residue_conditioner(batch), self.conditioner(bundle)]

        lag = lag_features(lag_ps.to(batch.backbone.ca_positions.dtype))
        parts.append(lag[batch.residues.batch_index])

        vectors: list[Tensor] = []
        rotation = frames.rotation
        for past in history:
            # Displacement of this residue's CA over the history step. The
            # *direction* seeds the l=1 channels -- equivariant and of unit norm,
            # so the body-order-3 term in each block cannot compound it -- while
            # the magnitude enters as bounded invariant scalars below. Feeding the
            # raw Angstrom vector here diverges at num_blocks >= 3; see
            # :func:`displacement_features`.
            displacement = batch.backbone.ca_positions - past.ca_positions
            magnitude = displacement.norm(dim=-1, keepdim=True)
            vectors.append(displacement / magnitude.clamp_min(_DISPLACEMENT_EPS))
            past_frames = build_residue_frames(
                past.n_positions, past.ca_positions, past.c_positions,
                prior_valid=past.frame_valid,
            )
            # How the frame turned over the history step, in the current frame:
            # invariant, and the 6D chart avoids the axis-angle wrap.
            turn = relative_rotation(rotation, past_frames.rotation)
            parts.append(
                torch.cat(
                    [
                        turn[:, :, :2].transpose(-1, -2).reshape(-1, 6),
                        displacement_features(magnitude),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(parts, dim=-1), vectors

    def _graph(self, batch: HierarchicalProteinBatch):
        """Sequence and CA-kNN edges -- the same semantics as Phase 1's backbone level.

        Rebuilt from the current coordinates on every call, because the neighbour
        list is a discrete function of them. Gradients flow through the edge
        vectors and distances, never through the topology.
        """
        residues = batch.residues
        sequence = build_sequence_edges(
            residues.batch_index,
            residues.chain_index,
            residues.resid_original,
            max_offset=self.config.sequence_max_offset,
        )
        spatial = build_knn_edges(
            batch.backbone.ca_positions,
            residues.batch_index,
            k=self.config.residue_knn,
            cutoff=self.config.backbone_cutoff,
        )
        edges = merge_edge_sets([sequence, spatial], "residue__message__residue")
        geometry = edge_geometry(
            batch.backbone.ca_positions, batch.backbone.ca_positions, edges
        )
        sh = edge_spherical_harmonics(geometry.unit_vector, self.config.irreps.lmax)
        return edges, sh, geometry.distance

    def _heads(self, node: Tensor, frames: ResidueFrames) -> TransitionPrediction:
        """Global-frame outputs, expressed in each residue's own frame.

        The rotation head predicts a **correction to the current frame's own two
        leading axes** rather than a rotation from scratch. Zero-initialised, that
        reproduces ``R_cur`` exactly, so ``R_rel`` starts at the identity; and it
        stays equivariant, because the axes it corrects rotate with the protein.
        """
        rotation = frames.rotation
        inverse = rotation.transpose(-1, -2)

        translation_global = self.translation_head(node)
        translation_local = torch.einsum("nij,nj->ni", inverse, translation_global)

        delta = self.rotation_head(node).reshape(-1, 2, 3)
        axes = rotation[:, :, :2].transpose(-1, -2) + delta
        predicted_global = rotation_from_6d(axes.reshape(-1, 6))
        relative = relative_rotation(rotation, predicted_global)

        valid = frames.valid
        zero = torch.zeros((), dtype=translation_local.dtype, device=translation_local.device)
        eye = torch.eye(3, dtype=relative.dtype, device=relative.device).expand_as(relative)
        return TransitionPrediction(
            translation_local=torch.where(valid.unsqueeze(-1), translation_local, zero),
            rotation=torch.where(valid[:, None, None], relative, eye),
        )
