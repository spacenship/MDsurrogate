"""The Phase 1.5 transition target: where each residue frame goes.

    (q_{t-1}, q_t)  ->  q_{t+lag}

The target is **not** the raw future coordinates. mdCATH's protein diffuses and
tumbles in its box, and over 1-4 ns that global rigid motion dominates every
displacement: measured on real pairs, mean unaligned ``|CA(t+4ns) - CA(t)|``
reaches ~18 A against ~1-4 A of actual conformational change. A model scored on
raw coordinates would spend all its capacity predicting Brownian tumbling, and a
model that learned tumbling perfectly would still know nothing about the protein.

So the future structure is first canonicalised:

1. take the residues whose N/CA/C define a valid frame in **both** structures;
2. fit the proper rotation (``det = +1``, no reflection) that takes the future
   Ca positions onto the current ones;
3. apply it to the whole future structure.

What remains is the conformational change, and that is what the probe predicts.

**Conventions.** Frames follow Phase 1: ``R_i`` has the local axes as its
*columns*, so ``R_i`` maps local vectors to global ones. Both targets are
expressed in the current residue's local frame:

    delta_r_local_i = R_cur_i^T (CA_fut_aligned_i - CA_cur_i)
    R_rel_i         = R_cur_i^T R_fut_aligned_i         (right multiplication)

so that

    delta_r_global_i = R_cur_i  delta_r_local_i
    R_fut_aligned_i  = R_cur_i  R_rel_i

Local-frame targets are what make the objective SE(3)-invariant without the
network ever seeing an aligned input: a global rotation of the *whole pair* leaves
both targets unchanged, which is asserted by
``test_targets_are_invariant_under_a_global_rigid_motion``.

**The rotation target is a matrix, not a chart.** No axis-angle, no quaternion,
no 6D at this stage; a chart is only needed where a network must emit a rotation
(Checkpoint 5). See :mod:`force_md.geometry.so3`.

**Kabsch is used here and in the metrics, and nowhere else.** The alignment is
computed *from the future*, so feeding an aligned structure to the encoder would
leak the label. The model input stays in the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor

from ..data.contracts import FrameGeometry, HierarchicalProteinBatch
from ..geometry.alignment import RigidAlignment, kabsch_rotation
from ..geometry.frames import (
    ResidueFrames,
    build_residue_frames,
    link_backbone_to_atom_positions,
)
from ..geometry.so3 import relative_rotation
from ..geometry.torsions import sequence_neighbours

__all__ = [
    "TransitionTarget",
    "TransitionPrediction",
    "build_transition_target",
    "identity_prediction",
    "reconstruct_backbone",
    "apply_prediction",
]


@dataclass
class TransitionTarget:
    """Canonicalised future, as a per-residue rigid update of the current frame.

    Args:
        translation_local: ``[N_res, 3]`` Ca displacement in the current local
            frame, in the batch's length unit.
        translation_global: ``[N_res, 3]`` the same displacement in the global
            frame of the *current* structure.
        rotation: ``[N_res, 3, 3]`` ``R_cur^T R_fut_aligned``.
        valid: ``[N_res]`` bool. False where either structure has a degenerate or
            missing frame, or the residue is masked. Zeros elsewhere are padding,
            not data.
        current_frames / future_frames_aligned: the two frames the target relates.
        alignment: the rigid motion that was removed. Kept so metrics can put
            atoms into the same canonical space without refitting.
        local_n / local_c: ``[N_res, 3]`` positions of this residue's N and C in
            its own current frame -- the internal geometry a frame-level model
            does not predict and therefore carries over. See
            :func:`reconstruct_backbone`.
        residue_batch_index: ``[N_res]`` graph id.
        previous / following: ``[N_res]`` sequence neighbours or -1, for torsions.
        resid_original / chain_index: carried so the metrics can measure sequence
            separation with the source numbering rather than with row indices,
            which would be wrong across a chain break or a numbering gap.
        num_graphs: graphs in the batch.
    """

    translation_local: Tensor
    translation_global: Tensor
    rotation: Tensor
    valid: Tensor
    current_frames: ResidueFrames
    future_frames_aligned: ResidueFrames
    alignment: RigidAlignment
    local_n: Tensor
    local_c: Tensor
    residue_batch_index: Tensor
    previous: Tensor
    following: Tensor
    resid_original: Tensor
    chain_index: Tensor
    num_graphs: int

    @property
    def num_residues(self) -> int:
        return int(self.translation_local.shape[0])

    @property
    def future_ca_aligned(self) -> Tensor:
        return self.future_frames_aligned.origin

    @property
    def current_ca(self) -> Tensor:
        return self.current_frames.origin

    def to(self, device) -> "TransitionTarget":
        return replace(
            self,
            translation_local=self.translation_local.to(device),
            translation_global=self.translation_global.to(device),
            rotation=self.rotation.to(device),
            valid=self.valid.to(device),
            current_frames=self.current_frames.to(device),
            future_frames_aligned=self.future_frames_aligned.to(device),
            alignment=self.alignment.to(device),
            local_n=self.local_n.to(device),
            local_c=self.local_c.to(device),
            residue_batch_index=self.residue_batch_index.to(device),
            previous=self.previous.to(device),
            following=self.following.to(device),
            resid_original=self.resid_original.to(device),
            chain_index=self.chain_index.to(device),
        )


@dataclass
class TransitionPrediction:
    """What a transition model emits, in the same convention as the target.

    Args:
        translation_local: ``[N_res, 3]`` predicted Ca displacement in the
            current local frame.
        rotation: ``[N_res, 3, 3]`` predicted ``R_cur^T R_fut``. Must be a proper
            rotation; the probe head produces it through a 6D Gram-Schmidt map so
            this holds by construction.
    """

    translation_local: Tensor
    rotation: Tensor

    def to(self, device) -> "TransitionPrediction":
        return TransitionPrediction(
            self.translation_local.to(device), self.rotation.to(device)
        )


def _check_alignment_of_rows(
    current: HierarchicalProteinBatch, future: FrameGeometry
) -> None:
    """Refuse a future frame that is not row-for-row the same topology.

    Kabsch pairs points *by row*. If the residue ordering differed between the
    two structures the fit would still succeed and return a rotation, and every
    number downstream would be wrong with nothing to show for it -- so the
    ordering is checked rather than trusted.
    """
    future.matches(current)
    if not torch.equal(future.residue_batch_index, current.residues.batch_index):
        raise ValueError(
            "future residue_batch_index differs from the current state's; the two "
            "frames must enumerate the same residues in the same order"
        )
    if not torch.equal(future.atom_batch_index, current.atoms.batch_index):
        raise ValueError(
            "future atom_batch_index differs from the current state's; the two "
            "frames must enumerate the same atoms in the same order"
        )
    if future.frame_index.shape != current.frame_index.shape:
        raise ValueError(
            f"future frame_index has {tuple(future.frame_index.shape)} entries but "
            f"the state has {tuple(current.frame_index.shape)} graphs"
        )
    if bool((future.frame_index <= current.frame_index).any()):
        raise ValueError(
            "a future frame index is not after the current one: "
            f"current {current.frame_index.tolist()}, future {future.frame_index.tolist()}. "
            "A transition target must point forwards in time."
        )


def build_transition_target(
    current: HierarchicalProteinBatch,
    future: FrameGeometry,
    *,
    require_contiguous_resid: bool = True,
) -> TransitionTarget:
    """Canonicalise ``future`` against ``current`` and express it as a rigid update.

    Args:
        current: the state at ``t``.
        future: geometry at ``t + lag``, row-aligned with ``current``.
        require_contiguous_resid: adjacency rule for the torsion neighbours,
            matching ``graph.edges.build_sequence_edges``.

    Raises:
        ValueError: on any row/ordering mismatch, or if the future frame is not
            later than the current one.
    """
    _check_alignment_of_rows(current, future)

    # One tensor of record for geometry, exactly as ``LocalPhysicsModel.forward``
    # does. Without this the target could be defined against slightly different
    # backbone coordinates than the model conditions on: a batch may carry N/CA/C
    # as separate tensors that have drifted from ``atoms.positions`` (measured at
    # 0.059 A on the synthetic fixture), and that difference would appear as a
    # conformational change that never happened. See docs/open_questions.md 4.
    current = link_backbone_to_atom_positions(current)
    backbone = current.backbone
    current_frames = build_residue_frames(
        backbone.n_positions, backbone.ca_positions, backbone.c_positions,
        prior_valid=backbone.frame_valid,
    )
    future_frames = build_residue_frames(
        future.n_positions, future.ca_positions, future.c_positions,
        prior_valid=future.frame_valid,
    )

    valid = current_frames.valid & future_frames.valid & current.residues.mask
    batch_index = current.residues.batch_index

    # Fit on the residues that are usable in both structures, so the canonical
    # frame does not depend on rows that are about to be masked out anyway.
    alignment = kabsch_rotation(
        future.ca_positions,
        backbone.ca_positions,
        batch_index,
        current.num_graphs,
        weights=valid.to(backbone.ca_positions.dtype),
    )
    valid = valid & alignment.valid[batch_index]

    aligned_n = alignment.apply(future.n_positions, batch_index)
    aligned_ca = alignment.apply(future.ca_positions, batch_index)
    aligned_c = alignment.apply(future.c_positions, batch_index)
    future_frames_aligned = build_residue_frames(
        aligned_n, aligned_ca, aligned_c, prior_valid=future.frame_valid
    )

    rot_cur = current_frames.rotation
    translation_global = aligned_ca - backbone.ca_positions
    translation_local = torch.einsum("nji,nj->ni", rot_cur, translation_global)
    rotation = relative_rotation(rot_cur, future_frames_aligned.rotation)

    zero = torch.zeros((), dtype=translation_local.dtype, device=translation_local.device)
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(rotation)
    mask = valid.unsqueeze(-1)
    translation_local = torch.where(mask, translation_local, zero)
    translation_global = torch.where(mask, translation_global, zero)
    rotation = torch.where(valid[:, None, None], rotation, eye)

    local_n = torch.einsum("nji,nj->ni", rot_cur, backbone.n_positions - backbone.ca_positions)
    local_c = torch.einsum("nji,nj->ni", rot_cur, backbone.c_positions - backbone.ca_positions)

    previous, following = sequence_neighbours(
        batch_index, current.residues.chain_index, current.residues.resid_original,
        require_contiguous_resid=require_contiguous_resid,
    )

    return TransitionTarget(
        translation_local=translation_local,
        translation_global=translation_global,
        rotation=rotation,
        valid=valid,
        current_frames=current_frames,
        future_frames_aligned=future_frames_aligned,
        alignment=alignment,
        local_n=local_n,
        local_c=local_c,
        residue_batch_index=batch_index,
        previous=previous,
        following=following,
        resid_original=current.residues.resid_original,
        chain_index=current.residues.chain_index,
        num_graphs=current.num_graphs,
    )


def identity_prediction(target: TransitionTarget) -> TransitionPrediction:
    """The "nothing moves" baseline: zero translation, identity rotation.

    This is the reference every reported number needs, for the same reason Phase 1
    quotes ``rmse_zero`` beside every force RMSE. A 1 ns Ca RMSD of 1.2 A means
    nothing on its own; it means a great deal once you know that predicting no
    change at all scores 1.25 A.
    """
    n = target.num_residues
    dtype = target.translation_local.dtype
    device = target.translation_local.device
    return TransitionPrediction(
        translation_local=torch.zeros((n, 3), dtype=dtype, device=device),
        rotation=torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3).contiguous(),
    )


def apply_prediction(
    prediction: TransitionPrediction, target: TransitionTarget
) -> tuple[Tensor, Tensor]:
    """Turn a prediction into global frames: ``(ca [N,3], rotation [N,3,3])``.

    ``ca = CA_cur + R_cur delta_r_local`` and ``R = R_cur R_rel``, both in the
    canonical frame the target lives in.
    """
    rot_cur = target.current_frames.rotation
    ca = target.current_ca + torch.einsum(
        "nij,nj->ni", rot_cur, prediction.translation_local
    )
    return ca, rot_cur @ prediction.rotation


def reconstruct_backbone(
    prediction: TransitionPrediction, target: TransitionTarget
) -> tuple[Tensor, Tensor, Tensor]:
    """Predicted N, CA, C positions, ``([N,3], [N,3], [N,3])``.

    A frame-level model predicts where each residue's rigid frame goes, not how
    the residue deforms internally, so the reconstruction carries the residue's
    **current** internal geometry (``local_n``, ``local_c``) onto the predicted
    frame. That is an approximation and is stated as one: bond lengths and angles
    do fluctuate between frames. It is good enough for the geometric checks it
    serves -- clash rate and backbone torsions -- and applying the identical
    reconstruction to the target as well keeps the comparison honest, because both
    sides then carry the same approximation.
    """
    ca, rotation = apply_prediction(prediction, target)
    n = torch.einsum("nij,nj->ni", rotation, target.local_n) + ca
    c = torch.einsum("nij,nj->ni", rotation, target.local_c) + ca
    return n, ca, c


def target_as_prediction(target: TransitionTarget) -> TransitionPrediction:
    """The target itself, in prediction form -- the perfect-model reference."""
    return TransitionPrediction(
        translation_local=target.translation_local, rotation=target.rotation
    )
