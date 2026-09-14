"""H0 -- place heavy atoms on predicted residue frames, and the scoring oracles.

The transition probe predicts a rigid update per residue frame and nothing else.
H0 asks what that implies at atomic resolution: take each residue's heavy atoms
*as they are now*, expressed in its current frame, and put them on the frame the
model predicted.

    y_ia = R_i(t)^T (x_ia(t) - r_i(t))          local, measured now
    x_ia = r_hat_i   + R_hat_i y_ia             placed on the predicted frame

**This is transport, not prediction.** The side-chain conformation is a copy of
the current one; the model never predicted a chi angle. Everything downstream
carries ``construction_mode="transported_current_conformer"`` so no table can
read H0 as a side-chain result. What H0 *does* measure honestly is what
frame-placement error does to inter-residue geometry: steric overlap, atom
contacts, and packing that a Cα-only view cannot see.

**The three oracles are diagnostics, not models.** Two of them read the future,
which is why each carries ``uses_future_for_scoring_only=True`` and why none of
them is registered as an arm. They exist to decompose the total heavy-atom error
into the part frame prediction is responsible for and the part it cannot reach:

* ``future_frame_current_local`` -- perfect frames, current side chains. The
  **internal-conformation floor**: the error left when frame prediction is
  solved and the conformer is still a copy.
* ``predicted_frame_future_local`` -- predicted frames, true side chains. The
  **frame-only component**.
* ``identity_current_atoms`` -- the current structure, unmoved. The baseline
  every number is quoted against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data.contracts import FrameGeometry, HierarchicalProteinBatch
from ..geometry.frames import build_residue_frames, link_backbone_to_atom_positions
from ..transition.targets import TransitionPrediction, TransitionTarget, apply_prediction

__all__ = [
    "HeavyAtomPlacement",
    "CONSTRUCTION_MODES",
    "local_atom_coordinates",
    "place_on_frames",
    "backmap_prediction",
    "future_frame_current_local",
    "predicted_frame_future_local",
    "identity_current_atoms",
    "heavy_atom_placements",
]

#: Every placement this module can produce, and what it is allowed to claim.
#: Consumed by the metric records so provenance is never re-derived by hand.
CONSTRUCTION_MODES: dict[str, dict] = {
    "transported_current_conformer": {
        "description": "current local heavy atoms placed on the model's predicted frames",
        "is_model_predicted": True,
        "uses_future_for_scoring_only": False,
        "sidechain_conformation_predicted": False,
    },
    "future_frame_current_local": {
        "description": "current local heavy atoms placed on the TRUE future frames",
        "is_model_predicted": False,
        "uses_future_for_scoring_only": True,
        "sidechain_conformation_predicted": False,
    },
    "predicted_frame_future_local": {
        "description": "TRUE future local heavy atoms placed on the predicted frames",
        "is_model_predicted": False,
        "uses_future_for_scoring_only": True,
        "sidechain_conformation_predicted": False,
    },
    "identity_current_atoms": {
        "description": "the current heavy atoms, unmoved",
        "is_model_predicted": False,
        "uses_future_for_scoring_only": False,
        "sidechain_conformation_predicted": False,
    },
}


@dataclass
class HeavyAtomPlacement:
    """One set of heavy-atom coordinates and what produced it.

    Args:
        positions: ``[N_atom, 3]``, row-aligned with the batch's atom array.
        construction_mode: a key of :data:`CONSTRUCTION_MODES`.
        valid: ``[N_atom]`` bool -- atoms whose residue had a usable frame in
            every structure the placement needed.
    """

    positions: Tensor
    construction_mode: str
    valid: Tensor

    @property
    def provenance(self) -> dict:
        return {"construction_mode": self.construction_mode,
                **CONSTRUCTION_MODES[self.construction_mode]}


def local_atom_coordinates(
    positions: Tensor,
    rotation: Tensor,
    origin: Tensor,
    atom_to_residue: Tensor,
) -> Tensor:
    """``y_ia = R_i^T (x_ia - r_i)`` -- atoms in their own residue's frame.

    ``R`` has the local axes as **columns** (the Phase 1 convention), so the
    transpose maps global to local. Getting this backwards silently mirrors every
    side chain, which for a chiral molecule is not a cosmetic error.
    """
    relative = positions - origin[atom_to_residue]
    return torch.einsum("aji,aj->ai", rotation[atom_to_residue], relative)


def place_on_frames(
    local: Tensor,
    rotation: Tensor,
    origin: Tensor,
    atom_to_residue: Tensor,
) -> Tensor:
    """``x_ia = r_i + R_i y_ia`` -- the inverse of :func:`local_atom_coordinates`."""
    return (
        torch.einsum("aij,aj->ai", rotation[atom_to_residue], local)
        + origin[atom_to_residue]
    )


def _residue_valid_to_atoms(valid: Tensor, atom_to_residue: Tensor) -> Tensor:
    return valid[atom_to_residue]


def backmap_prediction(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    batch: HierarchicalProteinBatch,
) -> HeavyAtomPlacement:
    """H0's required mode: current local atoms on the predicted frames.

    Reads no future coordinate. ``target`` is used only for the *current* frames
    it already carries and for the residue validity mask -- both are properties
    of time ``t``. ``test_backmapping_ignores_the_future`` asserts that by
    replacing the future with noise and requiring identical output.
    """
    linked = link_backbone_to_atom_positions(batch)
    atom_to_residue = linked.atoms.atom_to_residue
    current = target.current_frames
    local = local_atom_coordinates(
        linked.atoms.positions, current.rotation, current.origin, atom_to_residue
    )
    origin, rotation = apply_prediction(prediction, target)
    return HeavyAtomPlacement(
        positions=place_on_frames(local, rotation, origin, atom_to_residue),
        construction_mode="transported_current_conformer",
        valid=_residue_valid_to_atoms(target.valid, atom_to_residue),
    )


def future_frame_current_local(
    target: TransitionTarget, batch: HierarchicalProteinBatch
) -> HeavyAtomPlacement:
    """Scoring-only oracle: current local atoms on the **true** future frames.

    The internal-conformation floor. Whatever error this leaves is error no
    amount of frame accuracy can remove, because the side chain is still a copy
    of the current one.
    """
    linked = link_backbone_to_atom_positions(batch)
    atom_to_residue = linked.atoms.atom_to_residue
    current = target.current_frames
    local = local_atom_coordinates(
        linked.atoms.positions, current.rotation, current.origin, atom_to_residue
    )
    future = target.future_frames_aligned
    return HeavyAtomPlacement(
        positions=place_on_frames(local, future.rotation, future.origin, atom_to_residue),
        construction_mode="future_frame_current_local",
        valid=_residue_valid_to_atoms(target.valid, atom_to_residue),
    )


def predicted_frame_future_local(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    batch: HierarchicalProteinBatch,
    future: FrameGeometry,
) -> HeavyAtomPlacement:
    """Scoring-only oracle: **true future** local atoms on the predicted frames.

    Isolates the frame-prediction component. This one reads future *atom*
    coordinates and is therefore the most obviously non-deployable of the three;
    it is never registered as an arm and never reaches a conditioner.
    """
    linked = link_backbone_to_atom_positions(batch)
    atom_to_residue = linked.atoms.atom_to_residue
    aligned_future_atoms = target.alignment.apply(
        future.positions, future.atom_batch_index
    )
    future_frames = target.future_frames_aligned
    local = local_atom_coordinates(
        aligned_future_atoms, future_frames.rotation, future_frames.origin, atom_to_residue
    )
    origin, rotation = apply_prediction(prediction, target)
    return HeavyAtomPlacement(
        positions=place_on_frames(local, rotation, origin, atom_to_residue),
        construction_mode="predicted_frame_future_local",
        valid=_residue_valid_to_atoms(target.valid, atom_to_residue),
    )


def identity_current_atoms(
    target: TransitionTarget, batch: HierarchicalProteinBatch
) -> HeavyAtomPlacement:
    """The current heavy atoms, unmoved -- the "nothing happens" reference."""
    linked = link_backbone_to_atom_positions(batch)
    return HeavyAtomPlacement(
        positions=linked.atoms.positions.clone(),
        construction_mode="identity_current_atoms",
        valid=_residue_valid_to_atoms(target.valid, linked.atoms.atom_to_residue),
    )


def target_heavy_atoms(
    target: TransitionTarget, future: FrameGeometry, batch: HierarchicalProteinBatch
) -> Tensor:
    """The scoring target: future heavy atoms in the canonical frame.

    The **same** alignment the transition target used -- fitted once, from the
    future Cα onto the current Cα -- is applied to the future atom array. No
    second superposition, so a heavy-atom RMSD here is comparable with the Cα
    RMSD beside it.
    """
    return target.alignment.apply(future.positions, future.atom_batch_index)


def heavy_atom_placements(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    batch: HierarchicalProteinBatch,
    future: FrameGeometry,
    *,
    with_oracles: bool = True,
) -> dict[str, HeavyAtomPlacement]:
    """Every placement H0 scores, keyed by construction mode."""
    out = {
        "transported_current_conformer": backmap_prediction(prediction, target, batch),
        "identity_current_atoms": identity_current_atoms(target, batch),
    }
    if with_oracles:
        out["future_frame_current_local"] = future_frame_current_local(target, batch)
        out["predicted_frame_future_local"] = predicted_frame_future_local(
            prediction, target, batch, future
        )
    return out
