"""H1b -- a small SE(3) correction to the coarse predicted residue frames.

H1a can rotate side chains but cannot touch the backbone, and Stage M measured
the backbone as the problem: peptide C-N bond error 4-5x the identity baseline,
inter-residue angles 3.5-5x, Ca clash rate 361-1651x. Those are *inter-residue*
quantities, so only a model that moves frames relative to one another can fix
them. That is what this module does.

**The geometry reference is the current structure, not a table.** There are no
canonical bond lengths or angles in this repository and inventing them was
forbidden. There is something better available: the current frame is a real MD
snapshot, so its own peptide geometry *is* physical. The loss asks the corrected
prediction to keep

    d_i    = |C_i - N_{i+1}|
    alpha_i = angle(CA_i, C_i, N_{i+1})
    beta_i  = angle(C_i, N_{i+1}, CA_{i+1})

near the values measured at time ``t``. Reading the current structure is not
leakage: it is the model's input.

**This is soft regularisation, not a hard constraint.** Nothing here guarantees a
physical bond; the loss penalises deviation from one. Every report line says
``constraint-regularized`` rather than "guaranteed", because the difference
matters and the brief forbids conflating them.

**The shortcut this must not take.** A refiner rewarded for physical geometry can
get most of the way there by undoing the predicted motion -- the identity
baseline has near-perfect peptide geometry, because it *is* a real frame. So the
correction magnitude, the distance from the raw coarse prediction, and the
endpoint metrics are all recorded alongside, and
``test_refiner_does_not_collapse_to_identity`` exists to catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from ..geometry.so3 import so3_exp_map
from ..geometry.torsions import wrap_to_pi

__all__ = [
    "RefinerConfig",
    "BackboneFrameConstraintRefiner",
    "FrameCorrection",
    "current_peptide_geometry",
    "peptide_geometry_loss",
    "correction_magnitude",
    "REFINER_FEATURE_BLOCKS",
    "refiner_feature_dim",
    "refiner_features",
    "reconstruct_from_frames",
    "prediction_from_frames",
    "sequence_separation_exclusion",
]


@dataclass
class RefinerConfig:
    """Sizes and caps. Every one of them a config value, none inferred.

    Args:
        hidden: trunk width.
        max_translation: hard cap in angstrom on the residual translation's
            **length**, via a ``tanh`` on the norm. A cap rather than only a
            penalty, because an unbounded correction can move a residue further
            than the whole transition and the coarse prediction stops meaning
            anything. See :meth:`BackboneFrameConstraintRefiner._cap_norm` for
            why it is the norm and not the components.
        max_rotation_rad: hard cap on the residual rotation **angle**, likewise
            on the magnitude of the rotation vector.
        predict_uncertainty: emit a per-residue log-variance for the correction.
    """

    hidden: int = 128
    max_translation: float = 1.0
    max_rotation_rad: float = math.radians(15.0)
    predict_uncertainty: bool = False


@dataclass
class FrameCorrection:
    """What the refiner emits, before it is composed with the coarse frames.

    Args:
        delta_translation: ``[N_res, 3]`` angstrom, in the **coarse predicted**
            residue frame, so the correction is equivariant by construction.
        delta_rotation: ``[N_res, 3, 3]`` proper rotation, right-multiplied onto
            the coarse frame.
        raw_translation: the unclipped head output in angstrom. This is retained
            for cap diagnostics only; it is never composed into the prediction.
        raw_rotation_vector: the unclipped Lie-algebra head output in radians.
            This is retained for cap diagnostics only; ``delta_rotation`` is
            always built from the clipped vector.
        log_variance: optional ``[N_res]``.
    """

    delta_translation: Tensor
    delta_rotation: Tensor
    log_variance: Optional[Tensor] = None
    raw_translation: Optional[Tensor] = None
    raw_rotation_vector: Optional[Tensor] = None


class BackboneFrameConstraintRefiner(nn.Module):
    """Predicts a small residual SE(3) update per residue.

    The rotation goes through the **Lie algebra**: the head emits a rotation
    vector and :func:`so3_exp_map` turns it into a proper rotation. That keeps
    ``det = +1`` by construction with no orthogonalisation step, and it is
    numerically well behaved at the small angles this head is capped to -- which
    a quaternion normalisation is not, near zero norm.
    """

    def __init__(self, in_features: int, config: RefinerConfig | None = None):
        super().__init__()
        self.config = config or RefinerConfig()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, self.config.hidden),
            nn.SiLU(),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.SiLU(),
        )
        outputs = 6 + (1 if self.config.predict_uncertainty else 0)
        self.out = nn.Linear(self.config.hidden, outputs)
        # Zero-init: an untrained refiner is the identity map, so H1b starts
        # exactly at the coarse prediction and every later number is a departure
        # from it rather than from a random pose.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _cap_norm(vector: Tensor, maximum: float, *, eps: float = 1e-12) -> Tensor:
        """Clip the vector's **length** to ``maximum``, keeping its direction.

        ``|v_out| = min(|v|, maximum)``: the identity inside the ball, a radial
        projection onto the sphere outside it. The head's raw output is therefore
        the correction in physical units directly -- angstrom, radians -- with no
        reshaping of the range it is allowed to use.

        Two earlier forms were wrong, both in ways that trained without error.

        *Per-component ``tanh``* caps each axis, so the magnitude reaches
        ``sqrt(3) * maximum`` along a diagonal. It did: the first overfit probe
        hit 1.72 A and 25.96 deg against caps of 1.0 A and 15 deg, exactly
        ``sqrt(3)`` over in both. The caps are set against measured quantities
        (mean ``|delta_r|`` 2.5 A, mean frame rotation 33.8 deg), so a factor of
        1.73 is the difference between "enough to fix a bond" and "enough to redo
        a fifth of the transition".

        *``tanh`` on the norm* fixes the bound but squashes the whole range
        beneath it: ``|v_out| = tanh(|v|) * maximum`` reaches only 76% of the cap
        at ``|v| = 1`` and approaches it asymptotically, where ``tanh'`` vanishes
        and the gradient dies. Clipping leaves everything below the cap untouched
        and full-gradient, and above it keeps the tangential gradient so the
        *direction* still trains while the magnitude is held.

        The norm is taken as ``sqrt(clamp(sum v^2, eps^2))`` rather than
        ``v.norm()`` so the square root is never evaluated at exactly zero, where
        its gradient is undefined. At the origin the scale sits at its ceiling of
        1, so a zero-init head emits exactly zero **and** keeps an identity
        gradient -- not the dead fixed point that ``tanh(n)/clamp(n, eps)`` has,
        where scale and gradient are both ``0/eps = 0`` and the correction stays
        at zero forever.
        """
        norm = vector.pow(2).sum(-1, keepdim=True).clamp(min=eps * eps).sqrt()
        return vector * (maximum / norm).clamp(max=1.0)

    def forward(self, features: Tensor) -> FrameCorrection:
        raw = self.out(self.trunk(features))
        translation = self._cap_norm(raw[:, :3], self.config.max_translation)
        rotation_vector = self._cap_norm(raw[:, 3:6], self.config.max_rotation_rad)
        correction = FrameCorrection(
            delta_translation=translation,
            delta_rotation=so3_exp_map(rotation_vector),
            log_variance=raw[:, 6] if self.config.predict_uncertainty else None,
            raw_translation=raw[:, :3],
            raw_rotation_vector=raw[:, 3:6],
        )
        return correction


def compose(
    coarse_origin: Tensor,
    coarse_rotation: Tensor,
    correction: FrameCorrection,
) -> tuple[Tensor, Tensor]:
    """Apply a correction to coarse frames: ``(origin, rotation)``.

    The translation is expressed in the coarse frame and rotated into the global
    one, and the rotation is right-multiplied. Both choices make the composition
    **equivariant**: rotating the whole input rotates the output the same way,
    with no change to the predicted correction itself.
    """
    origin = coarse_origin + torch.einsum(
        "nij,nj->ni", coarse_rotation, correction.delta_translation
    )
    return origin, coarse_rotation @ correction.delta_rotation


def prediction_from_frames(origin: Tensor, rotation: Tensor, target):
    """Express global refined frames back as a :class:`TransitionPrediction`.

    The exact inverse of ``targets.apply_prediction``:
    ``delta_r_local = R_cur^T (origin - CA_cur)`` and ``R_rel = R_cur^T R``.

    This exists so the refined output can be scored by ``transition_loss`` --
    *the same* loss the coarse arm was trained with, with the same weights. H1b
    is only worth having if it fixes geometry **without** giving back the
    transition accuracy, and the cheapest way to keep that honest is to optimise
    and report the original objective rather than a lookalike written for this
    stage.
    """
    from ..transition.targets import TransitionPrediction  # noqa: PLC0415

    rotation_current = target.current_frames.rotation
    return TransitionPrediction(
        translation_local=torch.einsum(
            "nji,nj->ni", rotation_current, origin - target.current_ca
        ),
        rotation=rotation_current.transpose(-1, -2) @ rotation,
    )


def sequence_separation_exclusion(
    residue_index: Tensor, batch_index: Tensor, *, min_separation: int = 2
) -> Tensor:
    """``[N, N]`` bool: pairs too close in sequence (or in different graphs).

    Used for the **training** overlap penalty over reconstructed N/CA/C atoms,
    where no PSF mask is available because the atoms are constructed rather than
    read. Excluding ``|i - j| < 2`` removes every 1-2 and 1-3 pair among N, CA
    and C -- and also the 1-4 pairs (``N_i-N_{i+1}``, ``CA_i-CA_{i+1}``) that the
    H0 *metric* deliberately keeps.

    That difference is intentional and one-directional: the training penalty is a
    strict subset of the reported one, so H1b cannot lower its loss by rearranging
    pairs the metric would still count. The report always quotes the full
    PSF-based H0 metric, never this.
    """
    same_graph = batch_index[:, None] == batch_index[None, :]
    separation = (residue_index[:, None] - residue_index[None, :]).abs()
    return (~same_graph) | (separation < min_separation)


def correction_magnitude(correction: FrameCorrection) -> dict[str, Tensor]:
    """How far the refiner moved things. Recorded on every H1b evaluation.

    Without this a refiner that quietly reverts the transition looks like a
    refiner that learned physics.
    """
    from ..geometry.so3 import rotation_geodesic_angle

    with torch.no_grad():
        raw_translation = (
            correction.raw_translation
            if correction.raw_translation is not None
            else correction.delta_translation
        )
        raw_rotation = (
            correction.raw_rotation_vector
            if correction.raw_rotation_vector is not None
            else None
        )
        raw_rotation_deg = (
            torch.rad2deg(raw_rotation.norm(dim=-1))
            if raw_rotation is not None
            else torch.rad2deg(rotation_geodesic_angle(correction.delta_rotation))
        )
        return {
            "correction_translation_norm": correction.delta_translation.norm(dim=-1),
            "correction_rotation_deg": torch.rad2deg(
                rotation_geodesic_angle(correction.delta_rotation)
            ),
            "raw_translation_norm": raw_translation.norm(dim=-1),
            "raw_rotation_deg": raw_rotation_deg,
        }


def _angle_at(apex: Tensor, a: Tensor, b: Tensor, *, eps: float = 1e-12) -> Tensor:
    u = a - apex
    v = b - apex
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=eps)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.atan2(torch.linalg.cross(u, v, dim=-1).norm(dim=-1), (u * v).sum(-1))


def current_peptide_geometry(
    n_positions: Tensor,
    ca_positions: Tensor,
    c_positions: Tensor,
    following: Tensor,
) -> dict[str, Tensor]:
    """Peptide bond length and the two angles across it, measured at time ``t``.

    This is the reference H1b regularises towards. It is a **measurement of the
    model's own input**, not a constant table and not a future quantity, so using
    it introduces no leakage. Entries where there is no next residue are masked
    by ``valid``.
    """
    valid = following >= 0
    following_safe = following.clamp(min=0)
    n_next = n_positions[following_safe]
    ca_next = ca_positions[following_safe]
    return {
        "valid": valid,
        "bond_c_n": (n_next - c_positions).norm(dim=-1),
        "angle_ca_c_n": _angle_at(c_positions, ca_positions, n_next),
        "angle_c_n_ca": _angle_at(n_next, c_positions, ca_next),
    }


def peptide_geometry_loss(
    predicted: dict[str, Tensor],
    reference: dict[str, Tensor],
    *,
    bond_weight: float = 1.0,
    angle_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Squared deviation of predicted peptide geometry from the current frame's.

    Returned per term rather than summed, because the report has to show which
    term moved. Angles are compared through :func:`wrap_to_pi`, which for values
    in ``[0, pi]`` is the identity but costs nothing and protects a caller that
    passes signed angles.
    """
    valid = predicted["valid"] & reference["valid"]
    if not bool(valid.any()):
        zero = predicted["bond_c_n"].new_tensor(0.0)
        return {"bond": zero, "angle_ca_c_n": zero, "angle_c_n_ca": zero}
    bond = (predicted["bond_c_n"] - reference["bond_c_n"])[valid].pow(2).mean()
    first = wrap_to_pi(
        predicted["angle_ca_c_n"] - reference["angle_ca_c_n"]
    )[valid].pow(2).mean()
    second = wrap_to_pi(
        predicted["angle_c_n_ca"] - reference["angle_c_n_ca"]
    )[valid].pow(2).mean()
    return {
        "bond": bond_weight * bond,
        "angle_ca_c_n": angle_weight * first,
        "angle_c_n_ca": angle_weight * second,
    }


# --------------------------------------------------------------------------
# what the refiner is allowed to look at
# --------------------------------------------------------------------------

#: ``(name, width)`` in concatenation order. Every block is a **rotation- and
#: translation-invariant scalar**, so the only thing carrying a global frame is
#: the correction itself, which :func:`compose` expresses in the coarse predicted
#: frame. That is what makes the refined frames equivariant by construction
#: rather than by training, and ``test_refiner_features_are_invariant`` pins it.
#:
#: None of these come from inside the coarse probe. The refiner sees the coarse
#: **output** and the current structure, both already public, so the frozen model
#: is not modified and H1b cannot quietly borrow its representation capacity --
#: a fair capacity comparison later depends on that.
REFINER_FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("own_translation_local", 3),
    ("own_rotation", 9),
    ("own_local_n", 3),
    ("own_local_c", 3),
    ("peptide_to_next", 3),
    ("peptide_to_next_valid", 1),
    ("peptide_to_previous", 3),
    ("peptide_to_previous_valid", 1),
    ("next_translation_local", 3),
    ("next_rotation", 9),
    ("previous_translation_local", 3),
    ("previous_rotation", 9),
    ("next_ca_in_own_frame", 3),
    ("previous_ca_in_own_frame", 3),
    ("lag_ns", 1),
)


def refiner_feature_dim() -> int:
    """Width of :func:`refiner_features`, derived from the block table.

    Derived rather than written down, so a block added to the table cannot leave
    a stale constant behind for the head's ``in_features`` to disagree with.
    """
    return sum(width for _name, width in REFINER_FEATURE_BLOCKS)


def reconstruct_from_frames(
    origin: Tensor, rotation: Tensor, local_n: Tensor, local_c: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """``(N, CA, C)`` from a frame plus the residue's own internal geometry.

    The same carry-the-current-internal-geometry approximation
    ``targets.reconstruct_backbone`` makes, applied here to whichever frames the
    caller has -- current, coarse or refined. Using one function for all three
    keeps the peptide geometry measured at ``t`` and the peptide geometry the
    loss penalises on exactly the same footing; measuring them two different ways
    would put a constant offset into the loss and call it physics.
    """
    n = torch.einsum("nij,nj->ni", rotation, local_n) + origin
    c = torch.einsum("nij,nj->ni", rotation, local_c) + origin
    return n, origin, c


def _gather_neighbour(values: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    """``values[index]`` with ``-1`` meaning 'no neighbour', zeroed and flagged."""
    valid = index >= 0
    gathered = values[index.clamp(min=0)]
    shape = (-1,) + (1,) * (gathered.dim() - 1)
    return gathered * valid.view(shape).to(gathered.dtype), valid


def refiner_features(prediction, target, lag_ns: Tensor) -> Tensor:
    """``[N_res, refiner_feature_dim()]`` invariant features for H1b.

    H1b's job is *inter-residue*: the quantities Stage M found broken -- peptide
    C-N bond, the two angles across it, Cα-Cα spacing -- all live between a
    residue and its sequence neighbours, and none of them can be fixed by a
    residue that can only see itself. So the window is the residue plus its
    previous and following neighbours, and the block table above is that window
    written out.

    Args:
        prediction: the frozen coarse :class:`TransitionPrediction`.
        target: the :class:`TransitionTarget` for the same batch. **Only its
            current-side fields are read** -- ``current_frames``, ``current_ca``,
            ``local_n``, ``local_c``, ``previous``, ``following``,
            ``residue_batch_index``. The future-side fields are not touched here;
            they are the supervision, and a feature builder that read them would
            be leakage. ``test_refiner_features_ignore_the_future`` perturbs the
            future fields and requires the output not to move.
        lag_ns: ``[num_graphs]`` physical lag. Broadcast per residue.
    """
    rotation_current = target.current_frames.rotation
    ca_current = target.current_ca
    n_current, _, c_current = reconstruct_from_frames(
        ca_current, rotation_current, target.local_n, target.local_c
    )
    to_next = current_peptide_geometry(
        n_current, ca_current, c_current, target.following
    )
    # The bond and the two angles *behind* this residue are the same three
    # numbers read from the previous residue's row, not a second measurement.
    previous = target.previous
    behind_valid = previous >= 0
    behind = torch.stack(
        [
            to_next["bond_c_n"][previous.clamp(min=0)],
            to_next["angle_ca_c_n"][previous.clamp(min=0)],
            to_next["angle_c_n_ca"][previous.clamp(min=0)],
        ],
        dim=-1,
    ) * behind_valid.unsqueeze(-1).to(ca_current.dtype)

    ahead = torch.stack(
        [to_next["bond_c_n"], to_next["angle_ca_c_n"], to_next["angle_c_n_ca"]],
        dim=-1,
    ) * to_next["valid"].unsqueeze(-1).to(ca_current.dtype)

    translation = prediction.translation_local
    rotation = prediction.rotation.reshape(-1, 9)
    next_translation, next_valid = _gather_neighbour(translation, target.following)
    next_rotation, _ = _gather_neighbour(rotation, target.following)
    previous_translation, previous_valid = _gather_neighbour(translation, previous)
    previous_rotation, _ = _gather_neighbour(rotation, previous)

    # A neighbour's Cα seen in *this* residue's current frame: invariant, and the
    # only place the window's actual spacing enters.
    def relative_ca(index: Tensor) -> Tensor:
        gathered, valid = _gather_neighbour(ca_current, index)
        offset = torch.where(
            valid.unsqueeze(-1), gathered - ca_current, torch.zeros_like(ca_current)
        )
        return torch.einsum("nji,nj->ni", rotation_current, offset)

    lag = lag_ns.to(ca_current.dtype)[target.residue_batch_index].unsqueeze(-1)
    blocks = [
        translation,
        rotation,
        target.local_n,
        target.local_c,
        ahead,
        to_next["valid"].unsqueeze(-1).to(ca_current.dtype),
        behind,
        behind_valid.unsqueeze(-1).to(ca_current.dtype),
        next_translation,
        next_rotation,
        previous_translation,
        previous_rotation,
        relative_ca(target.following),
        relative_ca(previous),
        lag,
    ]
    features = torch.cat(blocks, dim=-1)
    expected = refiner_feature_dim()
    if features.shape[-1] != expected:
        raise ValueError(
            f"refiner_features produced {features.shape[-1]} columns but "
            f"REFINER_FEATURE_BLOCKS declares {expected}. The block table and "
            "the builder have drifted apart."
        )
    del next_valid, previous_valid
    return features


def soft_overlap_penalty(
    positions: Tensor,
    vdw_radius: Tensor,
    excluded: Tensor,
    valid: Tensor,
    *,
    tolerance: float = 0.4,
) -> Tensor:
    """Differentiable Bondi overlap penalty, ``mean(relu(o_ij - tolerance)^2)``.

    The same ``o_ij = r_i + r_j - d_ij`` the H0 metric reports, hinged at the
    same 0.4 A so the thing optimised and the thing measured agree. Squared
    rather than linear so a single deep interpenetration dominates a scatter of
    shallow touches, which is the physically right ordering.
    """
    keep = valid.nonzero(as_tuple=True)[0]
    if keep.numel() < 2:
        return positions.new_tensor(0.0)
    selected = positions[keep]
    radii = vdw_radius[keep]
    distance = torch.cdist(
        selected, selected, compute_mode="donot_use_mm_for_euclid_dist"
    )
    overlap = radii[:, None] + radii[None, :] - distance
    mask = torch.triu(
        torch.ones_like(overlap, dtype=torch.bool), diagonal=1
    ) & ~excluded[keep][:, keep]
    if not bool(mask.any()):
        return positions.new_tensor(0.0)
    hinged = torch.relu(overlap[mask] - tolerance)
    return hinged.pow(2).mean()
