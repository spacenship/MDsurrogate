"""Transition losses.

    L = lambda_pos   * robust translation loss
      + lambda_rot   * rotation loss
      + lambda_pair  * pair-distance auxiliary
      + lambda_clash * differentiable clash penalty (optional, small)

**Position and rotation are divided by their own scales before being added.**
Angstrom and radians are not commensurable, and adding them raw silently sets
the trade-off to whatever the units happen to be. The default scales are the
measured identity-baseline magnitudes at these lags -- ~2.5 A of Ca displacement
and ~0.5 rad of frame rotation -- so both terms enter at O(1) and ``lambda_pos``
and ``lambda_rot`` mean what they say. The numbers and their source are in
:class:`TransitionLossWeights`, not buried in a formula.

**The rotation loss is chordal by default, and reported as geodesic.** The
geodesic angle ``theta = atan2(|vee(R - R^T)/2|, (tr R - 1)/2)`` is the right
*metric* and a poor *loss*: its gradient carries a ``1/sin(theta)`` factor and is
singular at ``theta = 0``, which is exactly where a converging model sits. The
chordal loss ``||R_pred - R_target||_F^2 = 8 sin^2(theta/2)`` is smooth
everywhere, monotone in ``theta`` on ``[0, pi]``, and has the same minimiser.
``rotation_loss="geodesic"`` is available for comparison and documented as the
less stable option rather than removed.

Everything is masked to residues that are valid in **both** structures, and the
mean is over unmasked rows, so a batch with a masked residue does not quietly
scale the loss down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor

from ..geometry.so3 import rotation_geodesic_angle
from .targets import TransitionPrediction, TransitionTarget, apply_prediction

__all__ = ["TransitionLossWeights", "transition_loss"]


@dataclass(frozen=True)
class TransitionLossWeights:
    """Loss weights and the unit scales they are defined against.

    Args:
        translation / rotation: the two objectives. Equal by default -- placing a
            residue and orienting it are both the task, and there is no measured
            reason yet to prefer one.
        pair_distance: auxiliary term on Ca-Ca distances. Small: it is a shape
            regulariser, and letting it dominate would ask the model to preserve
            the current shape rather than to predict its change.
        clash: hinge penalty on non-neighbouring Ca closer than
            ``clash_min_angstrom``. **Zero by default.** Measured on real mdCATH
            pairs the clash rate of both the current and the future structure is
            0.000, so there is nothing for it to fix until a model starts
            producing collapses; it is implemented so that it can be switched on
            with evidence rather than added in a hurry.
        translation_scale_angstrom: divisor for the translation term. 2.5 A is
            the measured mean ``|delta_r|`` over real 1-4 ns pairs.
        rotation_scale_radians: divisor for the rotation term. 0.59 rad is the
            measured mean residue-frame rotation (33.8 degrees).
        huber_delta: transition to linear in the robust translation loss, in units
            of ``translation_scale_angstrom``. Residues in flexible tails move
            several times the mean and would otherwise dominate the gradient.
        rotation_loss: ``"chordal"`` (default, smooth) or ``"geodesic"``.
        pair_cutoff_angstrom: only pairs within this distance in the **current**
            structure contribute, so the term is about local packing rather than
            about the overall diameter.
    """

    translation: float = 1.0
    rotation: float = 1.0
    pair_distance: float = 0.1
    clash: float = 0.0
    translation_scale_angstrom: float = 2.5
    rotation_scale_radians: float = 0.59
    huber_delta: float = 1.0
    rotation_loss: str = "chordal"
    pair_cutoff_angstrom: float = 12.0
    clash_min_angstrom: float = 3.6

    def as_dict(self) -> dict:
        return asdict(self)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp(min=1.0)


def _pair_terms(
    predicted_ca: Tensor,
    target_ca: Tensor,
    current_ca: Tensor,
    batch_index: Tensor,
    valid: Tensor,
    weights: TransitionLossWeights,
) -> tuple[Tensor, Tensor]:
    """Pair-distance auxiliary and clash penalty, per graph.

    Loops over graphs because the pair set is ragged. Graph counts here are small
    (tens), and the alternative -- a padded dense pair tensor -- is the layout this
    project does not have.
    """
    zero = predicted_ca.new_zeros(())
    pair_total, pair_count = zero.clone(), 0.0
    clash_total, clash_count = zero.clone(), 0.0

    for graph in torch.unique(batch_index):
        rows = ((batch_index == graph) & valid).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        p, t, c = predicted_ca[rows], target_ca[rows], current_ca[rows]
        upper = torch.triu(
            torch.ones((rows.numel(), rows.numel()), dtype=torch.bool, device=p.device),
            diagonal=1,
        )
        d_current = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist")
        local = upper & (d_current <= weights.pair_cutoff_angstrom)
        if bool(local.any()):
            d_pred = torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")
            d_targ = torch.cdist(t, t, compute_mode="donot_use_mm_for_euclid_dist")
            pair_total = pair_total + (d_pred[local] - d_targ[local]).abs().sum()
            pair_count += float(local.sum())

        if weights.clash > 0:
            # Sequence-adjacent Ca are ~3.8 A apart by construction; only pairs
            # further apart along the chain can be said to clash.
            separation = (rows[:, None] - rows[None, :]).abs()
            eligible = upper & (separation > 2)
            if bool(eligible.any()):
                d_pred = torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")
                violation = torch.relu(weights.clash_min_angstrom - d_pred[eligible])
                clash_total = clash_total + violation.pow(2).sum()
                clash_count += float(eligible.sum())

    pair = pair_total / max(pair_count, 1.0)
    clash = clash_total / max(clash_count, 1.0)
    return pair, clash


def transition_loss(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    *,
    weights: TransitionLossWeights = TransitionLossWeights(),
) -> tuple[Tensor, dict[str, float]]:
    """Total loss and its components.

    Returns:
        ``(total, components)`` where ``components`` holds detached floats for
        logging, including the raw (un-normalised) translation and rotation errors
        in Angstrom and degrees so a training log is readable in physical units.
    """
    valid = target.valid
    if not bool(valid.any()):
        raise ValueError(
            "no valid residue in this batch: every residue was masked or had a "
            "degenerate frame in one of the two structures"
        )

    # -- translation, robust, in units of the measured displacement scale ---
    error = (prediction.translation_local - target.translation_local)
    distance = error.norm(dim=-1) / weights.translation_scale_angstrom
    delta = weights.huber_delta
    huber = torch.where(
        distance <= delta, 0.5 * distance.pow(2), delta * (distance - 0.5 * delta)
    )
    translation = _masked_mean(huber, valid)

    # -- rotation -----------------------------------------------------------
    angle = rotation_geodesic_angle(prediction.rotation, target.rotation)
    if weights.rotation_loss == "chordal":
        # ||R_a - R_b||_F^2 = 8 sin^2(theta/2): smooth at theta = 0.
        difference = prediction.rotation - target.rotation
        raw_rotation = difference.pow(2).sum(dim=(-2, -1))
        scale = 8.0 * torch.sin(
            torch.tensor(weights.rotation_scale_radians / 2.0, device=angle.device)
        ) ** 2
        rotation = _masked_mean(raw_rotation, valid) / scale
    elif weights.rotation_loss == "geodesic":
        rotation = _masked_mean(
            (angle / weights.rotation_scale_radians).pow(2), valid
        )
    else:
        raise ValueError(
            f"rotation_loss must be 'chordal' or 'geodesic', got "
            f"{weights.rotation_loss!r}"
        )

    # -- auxiliaries --------------------------------------------------------
    predicted_ca, _ = apply_prediction(prediction, target)
    pair, clash = _pair_terms(
        predicted_ca,
        target.future_ca_aligned,
        target.current_ca,
        target.residue_batch_index,
        valid,
        weights,
    )

    total = (
        weights.translation * translation
        + weights.rotation * rotation
        + weights.pair_distance * pair
        + weights.clash * clash
    )

    with torch.no_grad():
        components = {
            "total": float(total),
            "translation": float(translation),
            "rotation": float(rotation),
            "pair_distance": float(pair),
            "clash": float(clash),
            # physical units, for a log a human can read
            "translation_rmse_angstrom": float(
                _masked_mean(error.pow(2).sum(-1), valid).sqrt()
            ),
            "rotation_error_deg": float(torch.rad2deg(_masked_mean(angle, valid))),
            "valid_residues": float(valid.sum()),
        }
    return total, components
