"""Phase 1 losses.

    L = lambda_atom  * NLL(atom force)
      + lambda_res   * NLL(residue force)
      + lambda_tau   * NLL(residue torque)
      + lambda_hidden* NLL(omitted-atom residual)      [only when identifiable]
      + lambda_cons  * ||explained - aggregate||^2
      + lambda_energy* ||-grad U - f_target||^2
      + lambda_gauge * mean(U^2)

Three things deserve attention:

**Errors are evaluated in the residue-local frame.** The heads predict a diagonal
covariance, and a diagonal is only meaningful in a fixed frame. Rotating the
error into the local frame first is what makes the uncertainty
representation-invariant.

**The hidden residual is excluded from the aggregation-consistency term.** That
term ties the residue-level *explained* force to the sum of the predicted atom
forces. The hidden residual is by definition the part **not** carried by the
represented atoms, so including it would ask the two terms to cancel each other
and would let the model hide arbitrary mass in the residual.

**The conservative-force term is auxiliary, not an identity.** It pulls
``-grad U`` toward the force label, but the full effective force is not claimed
to be conservative -- the residual head carries the rest, and
``lambda_energy`` is small by default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data.contracts import HierarchicalProteinBatch
from ..geometry.frames import ResidueFrames, to_local_vectors
from .outputs import Phase1Output
from .projection import ResidueForceTargets

__all__ = ["LossWeights", "TargetNormalizer", "masked_gaussian_nll", "masked_mse",
           "phase1_loss"]

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class LossWeights:
    """Loss weights. Deliberately un-tuned in this first implementation.

    The NLL terms are the actual objective and sit at 1.0. The auxiliary terms
    are small: they are regularisers, and letting one of them dominate would
    quietly change what the model is being asked to do.
    """

    atom_force: float = 1.0
    residue_force: float = 1.0
    torque: float = 1.0
    hidden_force: float = 1.0
    aggregation_consistency: float = 0.1
    conservative_force: float = 0.1
    energy_gauge: float = 1e-4


@dataclass(frozen=True)
class TargetNormalizer:
    """Scales that put every target in O(1) before the loss.

    Fitted on the **training split only** -- fitting on all data leaks validation
    statistics into training. Defaults of 1.0 mean "unnormalised", which is
    correct but poorly conditioned; :meth:`fit` produces real values.

    Units: forces are kcal/mol/A and torques kcal/mol, so their scales differ by
    roughly a lever arm and must not share one number.
    """

    atom_force: float = 1.0
    residue_force: float = 1.0
    residue_torque: float = 1.0

    @staticmethod
    def fit(
        atom_forces: Tensor, residue_forces: Tensor, residue_torques: Tensor
    ) -> "TargetNormalizer":
        """RMS magnitude of each target. Call on the train split only."""
        rms = lambda t: float(t.reshape(-1, 3).pow(2).sum(-1).mean().sqrt())  # noqa: E731
        return TargetNormalizer(
            atom_force=max(rms(atom_forces), 1e-6),
            residue_force=max(rms(residue_forces), 1e-6),
            residue_torque=max(rms(residue_torques), 1e-6),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _masked_mean(per_node: Tensor, mask: Tensor) -> Tensor:
    """Mean over unmasked nodes; exactly zero when nothing is unmasked."""
    m = mask.to(per_node.dtype)
    denom = m.sum().clamp(min=1.0)
    return (per_node * m).sum() / denom


def masked_gaussian_nll(
    prediction: Tensor,
    target: Tensor,
    logvar: Tensor,
    mask: Tensor,
    *,
    scale: float = 1.0,
    frames: Optional[ResidueFrames] = None,
    index: Optional[Tensor] = None,
) -> Tensor:
    """Heteroscedastic diagonal Gaussian NLL over masked nodes.

    Args:
        prediction / target: ``[N, 3]`` global-frame vectors.
        logvar: ``[N, 3]`` log-variance of the **normalised** error, in the
            local frame when ``frames`` is given.
        mask: ``[N]`` bool; masked-out nodes contribute nothing.
        scale: divides the error before the NLL, so ``logvar`` is in normalised
            units.
        frames / index: rotate the error into the residue-local frame first.

    Returns:
        Scalar mean NLL per node (summed over the 3 components).
    """
    error = (prediction - target) / scale
    if frames is not None:
        if index is None:
            raise ValueError("frames given without index")
        error = to_local_vectors(error, frames, index)
    per_component = 0.5 * (_LOG_2PI + logvar + error.pow(2) * torch.exp(-logvar))
    return _masked_mean(per_component.sum(-1), mask)


def masked_mse(
    prediction: Tensor, target: Tensor, mask: Tensor, *, scale: float = 1.0
) -> Tensor:
    """Mean squared error over masked nodes, in normalised units."""
    return _masked_mean(((prediction - target) / scale).pow(2).sum(-1), mask)


def phase1_loss(
    output: Phase1Output,
    batch: HierarchicalProteinBatch,
    residue_targets: ResidueForceTargets,
    frames: ResidueFrames,
    *,
    hidden_force_target: Optional[Tensor] = None,
    atom_selection: Optional[Tensor] = None,
    weights: LossWeights = LossWeights(),
    normalizer: TargetNormalizer = TargetNormalizer(),
) -> tuple[Tensor, dict[str, float]]:
    """Total Phase 1 loss and its components.

    Args:
        output: model output.
        batch: the input batch, for labels and masks.
        residue_targets: labels from :class:`ResidueSumProjector`.
        frames: residue frames, for the local-frame error.
        hidden_force_target: ``[N_res, 3]`` omitted-atom residual label. Required
            iff the model predicts a hidden force; passing None with a predicting
            model raises, because an unsupervised residual absorbs arbitrary mass.
        atom_selection: ``[N_atom]`` bool, which atoms carry a usable force label.
        weights / normalizer: see above.

    Returns:
        ``(total, components)`` where ``components`` maps name -> float.
    """
    parts: dict[str, Tensor] = {}
    device = output.residue_force_mean.device
    zero = torch.zeros((), device=device, dtype=output.residue_force_mean.dtype)

    # ---- atom force NLL -------------------------------------------------
    atoms = batch.atoms
    if atoms.forces is not None:
        atom_mask = torch.ones(atoms.num_atoms, dtype=torch.bool, device=device)
        if atoms.force_valid is not None:
            atom_mask = atom_mask & atoms.force_valid
        if atom_selection is not None:
            atom_mask = atom_mask & atom_selection
        parts["atom_force_nll"] = masked_gaussian_nll(
            output.atom_force_mean, atoms.forces, output.atom_force_logvar,
            atom_mask, scale=normalizer.atom_force,
            frames=frames, index=atoms.atom_to_residue,
        )
    else:
        # No atomic labels: mask the term entirely and train on residue targets.
        parts["atom_force_nll"] = zero

    # ---- residue force / torque NLL -------------------------------------
    res_mask = residue_targets.valid & batch.residues.mask
    res_index = torch.arange(batch.num_residues, device=device)
    parts["residue_force_nll"] = masked_gaussian_nll(
        output.residue_force_mean, residue_targets.force, output.residue_force_logvar,
        res_mask, scale=normalizer.residue_force, frames=frames, index=res_index,
    )
    parts["torque_nll"] = masked_gaussian_nll(
        output.residue_torque_mean, residue_targets.torque, output.residue_torque_logvar,
        res_mask, scale=normalizer.residue_torque, frames=frames, index=res_index,
    )

    # ---- omitted-atom residual ------------------------------------------
    if output.residue_hidden_force is not None:
        if hidden_force_target is None:
            raise ValueError(
                "the model predicts residue_hidden_force but no target was given. "
                "An unsupervised additive residual is unidentifiable and will "
                "absorb arbitrary mass; either supply the all-atom minus "
                "heavy-atom residual or disable the head."
            )
        parts["hidden_force_mse"] = masked_mse(
            output.residue_hidden_force, hidden_force_target, res_mask,
            scale=normalizer.residue_force,
        )
    else:
        parts["hidden_force_mse"] = zero

    # ---- aggregation consistency (hidden residual deliberately excluded) --
    parts["aggregation_consistency"] = masked_mse(
        output.residue_explained_force, output.aggregated_atom_force, res_mask,
        scale=normalizer.residue_force,
    )

    # ---- conservative force (auxiliary) ----------------------------------
    if output.atom_force_conservative is not None and atoms.forces is not None:
        cons_mask = torch.ones(atoms.num_atoms, dtype=torch.bool, device=device)
        if atoms.force_valid is not None:
            cons_mask = cons_mask & atoms.force_valid
        if atom_selection is not None:
            cons_mask = cons_mask & atom_selection
        parts["conservative_force_mse"] = masked_mse(
            output.atom_force_conservative, atoms.forces, cons_mask,
            scale=normalizer.atom_force,
        )
    else:
        parts["conservative_force_mse"] = zero

    # ---- energy gauge ----------------------------------------------------
    # mdCATH has no energy label, so only the gradient of U is supervised. That
    # leaves the overall scale and offset of U unconstrained; this term keeps the
    # gauge bounded. It is not a fit to any measured energy.
    parts["energy_gauge"] = output.energy.pow(2).mean()

    total = (
        weights.atom_force * parts["atom_force_nll"]
        + weights.residue_force * parts["residue_force_nll"]
        + weights.torque * parts["torque_nll"]
        + weights.hidden_force * parts["hidden_force_mse"]
        + weights.aggregation_consistency * parts["aggregation_consistency"]
        + weights.conservative_force * parts["conservative_force_mse"]
        + weights.energy_gauge * parts["energy_gauge"]
    )
    components = {k: float(v.detach()) for k, v in parts.items()}
    components["total"] = float(total.detach())
    return total, components
