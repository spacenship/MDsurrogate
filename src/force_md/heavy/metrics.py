"""Heavy-atom metrics: RMSD decomposition, Bondi overlap, atom contacts.

Every metric here carries its own provenance (§3.3 of the brief), because at
heavy-atom resolution the same number can mean three different things depending
on how the coordinates were made. A side-chain RMSD computed on a *transported*
conformer is a statement about frame placement; the same number on a *predicted*
conformer would be a statement about side-chain modelling. The record says which.

**Overlap, not "clash".** The quantity is

    o_ij = r_i^vdW + r_j^vdW - d_ij

and it is reported as a rate at three fixed depths, with 0.4 A as the primary.
The name ``bondi_*`` is deliberate: it says which radii were used, so nobody has
to guess whether a rate is comparable with a published one.

**Exclusions come from the force field.** 1-2 and 1-3 pairs are excluded using
the PSF's own bond and angle lists, not a distance heuristic and not a graph
traversal. 1-4 pairs stay **in** the primary total -- an eclipsed 1-4 contact is
a real strain, not a bookkeeping artefact -- and are also tallied separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .chemistry import OVERLAP_THRESHOLDS, VDW_RADIUS_SOURCE

__all__ = [
    "AtomClassification",
    "HeavyAtomMetricConfig",
    "heavy_atom_rmsd",
    "bondi_overlap",
    "atom_contact_scores",
    "pair_class_masks",
]

_NAN = float("nan")


@dataclass(frozen=True)
class HeavyAtomMetricConfig:
    """Thresholds, every one of them fixed before any H0 number was produced.

    Args:
        contact_cutoff: any-heavy-atom contact distance, angstrom. 5.0 A is the
            usual all-atom contact definition and is **not** revisited after
            seeing a result.
        contact_min_residue_separation: minimum ``|resid_i - resid_j|`` within a
            chain for an atom pair to count as a contact. 2 rather than the
            Cα map's 6, because atoms of residues 3 apart genuinely pack against
            each other; the Cα definition's larger value exists to skip the
            near-diagonal band a Cα trace cannot resolve.
        overlap_thresholds: depth in angstrom -> label. Primary is ``serious``.
        exclude_1_4: keep False. 1-4 pairs are reported separately and stay in
            the primary total.
        serious_label: which threshold the primary rate uses.
    """

    contact_cutoff: float = 5.0
    contact_min_residue_separation: int = 2
    overlap_thresholds: tuple[tuple[str, float], ...] = tuple(
        OVERLAP_THRESHOLDS.items()
    )
    exclude_1_4: bool = False
    serious_label: str = "serious"
    vdw_source: str = VDW_RADIUS_SOURCE


@dataclass
class AtomClassification:
    """Per-atom labels the metrics slice by. All ``[N_atom]``.

    Args:
        is_backbone: N, CA, C, O (and OXT). From ``residue_constants``.
        is_sidechain: heavy and not backbone and not a terminal cap patch.
        residue_index: row in the residue arrays.
        vdw_radius: Bondi radius, angstrom.
        valid: usable in this comparison (present and non-padding in both
            structures, residue frame valid).
    """

    is_backbone: Tensor
    is_sidechain: Tensor
    residue_index: Tensor
    vdw_radius: Tensor
    valid: Tensor


def heavy_atom_rmsd(
    predicted: Tensor, target: Tensor, classification: AtomClassification
) -> dict[str, float]:
    """All-heavy / backbone / side-chain RMSD, plus counts.

    No superposition: the transition target already removed the global rigid
    motion once, so these are directly comparable with the Cα RMSD beside them.
    """
    valid = classification.valid
    error = (predicted - target).pow(2).sum(-1)

    def rmsd(mask: Tensor) -> tuple[float, int]:
        selected = mask & valid
        count = int(selected.sum())
        if count == 0:
            return _NAN, 0
        return float(error[selected].mean().sqrt()), count

    all_rmsd, n_all = rmsd(torch.ones_like(valid))
    bb_rmsd, n_bb = rmsd(classification.is_backbone)
    sc_rmsd, n_sc = rmsd(classification.is_sidechain)
    return {
        "heavy_atom_rmsd": all_rmsd,
        "backbone_heavy_rmsd": bb_rmsd,
        "sidechain_heavy_rmsd": sc_rmsd,
        "n_valid_atoms": n_all,
        "n_valid_backbone_atoms": n_bb,
        "n_valid_sidechain_atoms": n_sc,
    }


def sidechain_centroid_error(
    predicted: Tensor, target: Tensor, classification: AtomClassification
) -> dict[str, float]:
    """Per-residue side-chain centre-of-geometry displacement error.

    Coarser than side-chain RMSD and less sensitive to a single flipped terminal
    group, so the two disagreeing is informative rather than contradictory.
    """
    mask = classification.is_sidechain & classification.valid
    if not bool(mask.any()):
        return {"sidechain_centroid_error": _NAN, "n_sidechain_residues": 0}
    residue = classification.residue_index[mask]
    unique, inverse = torch.unique(residue, return_inverse=True)
    counts = torch.zeros(len(unique), device=predicted.device).index_add_(
        0, inverse, torch.ones(len(residue), device=predicted.device)
    )
    def centroid(x: Tensor) -> Tensor:
        total = torch.zeros((len(unique), 3), dtype=x.dtype, device=x.device)
        total.index_add_(0, inverse, x[mask])
        return total / counts.unsqueeze(-1)
    displacement = (centroid(predicted) - centroid(target)).norm(dim=-1)
    return {
        "sidechain_centroid_error": float(displacement.mean()),
        "n_sidechain_residues": int(len(unique)),
    }


def pair_class_masks(classification: AtomClassification) -> dict[str, Tensor]:
    """``[n, n]`` masks for the backbone/side-chain pair decomposition."""
    bb = classification.is_backbone
    sc = classification.is_sidechain
    return {
        "backbone_backbone": bb[:, None] & bb[None, :],
        "backbone_sidechain": (bb[:, None] & sc[None, :]) | (sc[:, None] & bb[None, :]),
        "sidechain_sidechain": sc[:, None] & sc[None, :],
    }


def _pairwise(points: Tensor) -> Tensor:
    p = points.to(torch.float64)
    return torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")


def bondi_overlap(
    positions: Tensor,
    classification: AtomClassification,
    excluded: Tensor,
    is_1_4: Tensor,
    config: HeavyAtomMetricConfig = HeavyAtomMetricConfig(),
) -> dict:
    """Bondi van der Waals overlap statistics for one structure.

    Args:
        excluded: ``[n, n]`` bool, 1-2 and 1-3 pairs from the PSF, diagonal set.
        is_1_4: ``[n, n]`` bool, 1-4 pairs, already disjoint from ``excluded``.

    Returns a flat dict of rates, depths and counts. ``inter_residue`` is the
    primary; ``intra_residue`` overlap under a rigid transport is
    construction-invariant and is reported only as a sanity check.
    """
    valid = classification.valid
    n = int(positions.shape[0])
    device = positions.device
    usable = valid[:, None] & valid[None, :]
    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1
    )
    nonbonded = upper & usable & ~excluded
    if config.exclude_1_4:
        nonbonded = nonbonded & ~is_1_4

    distance = _pairwise(positions)
    radii = classification.vdw_radius.to(torch.float64)
    overlap = radii[:, None] + radii[None, :] - distance

    same_residue = (
        classification.residue_index[:, None] == classification.residue_index[None, :]
    )
    scopes = {
        "inter_residue": nonbonded & ~same_residue,
        "intra_residue": nonbonded & same_residue,
    }
    classes = pair_class_masks(classification)

    out: dict = {
        "n_nonbonded_pairs": int(nonbonded.sum()),
        "n_1_4_pairs": int((is_1_4 & upper & usable).sum()),
        "n_valid_heavy_atoms": int(valid.sum()),
        "vdw_source": config.vdw_source,
    }
    primary = scopes["inter_residue"]
    for label, depth in config.overlap_thresholds:
        hit = primary & (overlap >= depth) if depth > 0 else primary & (overlap > 0.0)
        count = int(hit.sum())
        total = int(primary.sum())
        out[f"bondi_overlap_rate_{label}"] = count / total if total else _NAN
        out[f"bondi_overlap_count_{label}"] = count

    serious_depth = dict(config.overlap_thresholds)[config.serious_label]
    serious = primary & (overlap >= serious_depth)
    total = int(primary.sum())
    n_atoms = int(valid.sum())
    out["bondi_heavy_overlap_rate"] = (
        int((primary & (overlap > 0.0)).sum()) / total if total else _NAN
    )
    out["bondi_serious_overlap_rate"] = int(serious.sum()) / total if total else _NAN
    depths = overlap[primary & (overlap > 0.0)]
    out["bondi_overlap_depth_mean"] = float(depths.mean()) if depths.numel() else 0.0
    out["bondi_overlap_depth_max"] = float(depths.max()) if depths.numel() else 0.0
    out["bondi_clashes_per_1000_heavy_atoms"] = (
        1000.0 * int(serious.sum()) / n_atoms if n_atoms else _NAN
    )

    for scope_name, scope in scopes.items():
        for class_name, class_mask in classes.items():
            selected = scope & class_mask
            total_c = int(selected.sum())
            hit = int((selected & (overlap >= serious_depth)).sum())
            out[f"bondi_serious_{scope_name}_{class_name}_rate"] = (
                hit / total_c if total_c else _NAN
            )
            out[f"bondi_serious_{scope_name}_{class_name}_count"] = hit
    fourteen = is_1_4 & upper & usable & ~same_residue
    total_14 = int(fourteen.sum())
    out["bondi_serious_1_4_rate"] = (
        int((fourteen & (overlap >= serious_depth)).sum()) / total_14
        if total_14 else _NAN
    )
    return out


def atom_contact_scores(
    predicted: Tensor,
    target: Tensor,
    current: Optional[Tensor],
    classification: AtomClassification,
    resid_original: Tensor,
    chain_index: Tensor,
    config: HeavyAtomMetricConfig = HeavyAtomMetricConfig(),
) -> dict:
    """Any-heavy-atom contact F1, split by pair class, plus formation/breakage.

    ``current`` may be ``None``, in which case the formed/broken block is
    omitted rather than filled with zeros.
    """
    valid = classification.valid
    n = int(predicted.shape[0])
    device = predicted.device
    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1
    )
    residue = classification.residue_index
    same_chain = chain_index[residue][:, None] == chain_index[residue][None, :]
    separation = (
        resid_original[residue][:, None] - resid_original[residue][None, :]
    ).abs()
    far_enough = (~same_chain) | (separation >= config.contact_min_residue_separation)
    eligible = upper & valid[:, None] & valid[None, :] & far_enough

    cutoff = config.contact_cutoff
    contact_pred = _pairwise(predicted) < cutoff
    contact_targ = _pairwise(target) < cutoff

    out: dict = {"n_eligible_atom_pairs": int(eligible.sum())}
    classes = {"any": torch.ones_like(eligible), **pair_class_masks(classification)}
    for name, class_mask in classes.items():
        scope = eligible & class_mask
        out.update(
            _prefixed(
                f"atom_contact_{name}", contact_pred, contact_targ, scope
            )
        )
    if current is None:
        return out

    contact_curr = _pairwise(current) < cutoff
    for event, condition in (
        ("formed", eligible & ~contact_curr),
        ("broken", eligible & contact_curr),
    ):
        want = contact_targ if event == "formed" else ~contact_targ
        got = contact_pred if event == "formed" else ~contact_pred
        out.update(_prefixed(f"atom_contact_{event}", got, want, condition))
    return out


def _prefixed(prefix: str, predicted: Tensor, target: Tensor, scope: Tensor) -> dict:
    tp = int((predicted & target & scope).sum())
    fp = int((predicted & ~target & scope).sum())
    fn = int((~predicted & target & scope).sum())
    precision = tp / (tp + fp) if (tp + fp) else _NAN
    recall = tp / (tp + fn) if (tp + fn) else _NAN
    # Same F1 convention as the Phase 1.6 extended suite: 2TP/(2TP+FP+FN) is the
    # harmonic mean wherever that is defined and stays defined when only one of
    # precision and recall is, so a model that predicted nothing scores 0 on
    # events that happened rather than NA.
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else _NAN
    return {
        f"{prefix}_tp": tp, f"{prefix}_fp": fp, f"{prefix}_fn": fn,
        f"{prefix}_precision": precision, f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_events_target": int((target & scope).sum()),
        f"{prefix}_events_predicted": int((predicted & scope).sum()),
    }
