"""Extended geometry metrics for a predicted transition (Phase 1.6, Stage M).

`metrics.py` answers "how far is the prediction from the target". This module
answers the question Stage B could not: **what kind of structure is the model
getting wrong**. It is an extension of that module, not a replacement -- it
imports the same target contract, the same Kabsch, the same geodesic angle and
the same torsion code, so a number here that shares a name with a number there
has the same definition.

Four families are added.

*Pair geometry.* ``dRMSD`` is the alignment-free view: it compares the *internal*
distance matrix, so it cannot be flattered or punished by a superposition. Split
by sequence separation, because ``|i-j| <= 2`` is fixed by the peptide bond and a
model that only gets that right has learned chemistry, not folding. The
``|i-j| >= 6`` subset is the one Phase 1.6's pair architecture exists to move.

*Contacts, and contact **events**.* A static future contact F1 at these lags is
dominated by the contacts that never changed -- the identity baseline scores
~0.82 -- so the formed/broken decomposition is what carries information about
transitions. An arm that improves static F1 while predicting no formation event
at all has improved nothing.

*Backbone torsions*, now including omega, with circular error: ``179 deg`` and
``-179 deg`` are two degrees apart.

*Physical validity.* Only what the reconstruction can actually support. The model
predicts residue **frames**, and ``reconstruct_backbone`` carries each residue's
*current* internal geometry onto the predicted frame, identically for the target.
Every intra-residue quantity is therefore equal on both sides **by construction**
-- N-CA and CA-C bond lengths, the N-CA-C angle, and Cbeta chirality. Those are
reported as ``None`` with a reason in :data:`NOT_APPLICABLE`, never as 0.0, and
``test_intra_residue_geometry_is_invariant_by_construction`` is the evidence.
Only the inter-residue quantities -- the C-N(next) peptide bond, the two angles
across it, omega, and consecutive Cα distance -- are measurements.

**Nothing here re-superposes by default.** ``build_transition_target`` already
removed the global rigid motion once, fitting the future onto the current; the
prediction is built from the current frames, so both live in that one canonical
frame and every metric below is already invariant to a rigid motion of the whole
pair. ``ca_rmsd`` and ``rotation_geodesic_mean_deg`` therefore reproduce the
Stage B numbers exactly. The re-superposed variants are computed as well, under
names that say so, and there the *same* rotation is applied to the frames as to
the coordinates -- which the existing ``ca_rmsd_aligned`` paired with
``rotation_geodesic_deg`` was not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..geometry.alignment import kabsch_rotation
from ..geometry.so3 import rotation_geodesic_angle
from ..geometry.torsions import backbone_omega, backbone_torsions, wrap_to_pi
from ..graph.edges import build_knn_edges
from .targets import (
    TransitionPrediction,
    TransitionTarget,
    apply_prediction,
    reconstruct_backbone,
    target_as_prediction,
)

__all__ = [
    "ExtendedMetricConfig",
    "NOT_APPLICABLE",
    "EXTENDED_METRIC_KEYS",
    "HIGHER_IS_BETTER",
    "extended_graph_metrics",
    "extended_metric_records",
    "length_bin",
]

_NAN = float("nan")

#: Metrics this reconstruction cannot support, and why. Written into every
#: manifest and every report so a null is never read as a zero or as an
#: oversight. Brief section 11: "0 으로 채워서 정상값처럼 보이게 만들지 않는다."
NOT_APPLICABLE: dict[str, str] = {
    "bond_n_ca_rmse": (
        "intra-residue: reconstruct_backbone carries the residue's current local N "
        "onto both the predicted and the target frame, so this error is identically "
        "zero by construction and measures nothing"
    ),
    "bond_ca_c_rmse": (
        "intra-residue: same construction as bond_n_ca_rmse"
    ),
    "angle_n_ca_c_mae_deg": (
        "intra-residue: N, CA and C of one residue are rigidly transported together, "
        "so this angle is identical in prediction and target by construction"
    ),
    "bond_length_violation_rate": (
        "no canonical bond-length range is defined in this repository "
        "(residue_constants.py has no bond table) and no chemistry library is "
        "installed in the `md` env; choosing a threshold now would be choosing it "
        "after seeing the results"
    ),
    "bond_angle_violation_rate": (
        "no canonical bond-angle range is defined in this repository; same reason as "
        "bond_length_violation_rate"
    ),
    "chirality_violation_count": (
        "the model predicts residue frames, not atoms: there is no Cbeta in the "
        "reconstruction. Carrying the current local Cbeta onto the predicted frame "
        "would make chirality invariant by construction, and a check that cannot "
        "fail is not a measurement"
    ),
    "heavy_atom_clash_rate": (
        "no side-chain coordinates are predicted; the Cα-only clash metric is "
        "reported instead and is labelled as such in clash_definition"
    ),
    "dssp_class": (
        "neither mdtraj nor Biopython is installed in the `md` env, and a "
        "re-evaluation of frozen checkpoints must not add a dependency"
    ),
    "rmsf_flexibility_class": (
        "the manifest samples 4 frames per trajectory; an RMSF over 4 frames is not "
        "an RMSF, and no reference trajectory is loaded by this evaluation"
    ),
}


@dataclass(frozen=True)
class ExtendedMetricConfig:
    """Thresholds and subset definitions. Every one of them fixed before the run.

    Args:
        contact_cutoff: primary Cα-Cα contact distance, Angstrom. 8.0 A, the
            usual Cα contact definition, and **not** revisited after seeing a
            result.
        contact_sequence_separation: primary minimum ``|resid_i - resid_j|``
            within a chain. 6, per the Phase 1.6 brief -- larger than the
            metrics.py legacy value of 3, which is why both are reported.
        contact_sensitivity_cutoffs: appendix only. Reported to show the primary
            conclusion is not an artefact of 8.0 A; never used to choose an arm.
        legacy_contact_cutoff / legacy_contact_sequence_separation: the Stage B
            definition, recomputed so the extended table can be checked against
            the published one.
        drmsd_local / drmsd_medium: inclusive ``|i-j|`` ranges.
        drmsd_long_range_min: inclusive lower bound for the long-range subset.
        spatial_knn / spatial_cutoff: the residue graph the probe actually built,
            for ``drmsd_current_spatial_edges``. Defaults match
            ``TransitionProbeConfig``; the evaluator overrides them from each
            arm's own saved config rather than trusting these.
        clash_min_ca_distance / clash_sequence_separation: reused verbatim from
            :class:`~force_md.transition.metrics.MetricConfig`.
        length_bin_edges: upper-exclusive edges for the protein-length
            stratification.
        torsion_histogram_bins: bins per 360 degrees for the optional torsion
            distribution diagnostic.
    """

    contact_cutoff: float = 8.0
    contact_sequence_separation: int = 6
    contact_sensitivity_cutoffs: tuple[float, ...] = (6.0, 10.0)
    legacy_contact_cutoff: float = 8.0
    legacy_contact_sequence_separation: int = 3
    drmsd_local: tuple[int, int] = (1, 2)
    drmsd_medium: tuple[int, int] = (3, 5)
    drmsd_long_range_min: int = 6
    spatial_knn: int = 16
    spatial_cutoff: float = 13.0
    clash_min_ca_distance: float = 3.6
    clash_sequence_separation: int = 2
    length_bin_edges: tuple[int, ...] = (100, 150, 200)
    torsion_histogram_bins: int = 36

    #: Fewest valid residues for a sample to produce any metric at all. Below
    #: three the Kabsch re-superposition is rank-deficient and a distance matrix
    #: has almost no pairs; such a sample is recorded with an invalid_reason
    #: rather than with numbers nobody should read.
    min_valid_residues: int = 4

    def contact_key_suffixes(self) -> tuple[str, ...]:
        return tuple(f"cut{c:g}" for c in self.contact_sensitivity_cutoffs)

    def __post_init__(self) -> None:
        """Refuse a config whose sensitivity cutoffs are not in the fixed schema.

        The appendix keys are named after their cutoff, so changing
        ``contact_sensitivity_cutoffs`` silently invents keys that
        :data:`EXTENDED_METRIC_KEYS` does not list -- and a row whose schema is
        wider than the declared one breaks every downstream reader that trusts
        the declaration. Adding a cutoff is fine; adding it here *and* to the key
        list is what this insists on.
        """
        unknown = [
            f"contact_f1_{s}" for s in self.contact_key_suffixes()
            if f"contact_f1_{s}" not in EXTENDED_METRIC_KEYS
        ]
        if unknown:
            raise ValueError(
                f"contact_sensitivity_cutoffs would produce {unknown}, which are "
                "not in EXTENDED_METRIC_KEYS. Add them there (and to "
                "HIGHER_IS_BETTER) so the record schema stays declared."
            )


#: Metrics where a **larger** value is better. Used by the delta-sign
#: normalisation so that "positive = the candidate improved" holds in every table
#: (brief section 13.2), and by nothing else.
HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "contact_precision", "contact_recall", "contact_f1", "contact_jaccard",
    "formed_contact_precision", "formed_contact_recall", "formed_contact_f1",
    "broken_contact_precision", "broken_contact_recall", "broken_contact_f1",
    "contact_f1_legacy_sep3",
    "contact_f1_cut6", "contact_f1_cut10",
})


#: Every float metric key a record carries, in report order. Fixed so that a
#: sample with too few valid residues still writes a full row of nulls instead of
#: a short row that silently changes the schema mid-file.
EXTENDED_METRIC_KEYS: tuple[str, ...] = (
    # endpoint
    "ca_rmsd",
    "ca_rmsd_superposed",
    "translation_rmse",
    "rotation_geodesic_mean_deg",
    "rotation_geodesic_median_deg",
    "rotation_geodesic_superposed_mean_deg",
    "rotation_geodesic_superposed_median_deg",
    # pair geometry
    "drmsd_all",
    "drmsd_local",
    "drmsd_medium",
    "drmsd_long_range",
    "drmsd_current_spatial_edges",
    # contacts
    "contact_precision", "contact_recall", "contact_f1", "contact_jaccard",
    "formed_contact_precision", "formed_contact_recall", "formed_contact_f1",
    "broken_contact_precision", "broken_contact_recall", "broken_contact_f1",
    "contact_f1_legacy_sep3",
    "contact_f1_cut6", "contact_f1_cut10",
    # torsions
    "phi_mae_deg", "psi_mae_deg", "omega_mae_deg", "backbone_torsion_mae_deg",
    # physical validity
    "bond_length_rmse", "bond_length_mae",
    "bond_angle_mae_deg",
    "bond_angle_ca_c_n_mae_deg", "bond_angle_c_n_ca_mae_deg",
    "ca_neighbor_distance_mae",
    "clash_rate", "clash_rate_target",
    # unsupported, always None
    "bond_n_ca_rmse", "bond_ca_c_rmse", "angle_n_ca_c_mae_deg",
    "chirality_violation_count",
)

#: Integer counts a record carries beside the floats.
EXTENDED_COUNT_KEYS: tuple[str, ...] = (
    "n_residues", "n_valid_residues", "n_valid_pairs", "n_valid_torsions",
    "n_phi", "n_psi", "n_omega",
    "n_pairs_local", "n_pairs_medium", "n_pairs_long_range",
    "n_current_spatial_edges",
    "contact_tp", "contact_fp", "contact_fn", "contact_target_positives",
    "formed_contact_tp", "formed_contact_fp", "formed_contact_fn",
    "formed_contact_events_target", "formed_contact_events_predicted",
    "broken_contact_tp", "broken_contact_fp", "broken_contact_fn",
    "broken_contact_events_target", "broken_contact_events_predicted",
    "n_bond_c_n_next", "n_bond_angles", "n_ca_neighbor",
    "clash_count", "clash_count_target", "n_clash_pairs",
)


def length_bin(n_valid: int, edges: Sequence[int]) -> str:
    """``"<100"``, ``"100-149"``, ... , ``">=200"`` -- a stable, sortable label."""
    previous = 0
    for edge in edges:
        if n_valid < edge:
            return f"<{edge}" if previous == 0 else f"{previous}-{edge - 1}"
        previous = edge
    return f">={previous}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _pairwise(points: Tensor) -> Tensor:
    """``[n, n]`` Euclidean distances, float64, numerically safe mode.

    ``donot_use_mm_for_euclid_dist`` for the same reason as everywhere else in
    this project: the matrix-multiply expansion of ``|a|^2 + |b|^2 - 2a.b`` is
    catastrophically unstable when coordinates are large relative to the
    distances, which is exactly the regime of a protein at 100 A from the origin
    whose contacts are 8 A apart.
    """
    p = points.to(torch.float64)
    return torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")


def _separation(resid: Tensor, chain: Tensor) -> Tensor:
    """``[n, n]`` int64 sequence separation, with cross-chain pairs at a sentinel.

    Separation uses the **source residue numbering** within a chain, never the
    row index: a numbering gap means residues are missing from the structure, and
    treating rows 40 and 41 as adjacent across a gap would put a genuinely
    long-range pair into the local subset. Residues on different chains have no
    sequence separation at all, so they are given a sentinel above every real
    separation, which places them in the long-range subset -- where a
    through-space relation between two chains belongs.
    """
    gap = (resid[:, None] - resid[None, :]).abs()
    same_chain = chain[:, None] == chain[None, :]
    sentinel = torch.full_like(gap, torch.iinfo(torch.int64).max // 4)
    return torch.where(same_chain, gap, sentinel)


def _upper(n: int, device) -> Tensor:
    return torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)


def _rmsd_of(diff: Tensor, mask: Tensor) -> float:
    """RMS of ``diff`` over ``mask``; NaN when the mask selects nothing."""
    if not bool(mask.any()):
        return _NAN
    return float(diff[mask].pow(2).mean().sqrt())


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    """``(precision, recall, f1, jaccard)``, with NaN only where truly undefined.

    Three decisions, all of which the identity baseline forces:

    *A sample with no event on either side is NA, not 1.0.* The target formed no
    contact and the model predicted none: there was nothing to get right. Scoring
    that 1.0 would let an arm that never predicts an event win the formed-contact
    table outright, since most samples are of exactly this kind.

    *Precision and recall keep their own empty denominators as NaN.* The identity
    baseline predicts no formed contact at all, so its formed-contact precision is
    ``0/0``. Reporting 0.0 there would assert the model got its predictions wrong;
    it made none.

    *F1 uses the ``2tp / (2tp + fp + fn)`` form.* That is algebraically the
    harmonic mean wherever the harmonic mean is defined, and it stays defined when
    only one of precision and recall is -- which is right, because the harmonic
    mean tends to 0 as either factor does, whatever the other one is. So the
    identity baseline scores **0.0** on formed contacts that did happen (it missed
    all of them) rather than NA, and NA only where no event existed at all. NA and
    0.0 are counted separately in the aggregation, so neither can hide in a mean.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else _NAN
    recall = tp / (tp + fn) if (tp + fn) > 0 else _NAN
    union = tp + fp + fn
    jaccard = tp / union if union > 0 else _NAN
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else _NAN
    return precision, recall, f1, jaccard


def _legacy_f1(predicted: Tensor, target: Tensor, eligible: Tensor) -> float:
    """Stage B's contact F1, reproduced bit for bit from ``metrics.py``.

    Kept separate from :func:`_prf` on purpose. That function's F1 stays defined
    when only one denominator is; this one returns NaN there, exactly as
    ``metrics._graph_metrics`` does. Reproducing the published number means
    reproducing its edge cases too, not only its happy path.
    """
    tp = float((predicted & target & eligible).sum())
    n_pred = float((predicted & eligible).sum())
    n_targ = float((target & eligible).sum())
    precision = tp / n_pred if n_pred > 0 else _NAN
    recall = tp / n_targ if n_targ > 0 else _NAN
    if precision != precision or recall != recall:
        return _NAN
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _contact_scores(
    predicted: Tensor, target: Tensor, eligible: Tensor
) -> tuple[int, int, int, float, float, float, float]:
    """``(tp, fp, fn, precision, recall, f1, jaccard)`` over ``eligible`` pairs."""
    tp = int((predicted & target & eligible).sum())
    fp = int((predicted & ~target & eligible).sum())
    fn = int((~predicted & target & eligible).sum())
    return (tp, fp, fn, *_prf(tp, fp, fn))


def _current_spatial_pair_mask(
    current_ca_graph: Tensor, keep: Tensor, *, k: int, cutoff: float
) -> Tensor:
    """``[n, n]`` upper-triangular mask of the probe's own current spatial edges.

    ``current_ca_graph`` is **every** residue of the graph, valid or not, because
    that is what ``TransitionProbe._graph`` passes to ``build_knn_edges`` -- the
    neighbour list is built before any validity mask is applied, and rebuilding
    it on the valid subset would give a different graph. The mask is then
    restricted to ``keep`` (the valid rows) and re-indexed into the valid
    subset's numbering.

    Only the **current** coordinates are read. The future structure is not an
    argument of this function and cannot influence which edges are selected;
    ``test_current_edge_selection_ignores_the_future`` asserts that by replacing
    the future with noise.
    """
    m = int(current_ca_graph.shape[0])
    device = current_ca_graph.device
    batch = torch.zeros(m, dtype=torch.int64, device=device)
    edges = build_knn_edges(current_ca_graph, batch, k=k, cutoff=cutoff)
    full = torch.zeros((m, m), dtype=torch.bool, device=device)
    if edges.src.numel():
        # kNN is directed and asymmetric; the pair (i, j) is one undirected pair
        # either way round, so symmetrise before taking the upper triangle.
        full[edges.src, edges.dst] = True
        full = full | full.transpose(0, 1)
    index = keep.nonzero(as_tuple=True)[0]
    sub = full[index][:, index]
    return sub & _upper(int(index.numel()), device)


# --------------------------------------------------------------------------
# per-graph metrics
# --------------------------------------------------------------------------


def extended_graph_metrics(
    *,
    pred_ca: Tensor,
    targ_ca: Tensor,
    curr_ca: Tensor,
    pred_rotation: Tensor,
    targ_rotation: Tensor,
    pred_backbone: tuple[Tensor, Tensor, Tensor],
    targ_backbone: tuple[Tensor, Tensor, Tensor],
    translation_error: Tensor,
    resid: Tensor,
    chain: Tensor,
    previous: Tensor,
    following: Tensor,
    spatial_mask: Tensor,
    config: ExtendedMetricConfig,
) -> dict:
    """Every extended metric for one graph's valid residues.

    All tensors are already restricted to the valid rows of one graph and are
    row-aligned; ``previous`` / ``following`` are indices **into that subset**,
    ``-1`` where the neighbour is absent or was masked out.

    Returns a dict keyed by :data:`EXTENDED_METRIC_KEYS` and
    :data:`EXTENDED_COUNT_KEYS`, plus ``invalid_reason``.
    """
    out: dict = {key: None for key in EXTENDED_METRIC_KEYS}
    out.update({key: 0 for key in EXTENDED_COUNT_KEYS})
    out["invalid_reason"] = None
    out["clash_definition"] = (
        f"ca_only_min_distance_{config.clash_min_ca_distance:g}A"
        f"_sep>={config.clash_sequence_separation}"
    )
    out["bond_length_scope"] = "peptide_C_N_next_only"
    out["bond_angle_scope"] = "CA_C_Nnext_and_C_Nnext_CAnext"

    n = int(pred_ca.shape[0])
    out["n_valid_residues"] = n
    if n < config.min_valid_residues:
        out["invalid_reason"] = (
            f"only {n} valid residues (< {config.min_valid_residues}); the "
            "re-superposition is rank-deficient and the distance matrix has too "
            "few pairs to mean anything"
        )
        return out

    device = pred_ca.device
    if not (torch.isfinite(pred_ca).all() and torch.isfinite(targ_ca).all()):
        out["invalid_reason"] = "non-finite coordinate in the prediction or target"
        return out

    # -- endpoint -------------------------------------------------------
    out["ca_rmsd"] = float((pred_ca - targ_ca).pow(2).sum(-1).mean().sqrt())
    out["translation_rmse"] = float(translation_error.pow(2).sum(-1).mean().sqrt())

    angle = torch.rad2deg(rotation_geodesic_angle(pred_rotation, targ_rotation))
    out["rotation_geodesic_mean_deg"] = float(angle.mean())
    out["rotation_geodesic_median_deg"] = float(angle.median())

    # Re-superposed view. The *same* rotation Q is applied to the frames as to the
    # coordinates (R -> Q R, the column convention), which is what makes the two
    # re-superposed numbers a consistent pair. Reported as a diagnostic; the
    # canonical-frame values above are the ones that reproduce Stage B.
    zeros = torch.zeros(n, dtype=torch.int64, device=device)
    alignment = kabsch_rotation(pred_ca, targ_ca, zeros, 1)
    if bool(alignment.valid[0]):
        superposed = alignment.apply(pred_ca, zeros)
        out["ca_rmsd_superposed"] = float(
            (superposed - targ_ca).pow(2).sum(-1).mean().sqrt()
        )
        rotated = alignment.apply_frames(pred_rotation, zeros)
        angle_s = torch.rad2deg(rotation_geodesic_angle(rotated, targ_rotation))
        out["rotation_geodesic_superposed_mean_deg"] = float(angle_s.mean())
        out["rotation_geodesic_superposed_median_deg"] = float(angle_s.median())

    # -- pair geometry --------------------------------------------------
    d_pred = _pairwise(pred_ca)
    d_targ = _pairwise(targ_ca)
    d_curr = _pairwise(curr_ca)
    error = d_pred - d_targ

    upper = _upper(n, device)
    separation = _separation(resid, chain)
    local_lo, local_hi = config.drmsd_local
    medium_lo, medium_hi = config.drmsd_medium
    subsets = {
        "drmsd_all": upper,
        "drmsd_local": upper & (separation >= local_lo) & (separation <= local_hi),
        "drmsd_medium": upper & (separation >= medium_lo) & (separation <= medium_hi),
        "drmsd_long_range": upper & (separation >= config.drmsd_long_range_min),
        "drmsd_current_spatial_edges": upper & spatial_mask,
    }
    for name, mask in subsets.items():
        out[name] = _rmsd_of(error, mask)
    out["n_valid_pairs"] = int(subsets["drmsd_all"].sum())
    out["n_pairs_local"] = int(subsets["drmsd_local"].sum())
    out["n_pairs_medium"] = int(subsets["drmsd_medium"].sum())
    out["n_pairs_long_range"] = int(subsets["drmsd_long_range"].sum())
    out["n_current_spatial_edges"] = int(subsets["drmsd_current_spatial_edges"].sum())

    # -- contacts -------------------------------------------------------
    eligible = upper & (separation >= config.contact_sequence_separation)
    contact_pred = d_pred < config.contact_cutoff
    contact_targ = d_targ < config.contact_cutoff
    contact_curr = d_curr < config.contact_cutoff

    tp, fp, fn, precision, recall, f1, jaccard = _contact_scores(
        contact_pred, contact_targ, eligible
    )
    out.update(
        contact_tp=tp, contact_fp=fp, contact_fn=fn,
        contact_target_positives=int((contact_targ & eligible).sum()),
        contact_precision=precision, contact_recall=recall,
        contact_f1=f1, contact_jaccard=jaccard,
    )

    # Transition events. Formed is conditioned on "not in contact now", broken on
    # "in contact now", so both sides of each comparison are drawn from the same
    # population and a false positive means the model invented an event that did
    # not happen -- not that it disagreed about a contact that never changed.
    for name, condition in (
        ("formed", eligible & ~contact_curr),
        ("broken", eligible & contact_curr),
    ):
        want = contact_targ if name == "formed" else ~contact_targ
        got = contact_pred if name == "formed" else ~contact_pred
        e_tp, e_fp, e_fn, e_p, e_r, e_f1, _ = _contact_scores(got, want, condition)
        out.update({
            f"{name}_contact_tp": e_tp,
            f"{name}_contact_fp": e_fp,
            f"{name}_contact_fn": e_fn,
            f"{name}_contact_events_target": int((want & condition).sum()),
            f"{name}_contact_events_predicted": int((got & condition).sum()),
            f"{name}_contact_precision": e_p,
            f"{name}_contact_recall": e_r,
            f"{name}_contact_f1": e_f1,
        })

    # Legacy Stage B definition, recomputed so the extended table can be checked
    # against the published one rather than compared across two definitions.
    legacy = upper & (separation >= config.legacy_contact_sequence_separation)
    out["contact_f1_legacy_sep3"] = _legacy_f1(
        d_pred <= config.legacy_contact_cutoff,
        d_targ <= config.legacy_contact_cutoff,
        legacy,
    )

    # Appendix sensitivity. Reported, never used to choose an arm.
    for cutoff, suffix in zip(
        config.contact_sensitivity_cutoffs, config.contact_key_suffixes()
    ):
        out[f"contact_f1_{suffix}"] = _contact_scores(
            d_pred < cutoff, d_targ < cutoff, eligible
        )[5]

    # -- torsions -------------------------------------------------------
    phi_p, psi_p, phi_ok, psi_ok = backbone_torsions(*pred_backbone, previous, following)
    phi_t, psi_t, _, _ = backbone_torsions(*targ_backbone, previous, following)
    omega_p, omega_ok = backbone_omega(*pred_backbone, following)
    omega_t, _ = backbone_omega(*targ_backbone, following)

    pooled: list[Tensor] = []
    for name, pred_angle, targ_angle, ok in (
        ("phi", phi_p, phi_t, phi_ok),
        ("psi", psi_p, psi_t, psi_ok),
        ("omega", omega_p, omega_t, omega_ok),
    ):
        count = int(ok.sum())
        out[f"n_{name}"] = count
        if count:
            errors = torch.rad2deg(wrap_to_pi(pred_angle - targ_angle)[ok].abs())
            out[f"{name}_mae_deg"] = float(errors.mean())
            pooled.append(errors)
    if pooled:
        joined = torch.cat(pooled)
        out["backbone_torsion_mae_deg"] = float(joined.mean())
        out["n_valid_torsions"] = int(joined.numel())

    # -- physical validity ----------------------------------------------
    has_next = following >= 0
    next_safe = following.clamp(min=0)
    n_p, ca_p, c_p = pred_backbone
    n_t, ca_t, c_t = targ_backbone

    if bool(has_next.any()):
        sel = has_next
        nxt = next_safe[sel]
        bond_p = (n_p[nxt] - c_p[sel]).norm(dim=-1).to(torch.float64)
        bond_t = (n_t[nxt] - c_t[sel]).norm(dim=-1).to(torch.float64)
        delta = bond_p - bond_t
        out["n_bond_c_n_next"] = int(sel.sum())
        out["bond_length_rmse"] = float(delta.pow(2).mean().sqrt())
        out["bond_length_mae"] = float(delta.abs().mean())

        angles = []
        for apex_p, a_p, b_p, apex_t, a_t, b_t in (
            (c_p[sel], ca_p[sel], n_p[nxt], c_t[sel], ca_t[sel], n_t[nxt]),
            (n_p[nxt], c_p[sel], ca_p[nxt], n_t[nxt], c_t[sel], ca_t[nxt]),
        ):
            angles.append(
                (_angle_at(apex_p, a_p, b_p) - _angle_at(apex_t, a_t, b_t)).abs()
            )
        joined_angles = torch.cat(angles)
        out["n_bond_angles"] = int(joined_angles.numel())
        out["bond_angle_mae_deg"] = float(torch.rad2deg(joined_angles).mean())
        out["bond_angle_ca_c_n_mae_deg"] = float(
            torch.rad2deg(angles[0]).mean()
        )
        out["bond_angle_c_n_ca_mae_deg"] = float(
            torch.rad2deg(angles[1]).mean()
        )

        ca_p_d = (ca_p[nxt] - ca_p[sel]).norm(dim=-1).to(torch.float64)
        ca_t_d = (ca_t[nxt] - ca_t[sel]).norm(dim=-1).to(torch.float64)
        out["n_ca_neighbor"] = int(sel.sum())
        out["ca_neighbor_distance_mae"] = float((ca_p_d - ca_t_d).abs().mean())

    clash_eligible = upper & (separation >= config.clash_sequence_separation)
    total = int(clash_eligible.sum())
    out["n_clash_pairs"] = total
    if total:
        out["clash_count"] = int(
            ((d_pred < config.clash_min_ca_distance) & clash_eligible).sum()
        )
        out["clash_count_target"] = int(
            ((d_targ < config.clash_min_ca_distance) & clash_eligible).sum()
        )
        out["clash_rate"] = out["clash_count"] / total
        out["clash_rate_target"] = out["clash_count_target"] / total

    return out


def _angle_at(apex: Tensor, a: Tensor, b: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Angle ``a-apex-b`` in radians, ``[N]``, via ``atan2``.

    ``atan2(|u x v|, u.v)`` rather than ``arccos(u.v / |u||v|)`` for the reason
    this project uses everywhere: ``arccos`` loses precision near 0 and pi, and a
    backbone angle sits at neither extreme but its *error* is near zero, which is
    exactly where the derivative blows up.
    """
    u = (a - apex).to(torch.float64)
    v = (b - apex).to(torch.float64)
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=eps)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.atan2(torch.linalg.cross(u, v, dim=-1).norm(dim=-1), (u * v).sum(-1))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordContext:
    """Provenance stamped onto every row, so a record is self-describing.

    A metrics file that has to be joined against a run directory to be
    interpreted is a metrics file that will one day be interpreted against the
    wrong one.
    """

    arm: str
    canonical_arm: str
    oracle: bool
    seed: int
    manifest_hash: str
    phase1_checkpoint_hash: str
    transition_checkpoint_hash: str
    config_hash: str
    git_commit: str
    git_dirty: bool
    source_diff_hash: str
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {
            "arm": self.arm,
            "canonical_arm": self.canonical_arm,
            "oracle": self.oracle,
            "seed": self.seed,
            "manifest_hash": self.manifest_hash,
            "phase1_checkpoint_hash": self.phase1_checkpoint_hash,
            "transition_checkpoint_hash": self.transition_checkpoint_hash,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "source_diff_hash": self.source_diff_hash,
        }
        row.update(self.extra)
        return row


@torch.no_grad()
def extended_metric_records(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    *,
    pairs: Sequence,
    context: RecordContext,
    config: ExtendedMetricConfig = ExtendedMetricConfig(),
    identity: bool = False,
) -> list[dict]:
    """One row per graph, keyed by ``pair_id``, with the full extended schema.

    Args:
        pairs: the batch's :class:`~force_md.data.adapters.lag_pairs.LagPair`
            rows, one per graph, in graph order. They carry the identity of the
            sample; they are passed in rather than read from the batch so this
            module keeps no dependency on the dataset layer, exactly as
            ``metrics.metric_records`` does.
        identity: label the rows as the "nothing moves" baseline. The caller
            supplies the identity prediction; this only sets the ``arm`` fields,
            so the baseline cannot silently be computed with a different
            definition than the arms it is quoted against.

    The **future** target is read here, to score. It reaches no model: this
    function has no model argument, and ``TransitionTarget`` is not a type any
    conditioner accepts.
    """
    if len(pairs) != target.num_graphs:
        raise ValueError(
            f"{len(pairs)} LagPair rows for {target.num_graphs} graphs; a record "
            "whose metadata came from a different graph is worse than no record"
        )

    pred_ca, pred_rotation = apply_prediction(prediction, target)
    targ_ca = target.future_ca_aligned
    targ_rotation = target.future_frames_aligned.rotation
    pred_backbone = reconstruct_backbone(prediction, target)
    targ_backbone = reconstruct_backbone(target_as_prediction(target), target)
    translation_error = prediction.translation_local - target.translation_local
    current_ca = target.current_ca

    base = context.as_row()
    records: list[dict] = []
    for graph, pair in enumerate(pairs):
        in_graph = target.residue_batch_index == graph
        select = in_graph & target.valid
        index = select.nonzero(as_tuple=True)[0]

        # Sequence neighbours are global row indices; remap into this graph's
        # valid subset, dropping links whose partner was masked out. Same
        # remapping as metrics.per_graph_transition_metrics, so a torsion that is
        # excluded there is excluded here.
        remap = torch.full_like(target.residue_batch_index, -1)
        remap[index] = torch.arange(index.numel(), device=index.device)
        local_previous = torch.where(
            target.previous[index] >= 0,
            remap[target.previous[index].clamp(min=0)],
            torch.full_like(index, -1),
        )
        local_following = torch.where(
            target.following[index] >= 0,
            remap[target.following[index].clamp(min=0)],
            torch.full_like(index, -1),
        )

        graph_rows = in_graph.nonzero(as_tuple=True)[0]
        spatial = _current_spatial_pair_mask(
            current_ca[graph_rows],
            select[graph_rows],
            k=config.spatial_knn,
            cutoff=config.spatial_cutoff,
        )

        row = extended_graph_metrics(
            pred_ca=pred_ca[index],
            targ_ca=targ_ca[index],
            curr_ca=current_ca[index],
            pred_rotation=pred_rotation[index],
            targ_rotation=targ_rotation[index],
            pred_backbone=tuple(t[index] for t in pred_backbone),
            targ_backbone=tuple(t[index] for t in targ_backbone),
            translation_error=translation_error[index],
            resid=target.resid_original[index],
            chain=target.chain_index[index],
            previous=local_previous,
            following=local_following,
            spatial_mask=spatial,
            config=config,
        )
        row["n_residues"] = int(in_graph.sum())
        row["length_bin"] = length_bin(
            row["n_valid_residues"], config.length_bin_edges
        )

        identity_fields = (
            {"arm": "identity_baseline", "canonical_arm": "identity_baseline",
             "oracle": False}
            if identity else {}
        )
        records.append({
            **base,
            **identity_fields,
            "pair_id": pair.pair_id,
            "domain_id": pair.domain,
            "temperature": str(pair.temperature),
            "replica_id": str(pair.replica),
            "current_frame_index": int(pair.current_frame),
            "future_frame_index": int(pair.future_frame),
            "lag_ps": float(pair.lag_ps),
            "lag_ns": float(pair.lag_ps) / 1000.0,
            **row,
        })
    return records


def torsion_histogram(
    values: Sequence[float], *, bins: int
) -> list[int]:
    """Counts over ``bins`` equal bins spanning ``[-180, 180)`` degrees.

    Binning is fixed by the caller's config and not adapted to the data, so two
    histograms are always comparable. Used only by the optional distribution
    diagnostic; it is not a primary metric of a deterministic model.
    """
    counts = [0] * bins
    width = 360.0 / bins
    for value in values:
        if value != value:
            continue
        index = int((value + 180.0) // width)
        counts[min(max(index, 0), bins - 1)] += 1
    return counts


def jensen_shannon(p: Sequence[int], q: Sequence[int]) -> float:
    """JSD in bits between two histograms. NaN when either side is empty."""
    import math

    total_p, total_q = sum(p), sum(q)
    if total_p == 0 or total_q == 0:
        return _NAN
    out = 0.0
    for a, b in zip(p, q):
        pa, qb = a / total_p, b / total_q
        m = 0.5 * (pa + qb)
        if pa > 0:
            out += 0.5 * pa * math.log2(pa / m)
        if qb > 0:
            out += 0.5 * qb * math.log2(qb / m)
    return out
