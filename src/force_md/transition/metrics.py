"""Geometry metrics for a predicted transition.

**No single number decides anything here.** A model can hold every Ca in roughly
the right place while rotating the residue frames at random, or keep the frames
right and shear the pair distances, or produce a structure that scores well on
both and contains atoms inside each other. So seven views are reported together:
placement (Ca RMSD, translation), orientation (frame geodesic angle), internal
shape (pair distances, contacts), physicality (clash rate, backbone torsions).

**Everything is quoted against the identity baseline.** "Nothing moves" -- zero
translation, identity rotation -- is a strong predictor at these lags: measured on
mdCATH, 1 ns Ca RMSD after alignment is ~1.1-2.9 A at 320 K and 4 ns only 15-35%
more. A model that reports 1.2 A has beaten nothing if the identity baseline is
1.25 A. This mirrors Phase 1, where every force RMSE is quoted beside the
zero-prediction baseline for exactly this reason, and it is what
``*_identity`` and ``*_relative`` carry.

**Aggregation is reported twice.** ``micro`` weights every residue equally, so
large domains dominate; ``domain_macro`` weights every domain equally. They
answer different questions and the Phase 1.5 report must show both, split by lag,
because a result that only survives one of them is not a result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor

from ..geometry.alignment import kabsch_rotation
from ..geometry.so3 import rotation_geodesic_angle
from ..geometry.torsions import backbone_torsions, wrap_to_pi
from .targets import (
    TransitionPrediction,
    TransitionTarget,
    apply_prediction,
    identity_prediction,
    reconstruct_backbone,
)

__all__ = [
    "MetricConfig",
    "per_graph_transition_metrics",
    "transition_metrics",
    "metric_records",
    "aggregate_metric_records",
]


@dataclass(frozen=True)
class MetricConfig:
    """Thresholds, all of them named and sourced.

    Args:
        contact_cutoff: Ca-Ca distance defining a contact, in Angstrom. 8.0 A is
            the usual Ca contact definition in structure prediction.
        contact_sequence_separation: minimum ``|resid_i - resid_j|`` within a
            chain for a pair to count. Near-diagonal pairs are fixed by the chain
            itself and would inflate every contact score.
        pair_distance_cutoff: if set, pair-distance MAE is restricted to pairs
            within this distance **in the target**. None uses all pairs, which
            weights the measure towards the large-separation pairs a domain has
            most of.
        clash_min_ca_distance: two non-neighbouring Ca closer than this count as
            a clash. 3.6 A is *below* the measured Ca-Ca virtual bond length in
            this dataset (audited 3.67-4.01 A over real frames, see
            ``docs/phase1_hierarchy_and_contracts.md``), so a violation means two
            residues are closer than two *bonded* ones -- a real geometric
            failure, not a tight packing. No van der Waals table is invented here.
        clash_sequence_separation: pairs closer than this along the chain are
            excluded, since their proximity is bonded, not a clash.
        compute_torsions: phi/psi MAE on the reconstructed backbone.
    """

    contact_cutoff: float = 8.0
    contact_sequence_separation: int = 3
    pair_distance_cutoff: Optional[float] = None
    clash_min_ca_distance: float = 3.6
    clash_sequence_separation: int = 2
    compute_torsions: bool = True


_NAN = float("nan")

#: Keys produced per graph. Fixed so a record row has a stable schema even when a
#: graph has too few valid residues to define some of them.
METRIC_KEYS = (
    "ca_rmsd",
    "ca_rmsd_aligned",
    "translation_rmse",
    "translation_mae",
    "rotation_geodesic_deg",
    "pair_distance_mae",
    "contact_f1",
    "contact_precision",
    "contact_recall",
    "clash_rate",
    "clash_rate_target",
    "phi_mae_deg",
    "psi_mae_deg",
)


def _pairwise_distances(points: Tensor) -> Tensor:
    """``[n, n]`` Euclidean distances, in the numerically safe mode.

    ``compute_mode="donot_use_mm_for_euclid_dist"`` for the same reason the graph
    builders use it: the default expands ``||a||^2 + ||b||^2 - 2a.b`` above 25
    rows, which is catastrophically unstable when coordinates are large compared
    with the distances being measured.
    """
    return torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")


def _eligible_pairs(
    resid: Tensor, chain: Tensor, separation: int
) -> Tensor:
    """``[n, n]`` bool: upper-triangular pairs far enough apart along the chain.

    Separation is measured with the **source residue numbering** within a chain;
    residues on different chains are always eligible. Using row indices instead
    would silently treat a numbering gap as adjacency.
    """
    n = resid.shape[0]
    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=resid.device), diagonal=1
    )
    same_chain = chain[:, None] == chain[None, :]
    gap = (resid[:, None] - resid[None, :]).abs()
    far_enough = (~same_chain) | (gap >= separation)
    return upper & far_enough


def _graph_metrics(
    pred_ca: Tensor,
    pred_rotation: Tensor,
    targ_ca: Tensor,
    targ_rotation: Tensor,
    pred_backbone: tuple[Tensor, Tensor, Tensor],
    targ_backbone: tuple[Tensor, Tensor, Tensor],
    translation_error: Tensor,
    resid: Tensor,
    chain: Tensor,
    previous: Tensor,
    following: Tensor,
    config: MetricConfig,
) -> dict[str, float]:
    """Every metric for one graph's valid residues."""
    out = {key: _NAN for key in METRIC_KEYS}
    n = pred_ca.shape[0]
    if n == 0:
        return out

    # -- placement ------------------------------------------------------
    out["ca_rmsd"] = float((pred_ca - targ_ca).pow(2).sum(-1).mean().sqrt())
    out["translation_rmse"] = float(translation_error.pow(2).sum(-1).mean().sqrt())
    out["translation_mae"] = float(translation_error.norm(dim=-1).mean())

    if n >= 3:
        zeros = torch.zeros(n, dtype=torch.int64, device=pred_ca.device)
        alignment = kabsch_rotation(pred_ca, targ_ca, zeros, 1)
        realigned = alignment.apply(pred_ca, zeros)
        out["ca_rmsd_aligned"] = float(
            (realigned - targ_ca).pow(2).sum(-1).mean().sqrt()
        )

    # -- orientation ----------------------------------------------------
    angle = rotation_geodesic_angle(pred_rotation, targ_rotation)
    out["rotation_geodesic_deg"] = float(torch.rad2deg(angle).mean())

    # -- internal shape -------------------------------------------------
    if n >= 2:
        d_pred = _pairwise_distances(pred_ca)
        d_targ = _pairwise_distances(targ_ca)
        eligible = _eligible_pairs(resid, chain, config.contact_sequence_separation)

        pair_mask = torch.triu(
            torch.ones_like(eligible), diagonal=1
        )
        if config.pair_distance_cutoff is not None:
            pair_mask = pair_mask & (d_targ <= config.pair_distance_cutoff)
        if bool(pair_mask.any()):
            out["pair_distance_mae"] = float(
                (d_pred[pair_mask] - d_targ[pair_mask]).abs().mean()
            )

        if bool(eligible.any()):
            contact_pred = (d_pred <= config.contact_cutoff) & eligible
            contact_targ = (d_targ <= config.contact_cutoff) & eligible
            true_positive = float((contact_pred & contact_targ).sum())
            n_pred = float(contact_pred.sum())
            n_targ = float(contact_targ.sum())
            precision = true_positive / n_pred if n_pred > 0 else _NAN
            recall = true_positive / n_targ if n_targ > 0 else _NAN
            out["contact_precision"] = precision
            out["contact_recall"] = recall
            if precision == precision and recall == recall and (precision + recall) > 0:
                out["contact_f1"] = 2 * precision * recall / (precision + recall)
            elif precision == precision and recall == recall:
                out["contact_f1"] = 0.0

        clash_eligible = _eligible_pairs(resid, chain, config.clash_sequence_separation)
        if bool(clash_eligible.any()):
            total = float(clash_eligible.sum())
            out["clash_rate"] = float(
                ((d_pred < config.clash_min_ca_distance) & clash_eligible).sum()
            ) / total
            out["clash_rate_target"] = float(
                ((d_targ < config.clash_min_ca_distance) & clash_eligible).sum()
            ) / total

    # -- physicality: backbone torsions ---------------------------------
    if config.compute_torsions and n >= 2:
        phi_p, psi_p, phi_ok, psi_ok = backbone_torsions(*pred_backbone, previous, following)
        phi_t, psi_t, _, _ = backbone_torsions(*targ_backbone, previous, following)
        if bool(phi_ok.any()):
            out["phi_mae_deg"] = float(
                torch.rad2deg(wrap_to_pi(phi_p - phi_t)[phi_ok].abs()).mean()
            )
        if bool(psi_ok.any()):
            out["psi_mae_deg"] = float(
                torch.rad2deg(wrap_to_pi(psi_p - psi_t)[psi_ok].abs()).mean()
            )
    return out


@torch.no_grad()
def per_graph_transition_metrics(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    *,
    config: MetricConfig = MetricConfig(),
) -> tuple[list[dict[str, float]], list[int]]:
    """Metrics for each graph separately.

    Returns:
        ``(rows, residue_counts)`` -- one dict per graph, keyed by
        :data:`METRIC_KEYS`, with NaN where a graph had too few valid residues to
        define a quantity. NaN rather than 0: a zero clash rate on a protein with
        no valid residues would read as a perfect score.
    """
    pred_ca, pred_rotation = apply_prediction(prediction, target)
    targ_ca = target.future_ca_aligned
    targ_rotation = target.future_frames_aligned.rotation
    pred_backbone_all = reconstruct_backbone(prediction, target)
    from .targets import target_as_prediction  # noqa: PLC0415 - avoids a cycle at import

    targ_backbone_all = reconstruct_backbone(target_as_prediction(target), target)
    translation_error_all = prediction.translation_local - target.translation_local

    rows: list[dict[str, float]] = []
    counts: list[int] = []
    for graph in range(target.num_graphs):
        select = (target.residue_batch_index == graph) & target.valid
        index = select.nonzero(as_tuple=True)[0]
        counts.append(int(index.numel()))
        # Sequence neighbours are global row indices; remap them into this
        # graph's local numbering, dropping links whose partner is masked out.
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
        rows.append(
            _graph_metrics(
                pred_ca[index], pred_rotation[index],
                targ_ca[index], targ_rotation[index],
                tuple(t[index] for t in pred_backbone_all),
                tuple(t[index] for t in targ_backbone_all),
                translation_error_all[index],
                target.resid_original[index], target.chain_index[index],
                local_previous, local_following,
                config,
            )
        )
    return rows, counts


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = 0.0
    total_weight = 0.0
    for value, weight in zip(values, weights):
        if value != value or weight != weight or weight <= 0:  # NaN-safe
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight > 0 else _NAN


@torch.no_grad()
def transition_metrics(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    *,
    config: MetricConfig = MetricConfig(),
    with_baseline: bool = True,
) -> dict[str, float]:
    """Residue-weighted metrics over the batch, with the identity baseline.

    Returns a flat dict: each key from :data:`METRIC_KEYS`, plus ``<key>_identity``
    and ``<key>_relative = value / identity`` when ``with_baseline``. A relative
    value at or above 1.0 means the model has not improved on predicting that
    nothing moved.
    """
    rows, counts = per_graph_transition_metrics(prediction, target, config=config)
    out = {
        key: _weighted_mean([row[key] for row in rows], counts) for key in METRIC_KEYS
    }
    out["residue_count"] = float(sum(counts))
    out["graph_count"] = float(len(counts))
    if not with_baseline:
        return out

    base_rows, _ = per_graph_transition_metrics(
        identity_prediction(target), target, config=config
    )
    for key in METRIC_KEYS:
        baseline = _weighted_mean([row[key] for row in base_rows], counts)
        out[f"{key}_identity"] = baseline
        if baseline == baseline and baseline > 0:
            value = out[key]
            out[f"{key}_relative"] = value / baseline if value == value else _NAN
        else:
            out[f"{key}_relative"] = _NAN
    return out


@torch.no_grad()
def metric_records(
    prediction: TransitionPrediction,
    target: TransitionTarget,
    *,
    domains: Sequence[str],
    lag_ps: Sequence[float],
    split: str = "val",
    config: MetricConfig = MetricConfig(),
    include_identity: bool = True,
) -> list[dict]:
    """One tidy row per graph, ready for a CSV.

    Args:
        domains / lag_ps: per-graph metadata, length ``target.num_graphs``. These
            come from the manifest's :class:`~force_md.data.adapters.lag_pairs.LagPair`
            rows; they are passed in rather than read from the batch so this
            module does not depend on the dataset layer.
    """
    if len(domains) != target.num_graphs or len(lag_ps) != target.num_graphs:
        raise ValueError(
            f"metadata length mismatch: {len(domains)} domains, {len(lag_ps)} lags, "
            f"{target.num_graphs} graphs"
        )
    rows, counts = per_graph_transition_metrics(prediction, target, config=config)
    base_rows = (
        per_graph_transition_metrics(identity_prediction(target), target, config=config)[0]
        if include_identity
        else [{} for _ in rows]
    )
    records = []
    for graph, (row, base, count) in enumerate(zip(rows, base_rows, counts)):
        record = {
            "split": split,
            "domain": domains[graph],
            "lag_ps": float(lag_ps[graph]),
            "lag_ns": float(lag_ps[graph]) / 1000.0,
            "residue_count": count,
            **row,
        }
        record.update({f"{k}_identity": v for k, v in base.items()})
        records.append(record)
    return records


def aggregate_metric_records(
    records: Iterable[dict],
    *,
    group_keys: Sequence[str] = ("split", "lag_ns"),
) -> list[dict]:
    """Micro and domain-macro averages per group.

    ``micro`` weights each graph by its valid-residue count, so a 250-residue
    domain counts 5x a 50-residue one. ``domain_macro`` averages within each
    domain first and then over domains, so every protein counts once. They can
    disagree, and when they do it is information: a gain that appears only in
    ``micro`` is a gain on large proteins, not a general one.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(k) for k in group_keys)].append(record)

    metric_names = sorted(
        {
            k
            for rows in grouped.values()
            for row in rows
            for k in row
            if isinstance(row[k], float) and k not in ("lag_ps", "lag_ns")
        }
    )

    out = []
    for key, rows in sorted(grouped.items(), key=lambda kv: [str(x) for x in kv[0]]):
        summary = dict(zip(group_keys, key))
        summary["graph_count"] = len(rows)
        summary["domain_count"] = len({row["domain"] for row in rows})
        summary["residue_count"] = sum(int(row["residue_count"]) for row in rows)
        for name in metric_names:
            values = [row.get(name, _NAN) for row in rows]
            weights = [float(row["residue_count"]) for row in rows]
            summary[f"{name}_micro"] = _weighted_mean(values, weights)

            per_domain: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                per_domain[row["domain"]].append(
                    (row.get(name, _NAN), float(row["residue_count"]))
                )
            domain_means = [
                _weighted_mean([v for v, _ in items], [w for _, w in items])
                for items in per_domain.values()
            ]
            finite = [v for v in domain_means if v == v]
            summary[f"{name}_domain_macro"] = (
                sum(finite) / len(finite) if finite else _NAN
            )
        out.append(summary)
    return out
