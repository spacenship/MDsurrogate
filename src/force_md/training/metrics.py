"""Evaluation metrics.

Every vector metric is reported next to a **zero baseline** -- the error you get
by predicting no force at all. Force RMSE alone is easy to misread: mdCATH's
per-atom force magnitudes are ~25-50 kcal/mol/A, so an RMSE of 40 means the model
has learned nothing, and only the comparison makes that obvious.

Angular error is reported separately from magnitude because they fail
differently: a model can get the direction of the local drift right while badly
mis-scaling it, and that is a far more useful model than the reverse.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["vector_metrics", "merge_metrics"]


def vector_metrics(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    *,
    prefix: str = "",
    min_norm: float = 1e-6,
) -> dict[str, float]:
    """RMSE, MAE, cosine, angular error and the zero-prediction baseline.

    Args:
        prediction / target: ``[N, 3]``.
        mask: ``[N]`` bool; None means all nodes count.
        prefix: prepended to every key.
        min_norm: targets shorter than this are excluded from the angular
            statistics, where their direction is meaningless.

    Returns:
        ``{f"{prefix}rmse": ..., f"{prefix}angular_error_deg": ..., ...}``.
        Empty (all-masked) input yields NaN rather than a misleading zero.
    """
    if mask is None:
        mask = torch.ones(prediction.shape[0], dtype=torch.bool, device=prediction.device)
    pred, targ = prediction[mask], target[mask]
    n = pred.shape[0]
    if n == 0:
        keys = ["rmse", "mae", "rmse_zero", "relative_rmse", "cosine",
                "angular_error_deg", "count"]
        return {f"{prefix}{k}": float("nan") for k in keys}

    error = pred - targ
    rmse = float(error.pow(2).sum(-1).mean().sqrt())
    mae = float(error.norm(dim=-1).mean())
    rmse_zero = float(targ.pow(2).sum(-1).mean().sqrt())

    long_enough = targ.norm(dim=-1) > min_norm
    if bool(long_enough.any()):
        p, t = pred[long_enough], targ[long_enough]
        cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).clamp(-1.0, 1.0)
        cosine = float(cos.mean())
        # atan2(|a x b|, a.b), not arccos(cos). arccos is ill-conditioned exactly
        # where a good model lives: at cos = 1 - eps the angle is ~sqrt(2 eps), so
        # float32 rounding alone turns a perfect prediction into ~0.03 degrees.
        # The atan2 form is well conditioned over the whole range.
        cross = torch.linalg.cross(p, t, dim=-1).norm(dim=-1)
        dot = (p * t).sum(-1)
        angular = float(torch.rad2deg(torch.atan2(cross, dot)).mean())
    else:
        cosine = angular = float("nan")

    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}rmse_zero": rmse_zero,
        f"{prefix}relative_rmse": rmse / rmse_zero if rmse_zero > 0 else float("nan"),
        f"{prefix}cosine": cosine,
        f"{prefix}angular_error_deg": angular,
        f"{prefix}count": float(n),
    }


def merge_metrics(accumulated: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-batch metric dicts, weighting by ``*count``.

    A plain mean over batches would over-weight small proteins, which is exactly
    the population mdCATH has most of.
    """
    if not accumulated:
        return {}
    keys = [k for k in accumulated[0] if not k.endswith("count")]
    out: dict[str, float] = {}
    for key in keys:
        prefix = key.rsplit("_", 1)[0] if "_" in key else ""
        count_key = None
        for candidate in accumulated[0]:
            if candidate.endswith("count") and key.startswith(candidate[: -len("count")]):
                count_key = candidate
                break
        total_w = 0.0
        total = 0.0
        for row in accumulated:
            value = row.get(key)
            weight = row.get(count_key, 1.0) if count_key else 1.0
            if value is None or value != value or weight != weight:  # NaN check
                continue
            total += value * weight
            total_w += weight
        out[key] = total / total_w if total_w > 0 else float("nan")
    for candidate in accumulated[0]:
        if candidate.endswith("count"):
            out[candidate] = sum(r.get(candidate, 0.0) for r in accumulated)
    return out
