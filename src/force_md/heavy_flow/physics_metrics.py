"""Force metrics, controls, and target audits for Stage 3.

Torque is measured about the current represented-heavy-atom centroid of each
residue. This origin is fixed by the current geometry for a metric run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor

from .force_losses import ForceNormalizer, heteroscedastic_force_nll
from .types import HeavyFlowSample

__all__ = [
    "ForceTargetAudit", "compute_force_metrics", "force_metrics", "residue_net_force_torque",
    "zero_force_baseline", "mean_force_baseline", "unconditional_mean_variance_baseline",
    "shuffled_force_labels", "audit_force_targets", "target_audit",
]


def _mask_for(force: Tensor, atom_mask: Optional[Tensor], force_mask: Optional[Tensor]) -> Tensor:
    if force.ndim != 3 or force.shape[-1] != 3:
        raise ValueError("force tensors must have shape [B,N,3]")
    mask = atom_mask if atom_mask is not None else torch.ones(force.shape[:2], dtype=torch.bool, device=force.device)
    if force_mask is not None:
        mask = mask & torch.as_tensor(force_mask, dtype=torch.bool, device=force.device)
    if mask.shape != force.shape[:2]:
        raise ValueError("atom/force mask must have shape [B,N]")
    return mask


def _float(value: Tensor) -> float:
    return float(value.detach().float().cpu())


def residue_net_force_torque(
    force: Tensor,
    positions: Tensor,
    atom_to_residue: Tensor,
    *,
    atom_mask: Optional[Tensor] = None,
    residue_mask: Optional[Tensor] = None,
    residue_centers: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return residue net force, torque, and the fixed metric origin."""
    if positions.shape != force.shape or positions.shape[-1] != 3:
        raise ValueError("positions and force must both have shape [B,N,3]")
    if atom_to_residue.shape != force.shape[:2]:
        raise ValueError("atom_to_residue must have shape [B,N]")
    b, n = force.shape[:2]
    l = int(atom_to_residue.max().item()) + 1 if atom_to_residue.numel() else 0
    if residue_mask is not None:
        if residue_mask.shape[0] != b:
            raise ValueError("residue_mask batch dimension disagrees")
        l = max(l, residue_mask.shape[1])
    active = _mask_for(force, atom_mask, None)
    if residue_mask is None:
        residue_mask = torch.ones((b, l), dtype=torch.bool, device=force.device)
    if residue_centers is None:
        centers = positions.new_zeros((b, l, 3))
        for batch in range(b):
            for residue in range(l):
                selected = active[batch] & (atom_to_residue[batch] == residue)
                if bool(selected.any()):
                    centers[batch, residue] = positions[batch, selected].mean(dim=0)
    else:
        centers = residue_centers
        if centers.shape != (b, l, 3):
            raise ValueError("residue_centers must have shape [B,L,3]")
    net = force.new_zeros((b, l, 3))
    torque = force.new_zeros((b, l, 3))
    for batch in range(b):
        for residue in range(l):
            selected = active[batch] & (atom_to_residue[batch] == residue) & residue_mask[batch, residue]
            if bool(selected.any()):
                net[batch, residue] = force[batch, selected].sum(dim=0)
                arm = positions[batch, selected] - centers[batch, residue]
                torque[batch, residue] = torch.cross(arm, force[batch, selected], dim=-1).sum(dim=0)
    return net, torque, centers


def _safe_corr(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    p, t = pred[mask].reshape(-1), target[mask].reshape(-1)
    if p.numel() < 2:
        return p.sum() * 0.0
    p, t = p - p.mean(), t - t.mean()
    denom = (p.square().sum() * t.square().sum()).sqrt()
    return torch.where(denom > 1e-12, (p * t).sum() / denom, p.sum() * 0.0)


def _normalized_fields(mean: Tensor, logvar: Optional[Tensor], target: Tensor, normalizer: Optional[ForceNormalizer]) -> tuple[Tensor, Optional[Tensor], Tensor]:
    mean, target = mean.float(), target.float()
    if normalizer is None:
        return mean, None if logvar is None else logvar.float(), target
    scale = mean.new_tensor(normalizer.scale).float()
    normalized_logvar = None if logvar is None else logvar.float() - 2.0 * torch.log(scale)
    return mean / scale, normalized_logvar, target / scale


def compute_force_metrics(
    force_mean: Tensor,
    target_force: Tensor,
    *,
    force_logvar: Optional[Tensor] = None,
    atom_mask: Optional[Tensor] = None,
    force_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    atom_to_residue: Optional[Tensor] = None,
    residue_mask: Optional[Tensor] = None,
    domains: Optional[Sequence[str]] = None,
    normalizer: Optional[ForceNormalizer] = None,
) -> dict[str, Any]:
    """Compute atom-force, proper-score, calibration, and torque metrics."""
    if force_mean.shape != target_force.shape:
        raise ValueError("force_mean and target_force must have equal shapes")
    mask = _mask_for(target_force, atom_mask, force_mask)
    mean, logvar, target = _normalized_fields(force_mean, force_logvar, target_force, normalizer)
    error = mean - target
    active = mask[..., None].expand_as(error)
    count = int(mask.sum())
    sse = error[active].square().sum()
    centered = target[active] - (target[active].mean() if count else target.sum() * 0.0)
    total = centered.square().sum().clamp_min(1e-12)
    cosine_denominator = (mean[active].square().sum() * target[active].square().sum()).sqrt()
    cosine = (mean[active] * target[active]).sum() / cosine_denominator.clamp_min(1e-12) if count else mean.sum() * 0.0
    result: dict[str, Any] = {
        "rmse": _float((sse / max(count * 3, 1)).sqrt()),
        "mae": _float(error[active].abs().mean() if count else error.sum() * 0.0),
        "correlation": _float(_safe_corr(mean, target, mask)),
        "r2": _float(1.0 - sse / total),
        "cosine": _float(cosine),
        "num_atoms": count,
        "num_proteins": int(mask.any(dim=-1).sum()),
    }
    if logvar is not None:
        result["nll"] = _float(heteroscedastic_force_nll(force_mean, force_logvar, target_force, mask, normalizer=normalizer))
        safe_logvar = logvar.float().clamp(-10.0, 10.0)
        sigma = torch.exp(0.5 * safe_logvar)
        if sigma.shape[-1] == 1:
            sigma = sigma.expand_as(error)
        standardized = error / sigma
        active_standardized = standardized[active]
        result["standardized_residual_mean"] = _float(active_standardized.abs().mean() if active_standardized.numel() else standardized.sum() * 0.0)
        result["standardized_residual_std"] = _float(active_standardized.std(unbiased=False) if active_standardized.numel() else standardized.sum() * 0.0)
        result["coverage_68"] = _float((active_standardized.abs() <= 1.0).float().mean() if active_standardized.numel() else standardized.sum() * 0.0)
        result["coverage_95"] = _float((active_standardized.abs() <= 1.96).float().mean() if active_standardized.numel() else standardized.sum() * 0.0)
        empirical = active.square().float().mean() if count else error.sum() * 0.0
        predicted = sigma[active].square().float().mean() if count else sigma.sum() * 0.0
        result["variance_calibration_ratio"] = _float(empirical / predicted.clamp_min(1e-12))
    if positions is not None or atom_to_residue is not None:
        if positions is None or atom_to_residue is None:
            raise ValueError("positions and atom_to_residue must be supplied together")
        pred_net, pred_torque, _ = residue_net_force_torque(mean, positions, atom_to_residue, atom_mask=mask, residue_mask=residue_mask)
        true_net, true_torque, _ = residue_net_force_torque(target, positions, atom_to_residue, atom_mask=mask, residue_mask=residue_mask)
        residue_active = torch.ones(pred_net.shape[:2], dtype=torch.bool, device=pred_net.device) if residue_mask is None else residue_mask.to(torch.bool)
        result["residue_net_force_rmse"] = _float(((pred_net - true_net)[residue_active].square().mean()).sqrt() if residue_active.any() else pred_net.sum() * 0.0)
        result["residue_torque_rmse"] = _float(((pred_torque - true_torque)[residue_active].square().mean()).sqrt() if residue_active.any() else pred_torque.sum() * 0.0)
    if domains is not None:
        if len(domains) != force_mean.shape[0]:
            raise ValueError("domains must contain one entry per batch item")
        result["domain_breakdown"] = {
            str(domain): compute_force_metrics(
                force_mean[i:i + 1], target_force[i:i + 1],
                force_logvar=None if force_logvar is None else force_logvar[i:i + 1],
                atom_mask=None if atom_mask is None else atom_mask[i:i + 1],
                force_mask=None if force_mask is None else force_mask[i:i + 1],
                normalizer=normalizer,
            ) for i, domain in enumerate(domains)
        }
    return result


force_metrics = compute_force_metrics


def zero_force_baseline(target_force: Tensor, *, atom_mask: Optional[Tensor] = None, force_mask: Optional[Tensor] = None) -> dict[str, Any]:
    return compute_force_metrics(torch.zeros_like(target_force), target_force, atom_mask=atom_mask, force_mask=force_mask)


def mean_force_baseline(train_force: Tensor, target_force: Tensor, *, train_mask: Optional[Tensor] = None, atom_mask: Optional[Tensor] = None, force_mask: Optional[Tensor] = None) -> dict[str, Any]:
    train_mask = torch.ones(train_force.shape[:2], dtype=torch.bool, device=train_force.device) if train_mask is None else train_mask
    values = train_force[train_mask]
    mean = values.mean(dim=0) if values.numel() else train_force.sum(dim=(0, 1)) * 0.0
    return compute_force_metrics(mean.view(1, 1, 3).expand_as(target_force), target_force, atom_mask=atom_mask, force_mask=force_mask)


def unconditional_mean_variance_baseline(train_force: Tensor, target_force: Tensor, *, train_mask: Optional[Tensor] = None, atom_mask: Optional[Tensor] = None, force_mask: Optional[Tensor] = None, normalizer: Optional[ForceNormalizer] = None) -> dict[str, Any]:
    train_mask = torch.ones(train_force.shape[:2], dtype=torch.bool, device=train_force.device) if train_mask is None else train_mask
    values = train_force[train_mask]
    mean = values.mean(dim=0) if values.numel() else train_force.sum(dim=(0, 1)) * 0.0
    variance = (values - mean).square().mean().clamp_min(1e-6) if values.numel() else train_force.new_tensor(1.0)
    pred_mean = mean.view(1, 1, 3).expand_as(target_force)
    pred_logvar = torch.log(variance).view(1, 1, 1).expand(target_force.shape[0], target_force.shape[1], 1)
    result = compute_force_metrics(pred_mean, target_force, force_logvar=pred_logvar, atom_mask=atom_mask, force_mask=force_mask, normalizer=normalizer)
    result["train_mean"] = [float(x) for x in mean.detach().cpu()]
    result["train_isotropic_variance"] = float(variance.detach().cpu())
    return result


def shuffled_force_labels(target_force: Tensor, *, generator: Optional[torch.Generator] = None) -> Tensor:
    result = target_force.clone()
    for batch in range(target_force.shape[0]):
        result[batch] = target_force[batch, torch.randperm(target_force.shape[1], generator=generator, device=target_force.device)]
    return result


@dataclass
class ForceTargetAudit:
    split_name: str
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    per_domain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# Force target audit: {self.split_name}", "", f"- passed: `{self.passed}`"]
        if self.failures:
            lines += ["", "## Failures", ""] + [f"- {failure}" for failure in self.failures]
        lines += ["", "## Summary", "", "```text", repr(self.summary), "```"]
        return "\n".join(lines) + "\n"


def _quantiles(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    q = torch.quantile(values.float(), torch.tensor([0.5, 0.95, 0.99], device=values.device))
    return {"p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(values.float().max())}


def audit_force_targets(samples: Iterable[HeavyFlowSample], *, split_name: str = "train") -> ForceTargetAudit:
    """Audit provenance, units, ordering, force outliers, and simple baselines."""
    materialized = list(samples)
    audit = ForceTargetAudit(split_name=split_name, passed=True)
    if not materialized:
        audit.passed = False
        audit.failures.append("empty split")
        return audit
    units: set[tuple[str, str, str]] = set()
    frame_ids: list[int] = []
    domains: list[str] = []
    replicas: list[str] = []
    temperatures: list[float] = []
    norms: list[Tensor] = []
    forces: list[Tensor] = []
    masks: list[Tensor] = []
    per_domain: dict[str, list[Tensor]] = {}
    atom_type_norms: dict[int, list[Tensor]] = {}
    for sample in materialized:
        try:
            sample.validate()
        except Exception as exc:
            audit.failures.append(f"schema validation: {exc}")
            continue
        c, t = sample.condition, sample.targets
        cp, tp = c.provenance, t.provenance
        if cp is None or tp is None:
            audit.failures.append("missing condition/target provenance")
            continue
        units.add((cp.length_unit, cp.force_unit, cp.temperature_unit))
        if cp.force_scope != "direct_heavy_atom" or tp.force_scope != "direct_heavy_atom":
            audit.failures.append("force scope is not direct_heavy_atom")
        if cp.domain != tp.domain or cp.frame != tp.frame or cp.replica != tp.replica:
            audit.failures.append("condition/target provenance identity mismatch")
        domain = str(cp.domain[0]) if cp.domain else "unknown"
        domains.append(domain)
        replicas.extend(str(x) for x in cp.replica)
        frame_ids.extend(int(x) for x in cp.frame)
        temperatures.extend(float(x) for x in c.temperature.detach().cpu())
        mask = t.force_mask if t.force_mask is not None else c.atom_mask
        force = t.force_current
        valid = mask & torch.isfinite(force).all(dim=-1)
        norm = force.norm(dim=-1)[valid]
        norms.append(norm)
        forces.append(force)
        masks.append(mask)
        per_domain.setdefault(domain, []).append(norm)
        for atom_type in torch.unique(c.atom_type[mask]):
            atom_type_norms.setdefault(int(atom_type), []).append(norm[c.atom_type[mask] == atom_type])
    if len(units) > 1:
        audit.failures.append(f"inconsistent units: {sorted(units)}")
    if not norms:
        audit.passed = False
        audit.failures.append("no valid force atoms")
        return audit
    flat_norm = torch.cat(norms)
    force = torch.cat(forces, dim=0)
    mask = torch.cat(masks, dim=0)
    zero = zero_force_baseline(force, force_mask=mask)
    train_mean = force[mask].mean(dim=0)
    mean_stats = compute_force_metrics(train_mean.view(1, 1, 3).expand_as(force), force, force_mask=mask)
    audit.summary = {
        "samples": len(materialized), "domains": sorted(set(domains)), "replicas": sorted(set(replicas)),
        "frame_min": min(frame_ids) if frame_ids else None, "frame_max": max(frame_ids) if frame_ids else None,
        "temperature_min": min(temperatures) if temperatures else None, "temperature_max": max(temperatures) if temperatures else None,
        "valid_atoms": int(mask.sum()), "force_norm": _quantiles(flat_norm),
        "zero_force_rmse": zero["rmse"], "unconditional_mean_rmse": mean_stats["rmse"],
        "frame_alignment": "same-frame direct atom force; aligned-rotation augmentation not applied",
        "coordinate_force_units": sorted(units),
    }
    for domain, values in per_domain.items():
        audit.per_domain[domain] = {"valid_atoms": int(torch.cat(values).numel()), "force_norm": _quantiles(torch.cat(values))}
    audit.checks = {
        "unit_consistency": len(units) == 1,
        "direct_force_scope": not any("force scope" in x for x in audit.failures),
        "provenance_identity": not any("provenance identity" in x for x in audit.failures),
        "finite_forces": bool(torch.isfinite(force).all()),
        "atom_ordering": "condition atom order is target force order (direct reader index)",
        "normalizer": "fit from train split only; no normalizer was fitted by this audit",
        "augmentation": "not applied",
        "per_atom_type_norms": {str(key): _quantiles(torch.cat(value)) for key, value in atom_type_norms.items()},
    }
    audit.passed = not audit.failures
    return audit


target_audit = audit_force_targets
