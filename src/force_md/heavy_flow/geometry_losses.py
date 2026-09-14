"""Differentiable geometry regularizers for the direct Cartesian flow.

The rectified-flow objective is the primary Stage 4 loss.  These terms are
small, endpoint-only regularizers: they constrain the decoded ``X_hat1`` to
retain the covalent geometry of the aligned target without changing the flow
target or introducing a teacher trajectory into the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from ..data.residue_constants import ATOM_NAME_TO_ID, NUM_ATOM_NAMES
from ..geometry.torsions import dihedral_angle
from .atom_graph import COVALENT_EDGE
from .flow_types import FlowAtomTopology

__all__ = [
    "GeometryLossConfig",
    "GeometryLossBreakdown",
    "geometry_regularization",
    "compute_geometry_regularization",
    "geometry_gradient_norms",
    "combined_flow_loss",
]


@dataclass(frozen=True)
class GeometryLossConfig:
    bond_weight: float = 1.0
    angle_weight: float = 0.1
    peptide_weight: float = 0.1
    backbone_torsion_weight: float = 0.05
    sidechain_torsion_weight: float = 0.05
    chirality_weight: float = 0.05
    clash_weight: float = 0.02
    clash_min_distance: float = 1.5
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "bond_weight", "angle_weight", "peptide_weight",
            "backbone_torsion_weight", "sidechain_torsion_weight",
            "chirality_weight", "clash_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.clash_min_distance <= 0 or self.epsilon <= 0:
            raise ValueError("clash_min_distance and epsilon must be positive")


@dataclass(frozen=True)
class GeometryLossBreakdown:
    bond: Tensor
    angle: Tensor
    peptide: Tensor
    backbone_torsion: Tensor
    sidechain_torsion: Tensor
    chirality: Tensor
    clash: Tensor

    @property
    def total(self) -> Tensor:
        return self.bond + self.angle + self.peptide + self.backbone_torsion + self.sidechain_torsion + self.chirality + self.clash

    @property
    def terms(self) -> dict[str, Tensor]:
        return {
            "bond": self.bond,
            "angle": self.angle,
            "peptide": self.peptide,
            "backbone_torsion": self.backbone_torsion,
            "sidechain_torsion": self.sidechain_torsion,
            "chirality": self.chirality,
            "clash": self.clash,
        }

    def as_dict(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.terms.items()} | {
            "total": float(self.total.detach())
        }


def _zero(reference: Tensor) -> Tensor:
    # Keeping a graph connection is useful when a batch happens to have no
    # valid geometry (e.g. a one-atom masked fixture).
    return reference.sum() * 0.0


def _dense_positions(value: Tensor, topology: FlowAtomTopology) -> Tensor:
    if value.ndim == 3 and value.shape == (topology.batch_size, topology.max_atoms, 3):
        return value
    if value.ndim == 2 and value.shape == (topology.atom_batch.shape[0], 3):
        dense = value.new_zeros((topology.batch_size, topology.max_atoms, 3))
        dense[topology.atom_batch, topology.atom_local] = value
        return dense
    raise ValueError("positions must be [B,N,3] or packed [M,3]")


def _unique_covalent_pairs(topology: FlowAtomTopology) -> Tensor:
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(topology.fixed_edge_index.shape[1]):
        if int(topology.fixed_edge_kind[i]) != COVALENT_EDGE:
            continue
        a, b = (int(topology.fixed_edge_index[0, i]), int(topology.fixed_edge_index[1, i]))
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            rows.append((a, b))
    if not rows:
        return torch.empty((0, 2), dtype=torch.int64, device=topology.atom_batch.device)
    return torch.tensor(rows, dtype=torch.int64, device=topology.atom_batch.device)


def _packed_active(topology: FlowAtomTopology) -> Tensor:
    return topology.atom_mask[topology.atom_batch, topology.atom_local]


def _per_batch_mean(values: Tensor, batch: Tensor, batch_size: int, valid: Optional[Tensor] = None) -> Tensor:
    if values.numel() == 0:
        return _zero(values)
    if valid is None:
        valid = torch.ones(values.shape, dtype=torch.bool, device=values.device)
    total = values.new_zeros((batch_size,))
    count = values.new_zeros((batch_size,))
    total.scatter_add_(0, batch[valid], values[valid])
    count.scatter_add_(0, batch[valid], torch.ones_like(values[valid]))
    return (total / count.clamp_min(1.0)).mean()


def _bond_loss(pred: Tensor, target: Tensor, topology: FlowAtomTopology, config: GeometryLossConfig) -> Tensor:
    pairs = _unique_covalent_pairs(topology)
    if pairs.numel() == 0:
        return _zero(pred)
    a, b = pairs.unbind(-1)
    pred_d = (pred[a] - pred[b]).norm(dim=-1)
    target_d = (target[a] - target[b]).norm(dim=-1)
    active = _packed_active(topology)
    valid = active[a] & active[b]
    values = (pred_d - target_d).square()
    batch = topology.atom_batch[a]
    return _per_batch_mean(values, batch, topology.batch_size, valid)


def _angle_triplets(topology: FlowAtomTopology) -> Tensor:
    neighbours: dict[int, set[int]] = {}
    for a, b in _unique_covalent_pairs(topology).tolist():
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    rows: list[tuple[int, int, int]] = []
    for centre, values in neighbours.items():
        values = sorted(values)
        rows.extend((left, centre, right) for i, left in enumerate(values) for right in values[i + 1:])
    if not rows:
        return torch.empty((0, 3), dtype=torch.int64, device=topology.atom_batch.device)
    return torch.tensor(rows, dtype=torch.int64, device=topology.atom_batch.device)


def _angle_loss(pred: Tensor, target: Tensor, topology: FlowAtomTopology, config: GeometryLossConfig) -> Tensor:
    triplets = _angle_triplets(topology)
    if triplets.numel() == 0:
        return _zero(pred)
    a, centre, b = triplets.unbind(-1)
    def cosine(x: Tensor) -> Tensor:
        left = x[a] - x[centre]
        right = x[b] - x[centre]
        return (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(config.epsilon)
    values = (cosine(pred) - cosine(target)).square()
    active = _packed_active(topology)
    valid = active[a] & active[centre] & active[b]
    batch = topology.atom_batch[centre]
    return _per_batch_mean(values, batch, topology.batch_size, valid)


def _metadata_indices(topology: FlowAtomTopology, names: Iterable[str], *, same_residue: bool = False) -> list[tuple[int, int, int, int]]:
    if topology.atom_name is None:
        return []
    wanted = {name: ATOM_NAME_TO_ID.get(name, 0) for name in names}
    rows: list[tuple[int, int, int, int]] = []
    for batch in range(topology.batch_size):
        active = torch.nonzero(topology.atom_mask[batch], as_tuple=False).flatten().tolist()
        by_residue: dict[int, dict[str, int]] = {}
        for atom in active:
            residue = int(topology.atom_to_residue[batch, atom]) if topology.atom_to_residue is not None else 0
            by_residue.setdefault(residue, {})
            for name, atom_id in wanted.items():
                if int(topology.atom_name[batch, atom]) == atom_id:
                    by_residue[residue][name] = atom
        residues = sorted(by_residue)
        if same_residue:
            for residue in residues:
                row = by_residue[residue]
                if all(name in row for name in names):
                    rows.append(tuple(row[name] for name in names))  # type: ignore[arg-type]
            else:
                for left, right in zip(residues[:-1], residues[1:]):
                    if right != left + 1:
                        continue
                    if topology.chain_break is not None and left < topology.chain_break.shape[1]:
                        if bool(topology.chain_break[batch, left]):
                            continue
                    first, second = by_residue[left], by_residue[right]
                # This branch is used only by peptide C(i)-N(i+1); the names
                # are intentionally explicit so chain breaks cannot connect it.
                if names == ("C", "N"):
                    if "C" in first and "N" in second:
                        rows.append((first["C"], second["N"], -1, batch))
    return rows


def _coerce_quadruplets(indices: Optional[Tensor], topology: FlowAtomTopology) -> Tensor:
    if indices is None:
        return torch.empty((0, 4), dtype=torch.int64, device=topology.atom_batch.device)
    value = torch.as_tensor(indices, dtype=torch.int64, device=topology.atom_batch.device)
    if value.ndim == 2 and value.shape[-1] == 4:
        return value
    if value.ndim == 3 and value.shape[-1] == 4:
        return value.reshape(-1, 4)
    raise ValueError("torsion/chirality indices must have shape [Q,4] or [B,Q,4]")


def _torsion_loss(pred: Tensor, target: Tensor, indices: Tensor, topology: FlowAtomTopology, config: GeometryLossConfig) -> Tensor:
    if indices.numel() == 0:
        return _zero(pred)
    a, b, c, d = indices.unbind(-1)
    pred_angle = dihedral_angle(pred[a], pred[b], pred[c], pred[d], eps=config.epsilon)
    target_angle = dihedral_angle(target[a], target[b], target[c], target[d], eps=config.epsilon)
    values = 2.0 - 2.0 * torch.cos(pred_angle - target_angle)
    active = _packed_active(topology)
    valid = active[a] & active[b] & active[c] & active[d]
    batch = topology.atom_batch[b]
    return _per_batch_mean(values, batch, topology.batch_size, valid)


def _chirality_loss(pred: Tensor, target: Tensor, indices: Tensor, topology: FlowAtomTopology, config: GeometryLossConfig) -> Tensor:
    if indices.numel() == 0:
        return _zero(pred)
    a, b, c, d = indices.unbind(-1)
    def volume(x: Tensor) -> Tensor:
        u, v, w = x[b] - x[a], x[c] - x[a], x[d] - x[a]
        denominator = u.norm(dim=-1) * v.norm(dim=-1) * w.norm(dim=-1)
        return (torch.linalg.cross(u, v, dim=-1) * w).sum(-1) / denominator.clamp_min(config.epsilon)
    # A smooth signed-volume match keeps the model from learning a mirrored
    # tetrahedron while remaining differentiable near ordinary protein geometry.
    values = (volume(pred) - volume(target)).square()
    active = _packed_active(topology)
    valid = active[a] & active[b] & active[c] & active[d]
    batch = topology.atom_batch[b]
    return _per_batch_mean(values, batch, topology.batch_size, valid)


def _clash_loss(pred: Tensor, topology: FlowAtomTopology, config: GeometryLossConfig) -> Tensor:
    total = _zero(pred)
    pair_keys: set[tuple[int, int]] = set()
    for i in range(topology.fixed_edge_index.shape[1]):
        a, b = (int(topology.fixed_edge_index[0, i]), int(topology.fixed_edge_index[1, i]))
        pair_keys.add((min(a, b), max(a, b)))
    pieces: list[Tensor] = []
    batches: list[Tensor] = []
    for batch in range(topology.batch_size):
        ids = torch.nonzero(topology.atom_mask[batch], as_tuple=False).flatten()
        if ids.numel() < 2:
            continue
        packed_ids = torch.nonzero(topology.atom_batch == batch, as_tuple=False).flatten()
        packed_ids = packed_ids[torch.isin(topology.atom_local[packed_ids], ids)]
        dist = torch.cdist(pred[packed_ids], pred[packed_ids])
        eligible = []
        for i in range(ids.numel()):
            for j in range(i + 1, ids.numel()):
                a, b = int(packed_ids[i]), int(packed_ids[j])
                if (min(a, b), max(a, b)) in pair_keys:
                    continue
                eligible.append((i, j))
        if eligible:
            ij = torch.tensor(eligible, dtype=torch.int64, device=pred.device)
            values = torch.relu(pred.new_tensor(config.clash_min_distance) - dist[ij[:, 0], ij[:, 1]]).square()
            pieces.append(values)
            batches.append(torch.full((values.numel(),), batch, dtype=torch.int64, device=pred.device))
    if not pieces:
        return total
    values = torch.cat(pieces)
    batch = torch.cat(batches)
    return _per_batch_mean(values, batch, topology.batch_size)


def compute_geometry_regularization(
    prediction: Tensor,
    target: Tensor,
    topology: FlowAtomTopology,
    *,
    config: Optional[GeometryLossConfig] = None,
    backbone_torsion_indices: Optional[Tensor] = None,
    sidechain_torsion_indices: Optional[Tensor] = None,
    chirality_indices: Optional[Tensor] = None,
) -> GeometryLossBreakdown:
    """Compute endpoint geometry terms, all masked by active atoms.

    Optional explicit index arrays are packed atom indices.  When omitted,
    backbone chirality is inferred from the atom-name/residue metadata and
    side-chain/backbone torsions remain zero unless a dataset provides their
    authoritative quadruplets.
    """
    topology.validate()
    pred = _dense_positions(prediction, topology)
    target = _dense_positions(target, topology)
    if pred.shape != target.shape:
        raise ValueError("prediction and target positions must have the same shape")
    pred_packed = pred[topology.atom_batch, topology.atom_local]
    target_packed = target[topology.atom_batch, topology.atom_local]
    config = config or GeometryLossConfig()

    peptide = _metadata_indices(topology, ("C", "N"))
    if peptide:
        peptide_idx = torch.tensor([(a, b) for a, b, _, _ in peptide], dtype=torch.int64, device=pred.device)
        peptide_batch = torch.tensor([batch for _, _, _, batch in peptide], dtype=torch.int64, device=pred.device)
        peptide_values = (pred_packed[peptide_idx[:, 0]] - pred_packed[peptide_idx[:, 1]]).norm(dim=-1)
        target_values = (target_packed[peptide_idx[:, 0]] - target_packed[peptide_idx[:, 1]]).norm(dim=-1)
        # The index rows are grouped by batch, so the extra batch mask avoids
        # accidentally comparing residue metadata across padded proteins.
        peptide_valid = torch.ones_like(peptide_values, dtype=torch.bool)
        peptide_term = _per_batch_mean((peptide_values - target_values).square(), peptide_batch, topology.batch_size, peptide_valid)
    else:
        peptide_term = _zero(pred)

    backbone = _coerce_quadruplets(backbone_torsion_indices, topology)
    sidechain = _coerce_quadruplets(sidechain_torsion_indices, topology)
    chirality = _coerce_quadruplets(chirality_indices, topology)
    if chirality.numel() == 0 and topology.atom_name is not None and topology.atom_to_residue is not None:
        inferred = _metadata_indices(topology, ("N", "CA", "C", "CB"), same_residue=True)
        if inferred:
            chirality = torch.tensor(inferred, dtype=torch.int64, device=pred.device)

    return GeometryLossBreakdown(
        bond=config.bond_weight * _bond_loss(pred_packed, target_packed, topology, config),
        angle=config.angle_weight * _angle_loss(pred_packed, target_packed, topology, config),
        peptide=config.peptide_weight * peptide_term,
        backbone_torsion=config.backbone_torsion_weight * _torsion_loss(pred_packed, target_packed, backbone, topology, config),
        sidechain_torsion=config.sidechain_torsion_weight * _torsion_loss(pred_packed, target_packed, sidechain, topology, config),
        chirality=config.chirality_weight * _chirality_loss(pred_packed, target_packed, chirality, topology, config),
        clash=config.clash_weight * _clash_loss(pred_packed, topology, config),
    )


geometry_regularization = compute_geometry_regularization


def geometry_gradient_norms(
    breakdown: GeometryLossBreakdown | Mapping[str, Tensor],
    parameters: Iterable[nn.Parameter],
) -> dict[str, float]:
    """Return per-term gradient norms for diagnosing over-strong regularizers."""
    terms = breakdown.terms if isinstance(breakdown, GeometryLossBreakdown) else dict(breakdown)
    params = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, float] = {}
    for name, term in terms.items():
        if not term.requires_grad or not params:
            result[name] = 0.0
            continue
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        squared = sum((grad.detach().square().sum() for grad in grads if grad is not None), term.new_zeros(()))
        result[name] = float(squared.sqrt())
    return result


def combined_flow_loss(
    flow_loss: Tensor,
    geometry: GeometryLossBreakdown,
    *,
    geometry_scale: float = 1.0,
) -> Tensor:
    if geometry_scale < 0:
        raise ValueError("geometry_scale must be non-negative")
    return flow_loss + geometry_scale * geometry.total
