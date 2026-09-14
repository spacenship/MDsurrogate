"""Data adapters and dense batching for :mod:`force_md.heavy_flow`.

The mdCATH bridge accepts the repository's existing *reader* output, but does
not consume its PLM embeddings or its hidden-force target. Forces remain the
direct force on the represented heavy atom at the same atom index.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor

from ..data.units import MDCATH_PS_PER_FRAME
from .types import HeavyFlowCondition, HeavyFlowProvenance, HeavyFlowSample, HeavyFlowTargets

__all__ = [
    "from_mdcath_example",
    "bonds_from_topology",
    "collate_heavy_flow",
    "make_history",
]


def _as_bonds(bond_index: Optional[Tensor], bond_type: Optional[Tensor], *, device, batch: int) -> tuple[Tensor, Tensor]:
    if bond_index is None:
        return (
            torch.empty((batch, 2, 0), dtype=torch.int64, device=device),
            torch.empty((batch, 0), dtype=torch.int64, device=device),
        )
    bonds = torch.as_tensor(bond_index, dtype=torch.int64, device=device)
    if bonds.ndim == 2 and bonds.shape[0] == 2:
        bonds = bonds.unsqueeze(0).expand(batch, -1, -1).clone()
    elif bonds.ndim == 2 and bonds.shape[1] == 2:
        bonds = bonds.t().unsqueeze(0).expand(batch, -1, -1).clone()
    elif bonds.ndim == 3 and bonds.shape[1] == 2:
        pass
    elif bonds.ndim == 3 and bonds.shape[2] == 2:
        bonds = bonds.transpose(1, 2).contiguous()
    else:
        raise ValueError("bond_index must be [2,E], [E,2], [B,2,E], or [B,E,2]")
    if bonds.shape[0] != batch:
        raise ValueError("bond_index batch dimension does not match the sample")
    e = bonds.shape[-1]
    if bond_type is None:
        types = torch.zeros((batch, e), dtype=torch.int64, device=device)
    else:
        types = torch.as_tensor(bond_type, dtype=torch.int64, device=device)
        if types.ndim == 1:
            types = types.unsqueeze(0).expand(batch, -1).clone()
        if types.shape != (batch, e):
            raise ValueError(f"bond_type must be [{batch},{e}], got {tuple(types.shape)}")
    return bonds, types


def bonds_from_topology(topology: Any, raw_to_batch: Tensor, *, device=None) -> tuple[Tensor, Tensor]:
    """Map PSF bonds into a heavy-atom batch order without inferring edges.

    ``topology`` is intentionally duck-typed. It may be the existing
    ``DomainHeavyTopology`` (whose ``topology.bonds`` is raw PSF indexing) or a
    small test object with ``bonds`` directly. Every bond with an omitted
    endpoint is excluded; no distance-based peptide or disulfide edge is added.
    """
    raw_to_batch = torch.as_tensor(raw_to_batch, dtype=torch.int64, device=device)
    raw_topology = getattr(topology, "topology", topology)
    bonds = torch.as_tensor(raw_topology.bonds, dtype=torch.int64, device=raw_to_batch.device)
    if bonds.ndim != 2:
        raise ValueError("topology.bonds must be [2,E]")
    if bonds.shape[0] != 2:
        bonds = bonds.t().contiguous()
    mapped = raw_to_batch[bonds]
    keep = (mapped >= 0).all(dim=0) & (mapped[0] != mapped[1])
    mapped = mapped[:, keep]
    return mapped, torch.zeros((mapped.shape[1],), dtype=torch.int64, device=mapped.device)


def make_history(current: Tensor, *, history_length: int = 1) -> Tensor:
    """Create an explicit repeated history for a frame-only smoke adapter."""
    if current.ndim != 2 or current.shape[-1] != 3:
        raise ValueError("current must have shape [N,3]")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    return current.unsqueeze(0).expand(history_length, -1, -1).clone()


def _terminal_and_break(chain_index: Tensor, residue_mask: Tensor) -> tuple[Tensor, Tensor]:
    l = int(chain_index.numel())
    terminal = torch.zeros((l,), dtype=torch.bool, device=chain_index.device)
    if l:
        active = residue_mask
        for i in range(l):
            if not bool(active[i]):
                continue
            before = i == 0 or not bool(active[i - 1]) or chain_index[i] != chain_index[i - 1]
            after = i == l - 1 or not bool(active[i + 1]) or chain_index[i] != chain_index[i + 1]
            terminal[i] = before or after
    chain_break = torch.zeros((max(l - 1, 0),), dtype=torch.bool, device=chain_index.device)
    if l > 1:
        chain_break[:] = (chain_index[1:] != chain_index[:-1]) | (~residue_mask[1:]) | (~residue_mask[:-1])
    return terminal, chain_break


def from_mdcath_example(
    example: Any,
    *,
    x_history: Optional[Tensor] = None,
    x_future: Optional[Tensor] = None,
    bond_index: Optional[Tensor] = None,
    bond_type: Optional[Tensor] = None,
    provenance: Optional[HeavyFlowProvenance] = None,
    allow_identity_future: bool = False,
) -> HeavyFlowSample:
    """Convert one existing mdCATH reader item into the new clean-slate contract.

    The legacy reader is used only to obtain coordinates, residue labels, atom
    names, and direct forces. ``hidden_force_target`` is deliberately ignored.
    """
    batch = example.batch
    atoms = batch.atoms
    residues = batch.residues
    if getattr(batch, "num_graphs", 1) != 1:
        raise ValueError("from_mdcath_example expects a single mdCATH frame")
    current = atoms.positions.detach()
    n = current.shape[0]
    if x_history is None:
        history = make_history(current)
    else:
        history = torch.as_tensor(x_history, dtype=current.dtype, device=current.device)
        if history.ndim == 4:
            if history.shape[0] != 1:
                raise ValueError("single-example x_history may have only batch size 1")
            history = history[0]
        if history.ndim != 3 or history.shape[1:] != (n, 3):
            raise ValueError("x_history must have shape [K,N,3]")
    if not torch.equal(history[-1], current):
        raise ValueError("history must end at the current frame in the same atom mapping")
    future = current if x_future is None and allow_identity_future else x_future
    if future is None:
        raise ValueError("x_future is required; identity future requires explicit allow_identity_future=True")
    future = torch.as_tensor(future, dtype=current.dtype, device=current.device)
    if future.ndim == 3 and future.shape[0] == 1:
        future = future[0]
    if future.shape != (n, 3):
        raise ValueError("x_future must have shape [N,3]")

    bidx, btype = _as_bonds(bond_index, bond_type, device=current.device, batch=1)
    residue_mask = residues.mask.to(device=current.device, dtype=torch.bool).unsqueeze(0)
    atom_mask = torch.ones((1, n), dtype=torch.bool, device=current.device)
    atom_to_residue = atoms.atom_to_residue.to(current.device).unsqueeze(0)
    is_backbone = atoms.is_backbone.to(current.device).unsqueeze(0)
    is_sidechain = (~atoms.is_backbone & ~atoms.is_cap & (atoms.atomic_number != 1)).to(current.device).unsqueeze(0)
    terminal, chain_break = _terminal_and_break(residues.chain_index.to(current.device), residues.mask.to(current.device))
    terminal = terminal.unsqueeze(0)
    chain_break = chain_break.unsqueeze(0)

    if provenance is None:
        units = batch.units
        provenance = HeavyFlowProvenance(
            length_unit=units.length,
            force_unit=units.force,
            temperature_unit=units.temperature,
            time_per_frame=MDCATH_PS_PER_FRAME,
            replica=tuple(str(int(x)) for x in batch.replica_index.tolist()),
            domain=tuple(str(x) for x in batch.domain_id),
            frame=tuple(int(x) for x in batch.frame_index.tolist()),
            force_scope="direct_heavy_atom",
        )
    condition = HeavyFlowCondition(
        sequence_tokens=residues.residue_type.to(current.device).unsqueeze(0),
        residue_mask=residue_mask,
        atom_type=atoms.atomic_number.to(current.device).unsqueeze(0),
        atom_name=atoms.atom_name_id.to(current.device).unsqueeze(0),
        atom_to_residue=atom_to_residue,
        atom_mask=atom_mask,
        x_history=history.unsqueeze(0),
        bond_index=bidx,
        bond_type=btype,
        temperature=batch.temperature.to(current.device),
        lag=torch.ones((1,), dtype=current.dtype, device=current.device),
        provenance=provenance,
        is_backbone=is_backbone,
        is_sidechain=is_sidechain,
        terminal_residue=terminal,
        chain_break=chain_break,
    )
    direct_force = atoms.forces.to(current.device)
    direct_mask = atoms.force_valid.to(current.device) if atoms.force_valid is not None else atom_mask[0]
    targets = HeavyFlowTargets(
        force_current=direct_force.unsqueeze(0),
        x_future=future.unsqueeze(0),
        force_mask=direct_mask.unsqueeze(0),
        future_mask=atom_mask,
        provenance=provenance,
    )
    sample = HeavyFlowSample(condition, targets)
    sample.validate()
    return sample


def collate_heavy_flow(samples: Sequence[HeavyFlowSample]) -> HeavyFlowSample:
    """Pad samples into one dense batch while preserving local atom ordering."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for sample in samples:
        sample.validate()
    b = len(samples)
    k = samples[0].condition.history_length
    if any(s.condition.history_length != k for s in samples):
        raise ValueError("all samples must have the same history length")
    l = max(s.condition.sequence_length for s in samples)
    n = max(s.condition.max_atoms for s in samples)
    e = max(s.condition.bond_index.shape[-1] for s in samples)
    device = samples[0].condition.sequence_tokens.device
    dtype = samples[0].condition.x_history.dtype
    seq = torch.zeros((b, l), dtype=torch.int64, device=device)
    rmask = torch.zeros((b, l), dtype=torch.bool, device=device)
    atype = torch.zeros((b, n), dtype=torch.int64, device=device)
    aname = torch.zeros((b, n), dtype=torch.int64, device=device)
    a2r = torch.zeros((b, n), dtype=torch.int64, device=device)
    amask = torch.zeros((b, n), dtype=torch.bool, device=device)
    hist = torch.zeros((b, k, n, 3), dtype=dtype, device=device)
    bonds = torch.full((b, 2, e), -1, dtype=torch.int64, device=device)
    btypes = torch.zeros((b, e), dtype=torch.int64, device=device)
    temp = torch.zeros((b,), dtype=samples[0].condition.temperature.dtype, device=device)
    lag = torch.zeros((b,), dtype=samples[0].condition.lag.dtype, device=device)
    backbone = torch.zeros((b, n), dtype=torch.bool, device=device)
    sidechain = torch.zeros((b, n), dtype=torch.bool, device=device)
    terminal = torch.zeros((b, l), dtype=torch.bool, device=device)
    chain_break = torch.zeros((b, max(l - 1, 0)), dtype=torch.bool, device=device)
    forces = torch.zeros((b, n, 3), dtype=samples[0].targets.force_current.dtype, device=device)
    future = torch.zeros_like(forces)
    force_mask = torch.zeros((b, n), dtype=torch.bool, device=device)
    future_mask = torch.zeros((b, n), dtype=torch.bool, device=device)
    for i, sample in enumerate(samples):
        c, t = sample.condition, sample.targets
        li, ni, ei = c.sequence_length, c.max_atoms, c.bond_index.shape[-1]
        seq[i, :li], rmask[i, :li] = c.sequence_tokens[0], c.residue_mask[0]
        atype[i, :ni], aname[i, :ni] = c.atom_type[0], c.atom_name[0]
        a2r[i, :ni], amask[i, :ni] = c.atom_to_residue[0], c.atom_mask[0]
        hist[i, :, :ni] = c.x_history[0]
        if ei:
            bonds[i, :, :ei], btypes[i, :ei] = c.bond_index[0], c.bond_type[0]
        temp[i], lag[i] = c.temperature[0], c.lag[0]
        if c.is_backbone is not None:
            backbone[i, :ni] = c.is_backbone[0]
        if c.is_sidechain is not None:
            sidechain[i, :ni] = c.is_sidechain[0]
        if c.terminal_residue is not None:
            terminal[i, :li] = c.terminal_residue[0]
        if c.chain_break is not None and li > 1:
            chain_break[i, :li - 1] = c.chain_break[0]
        forces[i, :ni], future[i, :ni] = t.force_current[0], t.x_future[0]
        force_mask[i, :ni] = t.force_mask[0] if t.force_mask is not None else c.atom_mask[0]
        future_mask[i, :ni] = t.future_mask[0] if t.future_mask is not None else c.atom_mask[0]
    provenance = HeavyFlowProvenance(
        length_unit=samples[0].condition.provenance.length_unit if samples[0].condition.provenance else "angstrom",
        force_unit=samples[0].condition.provenance.force_unit if samples[0].condition.provenance else "kcal/mol/angstrom",
        time_unit=samples[0].condition.provenance.time_unit if samples[0].condition.provenance else "frame",
        time_per_frame=samples[0].condition.provenance.time_per_frame if samples[0].condition.provenance else None,
        temperature_unit=samples[0].condition.provenance.temperature_unit if samples[0].condition.provenance else "kelvin",
        replica=tuple(x for s in samples for x in (s.condition.provenance.replica if s.condition.provenance else ())),
        domain=tuple(x for s in samples for x in (s.condition.provenance.domain if s.condition.provenance else ())),
        frame=tuple(x for s in samples for x in (s.condition.provenance.frame if s.condition.provenance else ())),
        history_frames=tuple(x for s in samples for x in (s.condition.provenance.history_frames if s.condition.provenance else ())),
        future_frame=tuple(x for s in samples for x in (s.condition.provenance.future_frame if s.condition.provenance else ())),
    )
    condition = HeavyFlowCondition(
        seq, rmask, atype, aname, a2r, amask, hist, bonds, btypes, temp, lag,
        provenance, backbone, sidechain, terminal, chain_break,
    )
    target = HeavyFlowTargets(forces, future, force_mask, future_mask, provenance)
    result = HeavyFlowSample(condition, target)
    result.validate()
    return result
