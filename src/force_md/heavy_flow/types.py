"""Typed, dense contracts for the clean-slate heavy-atom flow foundation.

This module deliberately does not import any of the legacy H0--H2 contracts.
The external representation is padded dense; the encoder packs only masked
atoms internally and always carries the mapping needed to restore atom order.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

__all__ = ["HeavyFlowCondition", "HeavyFlowTargets", "HeavyFlowProvenance", "HeavyFlowSample"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor(value: Any, name: str, ndim: int, dtype: torch.dtype | None = None) -> None:
    _require(isinstance(value, Tensor), f"{name} must be a torch.Tensor")
    _require(value.ndim == ndim, f"{name} must have rank {ndim}, got {tuple(value.shape)}")
    if dtype is not None:
        _require(value.dtype == dtype, f"{name} must have dtype {dtype}, got {value.dtype}")


@dataclass(frozen=True)
class HeavyFlowProvenance:
    """Source and unit metadata that travels with a condition/target pair."""

    length_unit: str = "angstrom"
    force_unit: str = "kcal/mol/angstrom"
    time_unit: str = "frame"
    time_per_frame: Optional[float] = None  # physical picoseconds per stored frame
    temperature_unit: str = "kelvin"
    replica: tuple[str, ...] = ()
    domain: tuple[str, ...] = ()
    frame: tuple[int, ...] = ()
    history_frames: tuple[tuple[int, ...], ...] = ()
    future_frame: tuple[int, ...] = ()
    preprocessing_version: str = "heavy_flow_v2"
    force_scope: str = "direct_heavy_atom"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class HeavyFlowCondition:
    """The only state accepted by the Stage-1 condition encoder.

    Shapes are dense and padded: sequence/residue fields ``[B,L]``, atom
    fields ``[B,N]``, history ``[B,K,N,3]``, and bonds ``[B,2,E]`` with -1
    padding plus ``[B,E]`` bond types.
    """

    sequence_tokens: Tensor
    residue_mask: Tensor
    atom_type: Tensor
    atom_name: Tensor
    atom_to_residue: Tensor
    atom_mask: Tensor
    x_history: Tensor
    bond_index: Tensor
    bond_type: Tensor
    temperature: Tensor
    lag: Tensor
    provenance: Optional[HeavyFlowProvenance] = None
    is_backbone: Optional[Tensor] = None
    is_sidechain: Optional[Tensor] = None
    terminal_residue: Optional[Tensor] = None
    chain_break: Optional[Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.sequence_tokens.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.sequence_tokens.shape[1])

    @property
    def max_atoms(self) -> int:
        return int(self.atom_type.shape[1])

    @property
    def history_length(self) -> int:
        return int(self.x_history.shape[1])

    @property
    def current_positions(self) -> Tensor:
        return self.x_history[:, -1]

    def validate(self) -> None:
        _tensor(self.sequence_tokens, "sequence_tokens", 2, torch.int64)
        _tensor(self.residue_mask, "residue_mask", 2, torch.bool)
        _require(self.sequence_tokens.shape == self.residue_mask.shape,
                 "sequence_tokens and residue_mask must have identical [B,L] shapes")
        b, l = self.sequence_tokens.shape
        for name, value in (("atom_type", self.atom_type), ("atom_name", self.atom_name),
                            ("atom_to_residue", self.atom_to_residue)):
            _tensor(value, name, 2, torch.int64)
            _require(value.shape[0] == b, f"{name} batch dimension disagrees with sequence_tokens")
        _tensor(self.atom_mask, "atom_mask", 2, torch.bool)
        _require(self.atom_mask.shape == self.atom_type.shape, "atom_mask must match atom fields")
        _tensor(self.x_history, "x_history", 4)
        _require(self.x_history.shape[:1] == (b,), "x_history batch dimension disagrees")
        _require(self.x_history.shape[2] == self.max_atoms and self.x_history.shape[3] == 3,
                 "x_history must have shape [B,K,N,3] matching atom fields")
        _require(self.history_length >= 1, "x_history must contain at least the current frame")
        _tensor(self.bond_index, "bond_index", 3, torch.int64)
        _require(self.bond_index.shape[0] == b and self.bond_index.shape[1] == 2,
                 "bond_index must have shape [B,2,E]")
        e = self.bond_index.shape[2]
        _tensor(self.bond_type, "bond_type", 2, torch.int64)
        _require(self.bond_type.shape == (b, e), "bond_type must have shape [B,E]")
        _tensor(self.temperature, "temperature", 1)
        _tensor(self.lag, "lag", 1)
        _require(self.temperature.shape == (b,), "temperature must have shape [B]")
        _require(self.lag.shape == (b,), "lag must have shape [B]")
        _require(bool(torch.all(self.atom_to_residue >= 0)), "atom_to_residue cannot be negative")
        _require(bool(torch.all(self.atom_to_residue < l)), "atom_to_residue points past sequence length")
        atom_residue_mask = self.residue_mask.gather(1, self.atom_to_residue)
        _require(bool(torch.all(~self.atom_mask | atom_residue_mask)),
                 "active atoms cannot belong to a masked residue")
        _require(bool(torch.all(self.x_history.isfinite())), "x_history contains non-finite values")

        valid_bonds = self.bond_index >= 0
        _require(bool(torch.all(valid_bonds.all(dim=1) | (~valid_bonds).all(dim=1))),
                 "bond_index entries must be complete pairs or -1 padded pairs")
        valid_pair = valid_bonds.all(dim=1)
        if bool(valid_pair.any()):
            endpoints = self.bond_index.permute(0, 2, 1)[valid_pair]
            _require(bool(torch.all(endpoints < self.max_atoms)), "bond endpoint exceeds N")
            _require(bool(torch.all(endpoints >= 0)), "valid bond endpoint is negative")
            batch_ids = torch.arange(b, device=self.bond_index.device)[:, None].expand(b, e)[valid_pair]
            endpoint_ok = self.atom_mask[batch_ids[:, None], endpoints]
            _require(bool(torch.all(endpoint_ok)), "bonds may only join active atoms")
            _require(bool(torch.all(endpoints[..., 0] != endpoints[..., 1])), "self-bonds are invalid")
            res0 = self.atom_to_residue[batch_ids, endpoints[..., 0]]
            res1 = self.atom_to_residue[batch_ids, endpoints[..., 1]]
            _require(bool(torch.all(self.residue_mask[batch_ids, res0] & self.residue_mask[batch_ids, res1])),
                     "bonds may not touch masked residues")

        optional = (("is_backbone", self.is_backbone, (b, self.max_atoms), torch.bool),
                    ("is_sidechain", self.is_sidechain, (b, self.max_atoms), torch.bool),
                    ("terminal_residue", self.terminal_residue, (b, l), torch.bool),
                    ("chain_break", self.chain_break, (b, max(l - 1, 0)), torch.bool))
        for name, value, shape, dtype in optional:
            if value is not None:
                _tensor(value, name, len(shape), dtype)
                _require(value.shape == shape, f"{name} must have shape {shape}")
        devices = {x.device for x in self._tensor_values()}
        _require(len(devices) == 1, "all HeavyFlowCondition tensors must share a device")

    def _tensor_values(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                yield value

    def to(self, device: torch.device | str) -> "HeavyFlowCondition":
        values = {field.name: (getattr(self, field.name).to(device)
                               if isinstance(getattr(self, field.name), Tensor)
                               else getattr(self, field.name))
                  for field in dataclasses.fields(self)}
        return dataclasses.replace(self, **values)


@dataclass
class HeavyFlowTargets:
    """Direct heavy-atom labels; no hydrogen-force aggregation is represented."""

    force_current: Tensor
    x_future: Tensor
    force_mask: Optional[Tensor] = None
    future_mask: Optional[Tensor] = None
    provenance: Optional[HeavyFlowProvenance] = None

    def validate(self, condition: HeavyFlowCondition) -> None:
        condition.validate()
        b, n = condition.batch_size, condition.max_atoms
        _tensor(self.force_current, "force_current", 3)
        _tensor(self.x_future, "x_future", 3)
        _require(self.force_current.shape == (b, n, 3), "force_current must be [B,N,3]")
        _require(self.x_future.shape == (b, n, 3), "x_future must be [B,N,3]")
        for name, value in (("force_mask", self.force_mask), ("future_mask", self.future_mask)):
            if value is not None:
                _tensor(value, name, 2, torch.bool)
                _require(value.shape == (b, n), f"{name} must be [B,N]")
                _require(bool(torch.all(~value | condition.atom_mask)),
                         f"{name} cannot activate padded atoms")
        _require(bool(torch.all(self.force_current.isfinite())), "force_current contains non-finite values")
        _require(bool(torch.all(self.x_future.isfinite())), "x_future contains non-finite values")
        _require(self.force_current.device == condition.sequence_tokens.device,
                 "targets and condition must share a device")


@dataclass
class HeavyFlowSample:
    condition: HeavyFlowCondition
    targets: HeavyFlowTargets

    def validate(self) -> None:
        self.condition.validate()
        self.targets.validate(self.condition)

    def to(self, device: torch.device | str) -> "HeavyFlowSample":
        return HeavyFlowSample(self.condition.to(device), dataclasses.replace(
            self.targets,
            force_current=self.targets.force_current.to(device),
            x_future=self.targets.x_future.to(device),
            force_mask=None if self.targets.force_mask is None else self.targets.force_mask.to(device),
            future_mask=None if self.targets.future_mask is None else self.targets.future_mask.to(device),
        ))
