"""Typed contracts for the three-level hierarchical protein state.

Representation choice: **flattened ragged**, not padded dense.
    Every level stores its nodes concatenated along one axis with an explicit
    ``batch_index``, e.g. atoms are ``[N_atom, 3]`` with ``batch_index[N_atom]``
    rather than ``[B, max_atoms, 3]`` with a padding mask. Three reasons:

    1. mdCATH domains span 60-467 residues (~1k-7.2k atoms), so dense padding
       would waste most of the tensor on the long tail.
    2. ``e3nn`` tensor products consume ``[N, irreps_dim]`` 2-D inputs. A dense
       layout has to be flattened at every block anyway.
    3. Padded rows silently participate in equivariant pooling and in radial
       cutoffs unless every single op remembers the mask. A row that does not
       exist cannot be forgotten.

    The two layouts are never mixed. There is no dense code path.

Level structure (see ``docs/phase1_hierarchy_and_contracts.md``)::

    BackboneFrameBatch   B_i   geometry / frames        [N_res] nodes
    ResidueSemanticBatch R_i   identity / PLM semantics [N_res] nodes
    ProteinAtomBatch     A_ia  atoms                    [N_atom] nodes

``B_i`` and ``R_i`` are 1:1 per residue but stay distinct node types: they carry
different physics (geometry+time vs. chemistry+PLM) and are ablated separately.

Validation is deliberately *not* run in ``__init__``. Constructing a batch is on
the training hot path; ``validate()`` is called by the adapter once per batch
build, by the synthetic fixtures, and by the tests.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Self

import torch
from torch import Tensor

from .units import UnitMetadata

__all__ = [
    "ProteinAtomBatch",
    "ResidueSemanticBatch",
    "BackboneFrameBatch",
    "HierarchicalProteinBatch",
    "FrameGeometry",
]


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_tensor(
    t: Tensor,
    name: str,
    *,
    shape: tuple[Optional[int], ...],
    dtype: str,
) -> None:
    """Check rank, per-axis sizes (``None`` = free) and dtype family."""
    _check(isinstance(t, Tensor), f"{name} must be a torch.Tensor, got {type(t)!r}")
    _check(
        t.ndim == len(shape),
        f"{name} must have {len(shape)} dim(s), got shape {tuple(t.shape)}",
    )
    for axis, want in enumerate(shape):
        if want is not None:
            _check(
                t.shape[axis] == want,
                f"{name} axis {axis} must be {want}, got {tuple(t.shape)}",
            )
    if dtype == "float":
        _check(t.is_floating_point(), f"{name} must be floating point, got {t.dtype}")
    elif dtype == "int64":
        _check(t.dtype == torch.int64, f"{name} must be int64, got {t.dtype}")
    elif dtype == "bool":
        _check(t.dtype == torch.bool, f"{name} must be bool, got {t.dtype}")
    else:  # pragma: no cover - programming error
        raise AssertionError(f"unknown dtype family {dtype!r}")


def _check_index_range(t: Tensor, name: str, *, high: int) -> None:
    """Check that an index tensor lies in ``[0, high)``. Empty is allowed."""
    if t.numel() == 0:
        return
    lo = int(t.min())
    hi = int(t.max())
    _check(lo >= 0, f"{name} has negative index {lo}")
    _check(hi < high, f"{name} indexes {hi} but only {high} target node(s) exist")


def _check_non_decreasing(t: Tensor, name: str) -> None:
    if t.numel() < 2:
        return
    _check(
        bool(torch.all(t[1:] >= t[:-1])),
        f"{name} must be non-decreasing (nodes of one group must be contiguous)",
    )


@dataclass
class _TensorContainer:
    """Mixin: move/inspect the tensor fields of a dataclass generically."""

    def _tensor_items(self) -> Iterator[tuple[str, Tensor]]:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                yield f.name, v

    def to(self, device: Any, non_blocking: bool = False) -> Self:
        """Return a copy with every tensor field moved to ``device``."""
        moved = {
            f.name: (
                v.to(device, non_blocking=non_blocking) if isinstance(v, Tensor) else v
            )
            for f in dataclasses.fields(self)
            for v in (getattr(self, f.name),)
        }
        return dataclasses.replace(self, **moved)

    @property
    def device(self) -> torch.device:
        for _, v in self._tensor_items():
            return v.device
        raise RuntimeError(f"{type(self).__name__} holds no tensors")

    def _check_same_device(self) -> None:
        devices = {v.device for _, v in self._tensor_items()}
        _check(
            len(devices) <= 1,
            f"{type(self).__name__} tensors span multiple devices: {sorted(map(str, devices))}",
        )


# --------------------------------------------------------------------------
# Level A: atoms
# --------------------------------------------------------------------------


@dataclass
class ProteinAtomBatch(_TensorContainer):
    """Atom child nodes ``A_ia``, flattened over the batch.

    Coordinate convention: ``positions`` are **global** coordinates in the
    batch's length unit. Residue-local coordinates ``y_ia = R_i^T (x_ia - r_i)``
    are derived in :mod:`force_md.geometry`, not stored here, so that a batch
    always has exactly one source of truth for geometry.

    Args:
        positions: ``[N_atom, 3]`` float, global frame.
        atom_to_residue: ``[N_atom]`` int64 into ``[0, N_res)``. Must be
            non-decreasing: the atoms of a residue are contiguous. The adapter
            sorts atoms into this order once, which keeps pooling segments and
            edge construction deterministic.
        batch_index: ``[N_atom]`` int64 into ``[0, B)``, non-decreasing.
        atomic_number: ``[N_atom]`` int64, ``z`` as stored by mdCATH.
        atom_name_id: ``[N_atom]`` int64 into
            :data:`force_md.data.residue_constants.ATOM_NAMES`.
        is_backbone: ``[N_atom]`` bool, N/CA/C/O. Excludes CHARMM cap atoms.
        is_cap: ``[N_atom]`` bool, CHARMM terminal-patch atoms (``CAY``, ``NT``,
            ...) that belong to no standard residue template.
        forces: optional ``[N_atom, 3]`` float label, global frame, in the
            batch's force unit.
        force_valid: optional ``[N_atom]`` bool. False where the force label must
            not be trained on. mdCATH contains trajectories whose ``forces``
            dataset is a byte-identical copy of ``coords``; those are masked here
            rather than dropped, because their coordinates remain usable.
    """

    positions: Tensor
    atom_to_residue: Tensor
    batch_index: Tensor
    atomic_number: Tensor
    atom_name_id: Tensor
    is_backbone: Tensor
    is_cap: Tensor
    forces: Optional[Tensor] = None
    force_valid: Optional[Tensor] = None

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def is_heavy(self) -> Tensor:
        """``[N_atom]`` bool, True for every non-hydrogen atom."""
        return self.atomic_number != 1

    def validate(self, *, num_residues: Optional[int] = None,
                 num_graphs: Optional[int] = None) -> None:
        n = self.num_atoms
        _check_tensor(self.positions, "positions", shape=(n, 3), dtype="float")
        _check_tensor(self.atom_to_residue, "atom_to_residue", shape=(n,), dtype="int64")
        _check_tensor(self.batch_index, "batch_index", shape=(n,), dtype="int64")
        _check_tensor(self.atomic_number, "atomic_number", shape=(n,), dtype="int64")
        _check_tensor(self.atom_name_id, "atom_name_id", shape=(n,), dtype="int64")
        _check_tensor(self.is_backbone, "is_backbone", shape=(n,), dtype="bool")
        _check_tensor(self.is_cap, "is_cap", shape=(n,), dtype="bool")
        _check_non_decreasing(self.atom_to_residue, "atom_to_residue")
        _check_non_decreasing(self.batch_index, "atom batch_index")
        if num_residues is not None:
            _check_index_range(self.atom_to_residue, "atom_to_residue", high=num_residues)
        if num_graphs is not None:
            _check_index_range(self.batch_index, "atom batch_index", high=num_graphs)
        if self.forces is not None:
            _check_tensor(self.forces, "forces", shape=(n, 3), dtype="float")
        if self.force_valid is not None:
            _check_tensor(self.force_valid, "force_valid", shape=(n,), dtype="bool")
            _check(
                self.forces is not None,
                "force_valid was given without forces; a validity mask over an "
                "absent label is meaningless",
            )
        self._check_same_device()


# --------------------------------------------------------------------------
# Level R: residue semantics
# --------------------------------------------------------------------------


@dataclass
class ResidueSemanticBatch(_TensorContainer):
    """Residue semantic nodes ``R_i``: identity and frozen PLM embedding.

    Everything here is an SE(3) **invariant** scalar. No geometry lives at this
    level; that is the backbone node's job. The PLM embedding is stored once per
    residue and reaches atoms through a learned gated broadcast, never by
    copying the vector onto each child atom.

    Args:
        residue_type: ``[N_res]`` int64 into
            :data:`force_md.data.residue_constants.RESIDUE_TYPES`.
        plm_embedding: ``[N_res, D_plm]`` float, frozen ESM-2 residue embedding
            aligned to residues with special tokens already removed.
        resid_original: ``[N_res]`` int64, the residue numbering as stored in the
            source file. mdCATH keeps original PDB numbering (e.g. 8..87), which
            is *not* the 0-based node index and must not be used as one.
        chain_index: ``[N_res]`` int64, 0-based chain id within its graph.
        batch_index: ``[N_res]`` int64 into ``[0, B)``, non-decreasing.
        mask: ``[N_res]`` bool, False for residues excluded from losses
            (nonstandard residue, missing frame atoms, ...).
    """

    residue_type: Tensor
    plm_embedding: Tensor
    resid_original: Tensor
    chain_index: Tensor
    batch_index: Tensor
    mask: Tensor

    @property
    def num_residues(self) -> int:
        return int(self.residue_type.shape[0])

    @property
    def plm_dim(self) -> int:
        return int(self.plm_embedding.shape[1])

    def validate(self, *, num_graphs: Optional[int] = None) -> None:
        n = self.num_residues
        _check_tensor(self.residue_type, "residue_type", shape=(n,), dtype="int64")
        _check_tensor(self.plm_embedding, "plm_embedding", shape=(n, None), dtype="float")
        _check_tensor(self.resid_original, "resid_original", shape=(n,), dtype="int64")
        _check_tensor(self.chain_index, "chain_index", shape=(n,), dtype="int64")
        _check_tensor(self.batch_index, "batch_index", shape=(n,), dtype="int64")
        _check_tensor(self.mask, "residue mask", shape=(n,), dtype="bool")
        _check_non_decreasing(self.batch_index, "residue batch_index")
        if num_graphs is not None:
            _check_index_range(self.batch_index, "residue batch_index", high=num_graphs)
        self._check_same_device()


# --------------------------------------------------------------------------
# Level B: backbone frames
# --------------------------------------------------------------------------


@dataclass
class BackboneFrameBatch(_TensorContainer):
    """Backbone-frame scaffold nodes ``B_i``: one rigid frame per residue.

    Stores the three frame-defining atom positions rather than a rotation
    matrix, so the frame stays a differentiable function of the coordinates and
    the ``R_i``/``det=+1`` invariants are established in exactly one place
    (:mod:`force_md.geometry.frames`).

    Args:
        n_positions / ca_positions / c_positions: ``[N_res, 3]`` float, global
            frame. The residue origin is ``r_i = ca_positions[i]``.
        residue_to_backbone: ``[N_res]`` int64, the 1:1 map from residue node to
            backbone node. Kept explicit even though it is ``arange(N_res)``
            today, so the two node types never become implicitly fused.
        frame_valid: ``[N_res]`` bool. False where N/CA/C are missing or
            collinear, which makes the frame degenerate.
        batch_index: ``[N_res]`` int64 into ``[0, B)``, non-decreasing.
    """

    n_positions: Tensor
    ca_positions: Tensor
    c_positions: Tensor
    residue_to_backbone: Tensor
    frame_valid: Tensor
    batch_index: Tensor

    @property
    def num_frames(self) -> int:
        return int(self.ca_positions.shape[0])

    def validate(self, *, num_residues: Optional[int] = None,
                 num_graphs: Optional[int] = None) -> None:
        n = self.num_frames
        _check_tensor(self.n_positions, "n_positions", shape=(n, 3), dtype="float")
        _check_tensor(self.ca_positions, "ca_positions", shape=(n, 3), dtype="float")
        _check_tensor(self.c_positions, "c_positions", shape=(n, 3), dtype="float")
        _check_tensor(self.residue_to_backbone, "residue_to_backbone", shape=(n,), dtype="int64")
        _check_tensor(self.frame_valid, "frame_valid", shape=(n,), dtype="bool")
        _check_tensor(self.batch_index, "batch_index", shape=(n,), dtype="int64")
        _check_non_decreasing(self.batch_index, "backbone batch_index")
        if num_residues is not None:
            _check(
                n == num_residues,
                f"backbone nodes ({n}) must be 1:1 with residues ({num_residues})",
            )
            _check_index_range(self.residue_to_backbone, "residue_to_backbone", high=n)
            srt = torch.sort(self.residue_to_backbone).values
            _check(
                bool(torch.equal(srt, torch.arange(n, device=srt.device))),
                "residue_to_backbone must be a permutation of arange(N_res); "
                "the residue<->backbone relation is 1:1",
            )
        if num_graphs is not None:
            _check_index_range(self.batch_index, "backbone batch_index", high=num_graphs)
        self._check_same_device()


# --------------------------------------------------------------------------
# The full hierarchical state
# --------------------------------------------------------------------------


@dataclass
class HierarchicalProteinBatch(_TensorContainer):
    """One batch of hierarchical protein states ``q_t = {B_i, R_i, A_ia}``.

    Args:
        atoms / residues / backbone: the three node levels.
        units: units of every physical tensor in this batch.
        temperature: ``[B]`` float, simulation temperature in kelvin. mdCATH is
            sampled at 320/348/379/413/450 K and temperature is a conditioning
            input, never a nuisance to be averaged over.
        domain_id: length-``B`` tuple of CATH domain identifiers.
        replica_index: ``[B]`` int64, mdCATH replica (0-4).
        frame_index: ``[B]`` int64, index of the frame within its trajectory.
            Deliberately in **frames, not nanoseconds**: mdCATH stores no
            per-frame timestamp, so a physical lag is only attached once the
            user supplies ``ps_per_frame``.
    """

    atoms: ProteinAtomBatch
    residues: ResidueSemanticBatch
    backbone: BackboneFrameBatch
    units: UnitMetadata
    temperature: Tensor
    domain_id: tuple[str, ...]
    replica_index: Tensor
    frame_index: Tensor

    # -- sizes ------------------------------------------------------------
    @property
    def num_graphs(self) -> int:
        return int(self.temperature.shape[0])

    @property
    def num_atoms(self) -> int:
        return self.atoms.num_atoms

    @property
    def num_residues(self) -> int:
        return self.residues.num_residues

    @property
    def device(self) -> torch.device:
        return self.atoms.positions.device

    # -- movement ---------------------------------------------------------
    def to(self, device: Any, non_blocking: bool = False) -> Self:
        return dataclasses.replace(
            self,
            atoms=self.atoms.to(device, non_blocking),
            residues=self.residues.to(device, non_blocking),
            backbone=self.backbone.to(device, non_blocking),
            temperature=self.temperature.to(device, non_blocking=non_blocking),
            replica_index=self.replica_index.to(device, non_blocking=non_blocking),
            frame_index=self.frame_index.to(device, non_blocking=non_blocking),
        )

    # -- derived indices --------------------------------------------------
    def atom_graph_index(self) -> Tensor:
        """``[N_atom]`` graph id implied by each atom's parent residue."""
        return self.residues.batch_index[self.atoms.atom_to_residue]

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        """Full structural check. Not on the training hot path."""
        b = self.num_graphs
        _check_tensor(self.temperature, "temperature", shape=(b,), dtype="float")
        _check_tensor(self.replica_index, "replica_index", shape=(b,), dtype="int64")
        _check_tensor(self.frame_index, "frame_index", shape=(b,), dtype="int64")
        _check(
            len(self.domain_id) == b,
            f"domain_id has {len(self.domain_id)} entries but batch size is {b}",
        )
        _check(
            isinstance(self.units, UnitMetadata),
            f"units must be UnitMetadata, got {type(self.units)!r}",
        )

        self.residues.validate(num_graphs=b)
        self.backbone.validate(num_residues=self.num_residues, num_graphs=b)
        self.atoms.validate(num_residues=self.num_residues, num_graphs=b)

        # cross-level consistency: an atom's graph must be its residue's graph.
        _check(
            bool(torch.equal(self.atoms.batch_index, self.atom_graph_index())),
            "atom batch_index disagrees with residues.batch_index[atom_to_residue]; "
            "an atom is assigned to a different graph than its parent residue",
        )
        # the backbone node of a residue must live in the same graph
        _check(
            bool(torch.equal(
                self.backbone.batch_index[self.backbone.residue_to_backbone],
                self.residues.batch_index,
            )),
            "backbone batch_index disagrees with residue batch_index",
        )
        # every residue must own at least one atom, otherwise pooling produces a
        # silently zero feature for a node that still contributes to the loss.
        counts = torch.bincount(
            self.atoms.atom_to_residue, minlength=self.num_residues
        )
        empty = int((counts == 0).sum())
        _check(
            empty == 0,
            f"{empty} residue(s) own no atoms; an empty pooling segment would "
            "produce a zero feature indistinguishable from a real one",
        )
        devices = {self.atoms.device, self.residues.device, self.backbone.device,
                   self.temperature.device}
        _check(
            len(devices) == 1,
            f"batch spans multiple devices: {sorted(map(str, devices))}",
        )


# --------------------------------------------------------------------------
# Phase 1.5: geometry-only frames (history and future)
# --------------------------------------------------------------------------


@dataclass
class FrameGeometry(_TensorContainer):
    """Positions of one *other* frame of the same trajectory.

    Phase 1.5 needs two kinds of extra frame beside the current state: the
    history frame ``t-1`` and the future frame ``t+lag``. Neither is a model
    input in the sense the current frame is -- the history contributes geometry
    only, and the future is a **label** -- so neither is represented as a full
    :class:`HierarchicalProteinBatch`.

    Two reasons this is a separate type rather than a second batch:

    1. Chemistry, sequence and the PLM embedding are constants of the domain, so
       carrying a second copy per frame would triple the memory for nothing.
    2. The future frame is a target. Giving it the same type as the input state
       would make ``model(future)`` a well-typed expression, and the one mistake
       that would invalidate every Phase 1.5 number is reading the future in the
       conditioning path. Making it a different type makes that a type error
       rather than a silent leak.

    Atom rows are aligned **row-for-row** with the current state's
    ``atoms.positions``, and residue rows with ``residues``/``backbone``; the
    same topology produced both, so an index means the same thing in each.

    Args:
        positions: ``[N_atom, 3]`` global frame, same atom order as the state.
        n_positions / ca_positions / c_positions: ``[N_res, 3]`` backbone atoms.
        frame_valid: ``[N_res]`` bool, False where N/CA/C were missing.
        atom_batch_index: ``[N_atom]`` int64 into ``[0, B)``, non-decreasing.
        residue_batch_index: ``[N_res]`` int64 into ``[0, B)``, non-decreasing.
        frame_index: ``[B]`` int64, the **raw** trajectory frame number.
    """

    positions: Tensor
    n_positions: Tensor
    ca_positions: Tensor
    c_positions: Tensor
    frame_valid: Tensor
    atom_batch_index: Tensor
    residue_batch_index: Tensor
    frame_index: Tensor

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_residues(self) -> int:
        return int(self.ca_positions.shape[0])

    @property
    def num_graphs(self) -> int:
        return int(self.frame_index.shape[0])

    def validate(
        self,
        *,
        num_atoms: Optional[int] = None,
        num_residues: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ) -> None:
        n_a, n_r = self.num_atoms, self.num_residues
        _check_tensor(self.positions, "frame positions", shape=(n_a, 3), dtype="float")
        for name in ("n_positions", "ca_positions", "c_positions"):
            _check_tensor(getattr(self, name), f"frame {name}", shape=(n_r, 3), dtype="float")
        _check_tensor(self.frame_valid, "frame_valid", shape=(n_r,), dtype="bool")
        _check_tensor(self.atom_batch_index, "frame atom_batch_index", shape=(n_a,), dtype="int64")
        _check_tensor(
            self.residue_batch_index, "frame residue_batch_index", shape=(n_r,), dtype="int64"
        )
        _check_tensor(self.frame_index, "frame_index", shape=(None,), dtype="int64")
        _check_non_decreasing(self.atom_batch_index, "frame atom_batch_index")
        _check_non_decreasing(self.residue_batch_index, "frame residue_batch_index")
        if num_atoms is not None:
            _check(
                n_a == num_atoms,
                f"frame has {n_a} atoms but the state it accompanies has {num_atoms}; "
                "the two frames must share one topology, row for row",
            )
        if num_residues is not None:
            _check(
                n_r == num_residues,
                f"frame has {n_r} residues but the state it accompanies has {num_residues}",
            )
        if num_graphs is not None:
            _check(
                self.num_graphs == num_graphs,
                f"frame spans {self.num_graphs} graphs but the state spans {num_graphs}",
            )
            _check_index_range(self.atom_batch_index, "frame atom_batch_index", high=num_graphs)
            _check_index_range(
                self.residue_batch_index, "frame residue_batch_index", high=num_graphs
            )
        self._check_same_device()

    def matches(self, batch: "HierarchicalProteinBatch") -> None:
        """Validate against the state this frame accompanies."""
        self.validate(
            num_atoms=batch.num_atoms,
            num_residues=batch.num_residues,
            num_graphs=batch.num_graphs,
        )
