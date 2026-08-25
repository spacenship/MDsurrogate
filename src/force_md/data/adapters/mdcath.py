"""mdCATH adapter.

Every field is read from the file; nothing about the schema is assumed. The
layout was verified by inspection before this was written (see
``docs/phase1_hierarchy_and_contracts.md``):

* per-atom ``z``, ``element``, ``resid``, ``resname``, ``chain`` describe the
  topology; atom **names** come from the ``pdbProteinAtoms`` PDB text, whose ATOM
  records are in the same order as those arrays.
* ``coords`` are Angstrom and ``forces`` kcal/mol/A, taken from the datasets'
  own ``unit`` attributes rather than assumed. Coordinates are **centred per
  frame** before leaving the adapter -- see ``__getitem__``; the stored values are
  absolute box positions, which are numerically poor input for a model that only
  ever uses differences of them.
* hydrogens are present, so the omitted-atom residual is a real target.
* five temperatures x five replicas per domain; no per-frame timestamp exists,
  so ``frame_index`` counts frames and no nanosecond value is invented.

Three failure modes are handled explicitly rather than silently:

**Corrupt force labels.** Some published trajectories ship a ``forces`` array
byte-identical to ``coords``. Those are marked through ``force_valid``, per
trajectory, from the quarantine file written by
``scripts/audit_mdcath_forces.py``. The coordinates stay usable.

**Saturated coordinates.** Some published frames have *every* atom at
``2.14748e7 A``, which is ``INT_MAX / 1000 nm`` -- an int32 overflow in the
upstream XTC writer. Those frames are dropped while the index is built, from
``scripts/audit_mdcath_coords.py``. They cannot be caught by the PBC check
below: with every atom at the same saturated value the CA-CA distances are
exactly zero, so a corrupt frame looks more intact than a real one.

**PBC.** The protein is stored whole in every frame inspected (consecutive CA-CA
distances 3.67-4.01 A). The adapter checks this per frame and raises instead of
silently unwrapping, because a wrong unwrap produces plausible-looking garbage.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ...conditioning.esm2 import Esm2Config, Esm2EmbeddingCache
from ..contracts import (
    BackboneFrameBatch,
    HierarchicalProteinBatch,
    ProteinAtomBatch,
    ResidueSemanticBatch,
)
from ..residue_constants import (
    ELEMENT_TO_Z,
    atom_name_id,
    is_backbone_atom,
    is_cap_atom,
    residue_type_id,
)
from ..synthetic import fake_plm_embedding
from ..units import MDCATH_UNITS

__all__ = ["MdCathConfig", "MdCathDataset", "TrainingExample", "split_domains"]

#: A CA-CA virtual bond longer than this means the chain is broken across the
#: periodic boundary (the real value is ~3.8 A).
_MAX_CA_CA = 5.0

#: Backstop for saturated coordinates. Real mdCATH coordinates are absolute box
#: positions and reach a few hundred Angstrom; the defect is at 2.1e7, so this
#: threshold is orders of magnitude clear of both. It should never fire once
#: ``coord_quarantine_path`` is supplied -- it exists so an unaudited shard fails
#: with a message that names the frame, instead of poisoning the forward pass
#: with a NaN several hours into a run.
_MAX_ABS_COORD = 1.0e4

#: Smallest spatial extent a real protein can have. The smallest domain here is
#: 50 residues and spans tens of Angstrom; a collapsed frame measures exactly 0.
#: Kept in step with ``MIN_EXTENT_ANGSTROM`` in ``scripts/audit_mdcath_coords.py``
#: -- the audit and this backstop must reject the same frames, or the audit will
#: pass data that then dies at load time.
_MIN_EXTENT = 1.0


@dataclass
class TrainingExample:
    """One frame plus the targets that are not part of the state contract.

    Args:
        batch: the hierarchical state (represented atoms only).
        hidden_force_target: ``[N_res, 3]`` summed force on the atoms that are
            *not* represented -- the identifiable omitted-atom residual. None
            when every atom is represented.
    """

    batch: HierarchicalProteinBatch
    hidden_force_target: Optional[Tensor]


@dataclass(frozen=True)
class MdCathConfig:
    """Adapter configuration.

    Args:
        data_dir: directory of ``mdcath_dataset_*.h5`` shards.
        esm2_cache_dir: directory of precomputed embeddings. When None, a
            deterministic fake embedding is used -- acceptable for tests, never
            for a reported result, so it must be opted into.
        quarantine_path: JSON from ``audit_mdcath_forces.py``; marks trajectories
            whose force labels are unusable, keeping their coordinates.
        coord_quarantine_path: JSON from ``audit_mdcath_coords.py``; drops
            individual frames whose coordinates saturated at INT_MAX. Unlike the
            force quarantine this removes frames from the index entirely, because
            a frame with no position information has nothing left to learn from.
        represented_scope: which atoms become nodes. ``heavy_atom`` is the
            deployable V1; ``all_atom`` represents hydrogens too.
        temperatures / replicas: subsets to use. None means all.
        frames_per_trajectory: evenly spaced frames sampled from each trajectory.
        max_domains / max_residues: caps for a mini-subset run.
        allow_fake_plm: permit the fake embedding fallback.
        check_pbc: verify the chain is whole in every loaded frame.
    """

    data_dir: str
    esm2_cache_dir: Optional[str] = None
    quarantine_path: Optional[str] = None
    coord_quarantine_path: Optional[str] = None
    represented_scope: str = "heavy_atom"
    temperatures: Optional[tuple[int, ...]] = None
    replicas: Optional[tuple[int, ...]] = None
    frames_per_trajectory: int = 4
    #: Physical time per frame. Not in the files; 1000.0 ps for mdCATH per the
    #: publication. Unused by Phase 1, required by Phase 2's multi-lag training.
    ps_per_frame: Optional[float] = None
    max_domains: Optional[int] = None
    max_residues: Optional[int] = None
    allow_fake_plm: bool = False
    check_pbc: bool = True
    plm_dim: int = 1280
    dtype: torch.dtype = torch.float32


@dataclass
class _Topology:
    """Per-domain constants, read once and reused across every frame."""

    domain: str
    atom_to_residue: np.ndarray
    atomic_number: np.ndarray
    atom_name_ids: np.ndarray
    is_backbone: np.ndarray
    is_cap: np.ndarray
    represented: np.ndarray
    residue_type: np.ndarray
    resid_original: np.ndarray
    chain_index: np.ndarray
    ca_index: np.ndarray
    n_index: np.ndarray
    c_index: np.ndarray
    num_residues: int
    plm: Tensor
    frame_atoms_complete: np.ndarray
    #: Permutation applied to the raw atom order so that atoms are grouped by
    #: residue, or None when the file is already grouped. Per domain, never
    #: stored on the dataset: a shared field would apply one domain's
    #: permutation to another's coordinates.
    atom_order: Optional[np.ndarray]


def split_domains(
    domains: Sequence[str], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split **by domain**, before any frame is enumerated.

    Splitting after windowing would put frames of the same protein on both sides
    and turn validation into a memorisation check: consecutive mdCATH frames of
    one domain are far from independent.
    """
    ordered = sorted(domains)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    val_idx = set(perm[:n_val].tolist())
    train = [d for i, d in enumerate(ordered) if i not in val_idx]
    val = [d for i, d in enumerate(ordered) if i in val_idx]
    return train, val


class MdCathDataset(torch.utils.data.Dataset):
    """One item = one frame of one (domain, temperature, replica) trajectory."""

    def __init__(
        self,
        config: MdCathConfig,
        domains: Optional[Sequence[str]] = None,
        *,
        build_index: bool = True,
    ):
        """Args:
            build_index: enumerate the Phase 1 frame index. ``False`` gives an
                empty ``index`` and leaves file discovery, quarantine loading and
                topology caching in place, which is what the Phase 1.5 lag-pair
                dataset composes: it needs the readers and the safety filters but
                builds its own index from **raw** frame numbers. Phase 1's index
                samples ``frames_per_trajectory`` frames evenly, ~12-13 frames
                apart, so reusing it for a 1-frame lag would be off by an order of
                magnitude.
        """
        import h5py  # noqa: PLC0415 - optional dependency

        self.config = config
        self._h5py = h5py
        self._files: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        self._topology: dict[str, _Topology] = {}

        for path in sorted(glob.glob(os.path.join(config.data_dir, "*.h5"))):
            domain = os.path.basename(path)[len("mdcath_dataset_") : -len(".h5")]
            self._files[domain] = path

        available = list(self._files)
        if domains is not None:
            missing = set(domains) - set(available)
            if missing:
                raise ValueError(f"domains not present in {config.data_dir}: {sorted(missing)[:5]}")
            available = [d for d in available if d in set(domains)]

        self.quarantine = self._load_quarantine()
        self.coord_quarantine = self._load_coord_quarantine()
        self.index: list[tuple[str, str, str, int]] = []
        if build_index:
            self._build_index(available)
        self.available_domains: list[str] = list(available)
        # Close before any fork: an inherited h5py handle is not safe to use
        # from a DataLoader worker. Handles reopen lazily, per process.
        self.close()

    # -- setup -------------------------------------------------------------

    def _require_quarantine(self, path: Optional[str], script: str) -> Optional[dict]:
        """Load a quarantine file, or refuse to run without one.

        ``None`` is an explicit opt-out and is honoured. A path that is *set* but
        missing is not: the config asked for filtering and there is none, and the
        old behaviour -- return an empty dict and carry on -- meant a fresh clone
        or a re-download trained on known-bad frames with nothing in the log to
        say so. Silence is the wrong answer to "the safety file is absent".
        """
        if path is None:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path!r} is configured as a quarantine file but does not exist. "
                f"mdCATH ships frames that are silently unusable; run "
                f"{script} to regenerate it, or set the path to null to train "
                f"without that filter deliberately."
            )
        return json.load(open(path))

    def _load_quarantine(self) -> dict[str, set[str]]:
        payload = self._require_quarantine(
            self.config.quarantine_path, "scripts/audit_mdcath_forces.py"
        )
        if payload is None:
            return {}
        return {k: set(v) for k, v in payload.get("quarantine", {}).items()}

    def _load_coord_quarantine(self) -> dict[str, dict[str, set[int]]]:
        payload = self._require_quarantine(
            self.config.coord_quarantine_path, "scripts/audit_mdcath_coords.py"
        )
        if payload is None:
            return {}
        return {
            domain: {traj: {int(f) for f in frames} for traj, frames in trajs.items()}
            for domain, trajs in payload.get("quarantine", {}).items()
        }

    def _open(self, domain: str):
        handle = self._handles.get(domain)
        if handle is None:
            handle = self._h5py.File(self._files[domain], "r")
            self._handles[domain] = handle
        return handle

    def _build_index(self, domains: Sequence[str]) -> None:
        kept = 0
        for domain in domains:
            if self.config.max_domains is not None and kept >= self.config.max_domains:
                break
            g = self._open(domain)[domain]
            if (
                self.config.max_residues is not None
                and int(g.attrs["numResidues"]) > self.config.max_residues
            ):
                continue
            temps = sorted(k for k in g.keys() if k.isdigit())
            if self.config.temperatures is not None:
                wanted = {str(t) for t in self.config.temperatures}
                temps = [t for t in temps if t in wanted]
            any_frame = False
            for temp in temps:
                reps = sorted(g[temp].keys())
                if self.config.replicas is not None:
                    wanted_r = {str(r) for r in self.config.replicas}
                    reps = [r for r in reps if r in wanted_r]
                for rep in reps:
                    n_frames = int(g[temp][rep].attrs["numFrames"])
                    if n_frames == 0:
                        continue
                    # Sample evenly over the frames that survive the coordinate
                    # quarantine, rather than sampling first and discarding the
                    # hits: dropping picks would silently thin an affected
                    # trajectory and leave a gap where the corruption was.
                    usable = np.arange(n_frames)
                    corrupt = self.coord_quarantine.get(domain, {}).get(f"{temp}/{rep}")
                    if corrupt:
                        usable = usable[~np.isin(usable, list(corrupt))]
                        if len(usable) == 0:
                            continue
                    slots = np.linspace(
                        0, len(usable) - 1, self.config.frames_per_trajectory
                    ).round().astype(int)
                    picks = np.unique(usable[slots])
                    for f in picks:
                        self.index.append((domain, temp, rep, int(f)))
                        any_frame = True
            if any_frame:
                kept += 1

    # -- topology ----------------------------------------------------------

    def _topology_for(self, domain: str) -> _Topology:
        cached = self._topology.get(domain)
        if cached is not None:
            return cached

        g = self._open(domain)[domain]
        resid = np.asarray(g["resid"][:])
        element = np.array([e.decode() for e in g["element"][:]])
        resname = np.array([r.decode() for r in g["resname"][:]])
        chain = np.array([c.decode() for c in g["chain"][:]])
        names = np.array(
            [
                line[12:16].strip()
                for line in g["pdbProteinAtoms"][()].decode().split("\n")
                if line.startswith("ATOM")
            ]
        )
        if len(names) != len(resid):
            raise ValueError(
                f"{domain}: {len(names)} ATOM records but {len(resid)} atoms; the "
                "PDB text and the per-atom arrays disagree"
            )

        unique_resid = np.unique(resid)
        a2r_all = np.searchsorted(unique_resid, resid)
        atom_order: Optional[np.ndarray] = None
        if not np.all(np.diff(a2r_all) >= 0):
            atom_order = np.argsort(a2r_all, kind="stable")
            a2r_all, element, resname, chain, names = (
                a2r_all[atom_order], element[atom_order], resname[atom_order],
                chain[atom_order], names[atom_order],
            )

        heavy = element != "H"
        represented = np.ones_like(heavy) if self.config.represented_scope == "all_atom" else heavy

        # residue-level arrays, taken from the first atom of each residue
        first = np.zeros(len(unique_resid), dtype=np.int64)
        for i in range(len(a2r_all) - 1, -1, -1):
            first[a2r_all[i]] = i
        residue_type = np.array([residue_type_id(resname[i]) for i in first])
        chain_names = {c: i for i, c in enumerate(sorted(set(chain)))}
        chain_index = np.array([chain_names[chain[i]] for i in first])

        n_res = len(unique_resid)
        n_idx = np.full(n_res, -1, dtype=np.int64)
        ca_idx = np.full(n_res, -1, dtype=np.int64)
        c_idx = np.full(n_res, -1, dtype=np.int64)
        for slot, target in ((0, "N"), (1, "CA"), (2, "C")):
            hit = np.nonzero(names == target)[0]
            dest = (n_idx, ca_idx, c_idx)[slot]
            dest[a2r_all[hit]] = hit
        complete = (n_idx >= 0) & (ca_idx >= 0) & (c_idx >= 0)

        topo = _Topology(
            domain=domain,
            atom_to_residue=a2r_all,
            atomic_number=np.array([ELEMENT_TO_Z.get(e, 6) for e in element]),
            atom_name_ids=np.array([atom_name_id(n) for n in names]),
            is_backbone=np.array([is_backbone_atom(n) for n in names]),
            is_cap=np.array([is_cap_atom(n) for n in names]),
            represented=represented,
            residue_type=residue_type,
            resid_original=unique_resid,
            chain_index=chain_index,
            ca_index=ca_idx,
            n_index=n_idx,
            c_index=c_idx,
            num_residues=n_res,
            plm=self._plm_for(domain, residue_type),
            frame_atoms_complete=complete,
            atom_order=atom_order,
        )
        self._topology[domain] = topo
        return topo

    def _plm_for(self, domain: str, residue_type: np.ndarray) -> Tensor:
        types = torch.tensor(residue_type, dtype=torch.int64)
        if self.config.esm2_cache_dir:
            cache = Esm2EmbeddingCache(self.config.esm2_cache_dir)
            if cache.exists(domain):
                emb = cache.load(domain)
                if emb.shape[0] != len(residue_type):
                    raise ValueError(
                        f"{domain}: cached embedding has {emb.shape[0]} rows for "
                        f"{len(residue_type)} residues"
                    )
                return emb
        if not self.config.allow_fake_plm:
            raise FileNotFoundError(
                f"no cached ESM-2 embedding for {domain} in "
                f"{self.config.esm2_cache_dir!r}. Run scripts/precompute_esm2.py, "
                "or set allow_fake_plm=True for a test run (never for a reported "
                "result)."
            )
        return fake_plm_embedding(
            types, self.config.plm_dim, 0, position_index=torch.arange(len(residue_type))
        )

    # -- items -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def topology_for(self, domain: str) -> _Topology:
        """Per-domain constants (atom order, residue map, PLM), read once."""
        return self._topology_for(domain)

    def __getitem__(self, i: int) -> TrainingExample:
        domain, temp, rep, frame = self.index[i]
        coords, forces, forces_valid = self.load_frame_arrays(domain, temp, rep, frame)
        return self.build_example(domain, temp, rep, frame, coords, forces, forces_valid)

    def load_frame_arrays(
        self, domain: str, temp: str, rep: str, frame: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Read one raw frame with every safety check applied.

        Split out of :meth:`__getitem__` so that a second index -- the Phase 1.5
        lag-pair dataset, which needs three or four frames of one trajectory per
        item -- reuses these checks instead of carrying a second copy of them. A
        duplicated backstop is a backstop that will be fixed in one place only.

        Returns:
            ``(coords, forces, forces_valid)``. ``coords`` is centred on its own
            centroid, as it is for Phase 1; ``forces_valid`` is False for a
            trajectory in the force quarantine.
        """
        topo = self._topology_for(domain)
        g = self._open(domain)[domain][temp][rep]

        coords = np.asarray(g["coords"][frame], dtype=np.float64)
        forces = np.asarray(g["forces"][frame], dtype=np.float64)
        if topo.atom_order is not None:
            coords = coords[topo.atom_order]
            forces = forces[topo.atom_order]

        # Backstop, cheap next to the read that just happened. These frames are
        # finite, so they pass every NaN guard downstream and only surface as a
        # NaN after the first tensor product -- by which point nothing names the
        # frame that caused it.
        #
        # Two invariants, because one is not enough. Magnitude catches the
        # saturated frames; **extent** catches the collapsed ones, whose
        # coordinates are all exactly (0,0,0) and therefore pass any magnitude
        # threshold however tight. Checking only magnitude here is what let a
        # single collapsed frame poison four training runs.
        peak = float(np.abs(coords).max())
        if not np.isfinite(peak) or peak > _MAX_ABS_COORD:
            raise ValueError(
                f"{domain}/{temp}/{rep} frame {frame}: |coordinate| reaches "
                f"{peak:.6g} A. 2.14748e7 is INT_MAX/1000 nm, an int32 overflow in "
                "the upstream XTC writer, and the frame holds no position "
                "information. Run scripts/audit_mdcath_coords.py and pass its "
                "output as coord_quarantine_path to drop these frames."
            )
        extent = float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).max())
        if extent < _MIN_EXTENT:
            raise ValueError(
                f"{domain}/{temp}/{rep} frame {frame}: every atom lies within "
                f"{extent:.3g} A of the centroid, so the frame has no spatial "
                "extent -- a final frame that was allocated but never written. "
                "Its forces look normal and its coordinate magnitudes pass every "
                "threshold; only extent catches it. Run "
                "scripts/audit_mdcath_coords.py and pass its output as "
                "coord_quarantine_path to drop these frames."
            )

        # Centre the frame. mdCATH stores absolute box positions reaching ~460 A,
        # while everything the model computes is a difference of two of them --
        # edge vectors, lever arms, frame axes -- of a few Angstrom. Subtracting
        # the centroid is physically a no-op for a translation-invariant model and
        # stops ~8 mantissa bits being spent on an offset that is never used.
        # Measured on one batch: with absolute coordinates, switching matmuls to
        # TF32 moved the loss from 2.43 to 12.68; centred, the same switch moves
        # it by 0.0008.
        # MUST stay after the saturation check above. In a corrupt frame every
        # atom holds the same saturated value, so its centroid is that value and
        # centring maps the whole frame to exactly zero -- which sails through a
        # magnitude test and hands the graph builder a protein of coincident atoms.
        coords = coords - coords.mean(axis=0, keepdims=True)

        if self.config.check_pbc:
            ca = coords[topo.ca_index[topo.frame_atoms_complete]]
            if len(ca) > 1:
                d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
                if float(d.max()) > _MAX_CA_CA:
                    raise ValueError(
                        f"{domain}/{temp}/{rep} frame {frame}: CA-CA distance "
                        f"{d.max():.2f} A exceeds {_MAX_CA_CA} A, so the chain is "
                        "broken across the periodic boundary. This adapter does "
                        "not unwrap; supply preprocessed coordinates."
                    )

        forces_valid = f"{temp}/{rep}" not in self.quarantine.get(domain, set())
        return coords, forces, forces_valid

    def build_example(
        self,
        domain: str,
        temp: str,
        rep: str,
        frame: int,
        coords: np.ndarray,
        forces: np.ndarray,
        forces_valid: bool,
    ) -> TrainingExample:
        """Turn checked frame arrays into the hierarchical state contract."""
        topo = self._topology_for(domain)
        dtype = self.config.dtype
        rep_mask = topo.represented
        keep = np.nonzero(rep_mask)[0]
        remap = np.full(len(rep_mask), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))

        # omitted-atom residual: the summed force on the atoms we do not represent
        hidden = None
        omitted = np.nonzero(~rep_mask)[0]
        if len(omitted):
            hidden_np = np.zeros((topo.num_residues, 3))
            np.add.at(hidden_np, topo.atom_to_residue[omitted], forces[omitted])
            hidden = torch.tensor(hidden_np, dtype=dtype)

        t = lambda a, d: torch.tensor(a, dtype=d)  # noqa: E731
        atoms = ProteinAtomBatch(
            positions=t(coords[keep], dtype),
            atom_to_residue=t(topo.atom_to_residue[keep], torch.int64),
            batch_index=torch.zeros(len(keep), dtype=torch.int64),
            atomic_number=t(topo.atomic_number[keep], torch.int64),
            atom_name_id=t(topo.atom_name_ids[keep], torch.int64),
            is_backbone=t(topo.is_backbone[keep], torch.bool),
            is_cap=t(topo.is_cap[keep], torch.bool),
            forces=t(forces[keep], dtype),
            force_valid=torch.full((len(keep),), forces_valid, dtype=torch.bool),
        )
        residues = ResidueSemanticBatch(
            residue_type=t(topo.residue_type, torch.int64),
            plm_embedding=topo.plm.to(dtype),
            resid_original=t(topo.resid_original, torch.int64),
            chain_index=t(topo.chain_index, torch.int64),
            batch_index=torch.zeros(topo.num_residues, dtype=torch.int64),
            mask=t(topo.frame_atoms_complete, torch.bool),
        )
        safe = lambda idx: np.where(idx >= 0, idx, 0)  # noqa: E731
        backbone = BackboneFrameBatch(
            n_positions=t(coords[safe(topo.n_index)], dtype),
            ca_positions=t(coords[safe(topo.ca_index)], dtype),
            c_positions=t(coords[safe(topo.c_index)], dtype),
            residue_to_backbone=torch.arange(topo.num_residues, dtype=torch.int64),
            frame_valid=t(topo.frame_atoms_complete, torch.bool),
            batch_index=torch.zeros(topo.num_residues, dtype=torch.int64),
        )
        batch = HierarchicalProteinBatch(
            atoms=atoms, residues=residues, backbone=backbone, units=MDCATH_UNITS,
            temperature=torch.tensor([float(temp)], dtype=dtype),
            domain_id=(domain,),
            replica_index=torch.tensor([int(rep)], dtype=torch.int64),
            frame_index=torch.tensor([frame], dtype=torch.int64),
        )
        return TrainingExample(batch=batch, hidden_force_target=hidden)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    @property
    def domains(self) -> list[str]:
        return sorted({d for d, _, _, _ in self.index})
