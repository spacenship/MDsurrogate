"""Per-domain heavy-atom topology, in the batch's own atom ordering.

The mdCATH adapter does two things to the raw protein atom array before it
becomes ``batch.atoms``: it may **permute** atoms so they group by residue, and
it **filters** to the represented scope (heavy atoms, by default). PSF indices
address the raw array, so they have to be carried through both steps or every
bond in the exclusion mask points at the wrong atom -- silently, and with
plausible-looking results.

Rather than trusting a re-implementation of the adapter's two steps to stay in
sync with it, this module **verifies** the mapping at runtime: after remapping,
the atomic numbers and atom-name ids it derives must equal the ones the batch
carries, element for element. A permutation change in the adapter turns into a
loud failure here instead of a quiet mislabelling.

Also surfaces two datasets that ship in every shard and that no other code reads:
``dssp`` (per frame, per residue) and ``rmsf`` (per residue, over the whole
trajectory). They were recorded as unavailable in the Phase 1.6 Stage M report
because neither could be *computed*; they do not need to be.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

from ..data.residue_order import residue_order

import numpy as np
import torch
from torch import Tensor

from ..data.residue_constants import (
    atom_name_id,
    canonical_resname,
    is_backbone_atom,
    is_cap_atom,
)
from .chemistry import BONDI_VDW_RADII
from .topology import MolecularTopology, build_topology, chi_instances

__all__ = ["DomainHeavyTopology", "load_domain_topology", "clear_cache"]


@dataclass
class DomainHeavyTopology:
    """Topology and per-atom labels for one domain, in **batch atom order**.

    Args:
        domain: CATH domain id.
        raw_to_batch: ``[num_raw_atoms]`` int64, row in the batch atom array for
            each raw protein atom, ``-1`` for atoms the batch does not represent
            (hydrogens, under the default scope).
        batch_to_raw: ``[num_batch_atoms]`` int64, the inverse.
        topology: PSF-derived bonds/angles/dihedrals/impropers, in **raw**
            indexing. Kept raw because that is what the chi resolver needs -- it
            has to see the hydrogens to know a torsion's moving set.
        atom_names / elements: raw indexing, length ``num_raw_atoms``.
        is_backbone / is_sidechain / is_cap: batch indexing.
        vdw_radius: batch indexing, angstrom.
        residue_index: batch indexing.
        residue_index_raw: raw indexing, for callers working in PSF atom order.
        chi: resolved chi instances, in raw indexing.
        dssp: ``[n_frames, n_residues]`` per-frame secondary structure, or None.
        rmsf: ``[n_residues]`` per-residue RMSF over the trajectory, or None.
    """

    domain: str
    raw_to_batch: Tensor
    batch_to_raw: Tensor
    topology: MolecularTopology
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    is_backbone: Tensor
    is_sidechain: Tensor
    is_cap: Tensor
    vdw_radius: Tensor
    residue_index: Tensor
    residue_index_raw: Tensor
    chi: tuple
    dssp: Optional[np.ndarray]
    rmsf: Optional[np.ndarray]
    _mask_cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_batch_atoms(self) -> int:
        return int(self.batch_to_raw.numel())

    def verify_against(self, atomic_number: Tensor, atom_name_ids: Tensor) -> None:
        """Assert this mapping reproduces the batch's own per-atom arrays.

        The whole correctness of the exclusion mask rests on this. Comparing the
        atomic numbers alone would pass for a permutation that swaps two carbons;
        comparing the name ids as well pins the actual atom.
        """
        expected_z = torch.tensor(
            [
                {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}.get(
                    self.elements[int(r)], 6
                )
                for r in self.batch_to_raw
            ],
            dtype=torch.int64,
        )
        if not torch.equal(expected_z, atomic_number.cpu()):
            wrong = int((expected_z != atomic_number.cpu()).sum())
            raise ValueError(
                f"{self.domain}: heavy-atom mapping disagrees with the batch on "
                f"{wrong} atomic number(s). The adapter's atom permutation or "
                "represented-scope filter changed; the PSF exclusion mask would "
                "point at the wrong atoms."
            )
        expected_names = torch.tensor(
            [atom_name_id(self.atom_names[int(r)]) for r in self.batch_to_raw],
            dtype=torch.int64,
        )
        if not torch.equal(expected_names, atom_name_ids.cpu()):
            wrong = int((expected_names != atom_name_ids.cpu()).sum())
            raise ValueError(
                f"{self.domain}: heavy-atom mapping disagrees with the batch on "
                f"{wrong} atom name(s), though the elements matched. This is the "
                "case an element-only check would have missed."
            )

    def exclusion_masks(self, *, exclude_1_4: bool = False, device=None):
        """``(excluded, is_1_4)`` in **batch** indexing, cached per domain.

        The masks are a function of the topology alone, not of any frame, so
        they are built once per domain and per device. Rebuilding them per graph
        costs a Python pass over ~17,000 PSF records and a 1500x1500 scatter
        every time -- measured at roughly half the H0 wall time before this cache
        existed, for an answer that never changes.
        """
        key = (bool(exclude_1_4), str(device))
        cached = self._mask_cache.get(key)
        if cached is None:
            from .topology import bonded_exclusion_mask

            keep = self.raw_to_batch >= 0
            excluded, is_1_4 = bonded_exclusion_mask(
                self.topology, keep, exclude_1_4=exclude_1_4
            )
            if device is not None:
                excluded, is_1_4 = excluded.to(device), is_1_4.to(device)
            cached = (excluded, is_1_4)
            self._mask_cache[key] = cached
        return cached


def _read_text(dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def _decode(array) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in array]


@lru_cache(maxsize=64)
def load_domain_topology(
    data_dir: str,
    domain: str,
    *,
    represented_scope: str = "heavy_atom",
    want_dssp: bool = True,
    want_rmsf: bool = True,
    dssp_trajectory: str = "320/0",
) -> DomainHeavyTopology:
    """Build (and cache) the heavy-atom topology of one domain.

    Reads the shard directly rather than going through the adapter, so no
    existing code path changes. The adapter's atom permutation is reproduced
    here and then **verified** by :meth:`DomainHeavyTopology.verify_against`.
    """
    import h5py  # noqa: PLC0415 - optional dependency, only needed for real data

    path = os.path.join(data_dir, f"mdcath_dataset_{domain}.h5")
    with h5py.File(path, "r") as handle:
        group = handle[domain]
        psf_text = _read_text(group["psf"])
        z_raw = np.asarray(group["z"][:], dtype=np.int64)
        resid = np.asarray(group["resid"][:])
        chain = np.asarray(group["chain"][:])
        resname = np.array(_decode(group["resname"][:]))
        element = np.array(_decode(group["element"][:]))
        pdb_text = _read_text(group["pdbProteinAtoms"])
        dssp = rmsf = None
        if dssp_trajectory in group:
            trajectory = group[dssp_trajectory]
            if want_dssp and "dssp" in trajectory:
                dssp = np.array(_decode(trajectory["dssp"][:].ravel()), dtype=object)
                dssp = dssp.reshape(trajectory["dssp"].shape)
            if want_rmsf and "rmsf" in trajectory:
                rmsf = np.asarray(trajectory["rmsf"][:], dtype=np.float64)

    names = np.array(
        [
            line[12:16].strip()
            for line in pdb_text.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
    )
    if len(names) != len(z_raw):
        raise ValueError(
            f"{domain}: {len(names)} PDB ATOM records but {len(z_raw)} atoms"
        )

    # -- reproduce the adapter's two steps, in the same order -----------------
    unique_resid, _, a2r = residue_order(resid, chain)
    z_original = z_raw.copy()
    order = None
    if not np.all(np.diff(a2r) >= 0):
        order = np.argsort(a2r, kind="stable")
        a2r, element, resname, names, z_raw = (
            a2r[order], element[order], resname[order], names[order], z_raw[order]
        )
    heavy = element != "H"
    represented = np.ones_like(heavy) if represented_scope == "all_atom" else heavy

    num_raw = len(names)
    raw_to_batch = np.full(num_raw, -1, dtype=np.int64)
    keep = np.nonzero(represented)[0]
    raw_to_batch[keep] = np.arange(len(keep))

    topology = build_topology(
        psf_text, num_raw, torch.as_tensor(z_original, dtype=torch.int64)
    )
    # The PSF indexes the *unpermuted* array. When the adapter permuted, the PSF
    # records must be permuted the same way or every bond points elsewhere.
    if order is not None:
        inverse = np.empty(num_raw, dtype=np.int64)
        inverse[order] = np.arange(num_raw)
        remap = torch.as_tensor(inverse, dtype=torch.int64)
        topology = MolecularTopology(
            bonds=remap[topology.bonds],
            angles=remap[topology.angles],
            dihedrals=remap[topology.dihedrals],
            impropers=remap[topology.impropers],
            num_atoms=num_raw,
            hydrogen_parent=torch.where(
                topology.hydrogen_parent >= 0,
                remap[topology.hydrogen_parent.clamp(min=0)],
                torch.full_like(topology.hydrogen_parent, -1),
            )[torch.as_tensor(order)],
            disulfides=remap[topology.disulfides],
            is_hydrogen=topology.is_hydrogen[torch.as_tensor(order)],
        )

    first = np.zeros(len(unique_resid), dtype=np.int64)
    for i in range(len(a2r) - 1, -1, -1):
        first[a2r[i]] = i
    residue_names = [canonical_resname(resname[i]) for i in first]

    backbone_raw = torch.tensor(
        [i for i, n in enumerate(names) if is_backbone_atom(n)], dtype=torch.int64
    )
    chi = tuple(
        chi_instances(
            topology,
            residue_names=residue_names,
            atom_to_residue=torch.as_tensor(a2r, dtype=torch.int64),
            atom_names=list(names),
            backbone_atom_indices=backbone_raw,
        )
    )

    batch_to_raw = torch.as_tensor(keep, dtype=torch.int64)
    batch_names = [names[i] for i in keep]
    batch_elements = [element[i] for i in keep]
    return DomainHeavyTopology(
        domain=domain,
        raw_to_batch=torch.as_tensor(raw_to_batch, dtype=torch.int64),
        batch_to_raw=batch_to_raw,
        topology=topology,
        atom_names=tuple(names),
        elements=tuple(element),
        is_backbone=torch.tensor([is_backbone_atom(n) for n in batch_names]),
        is_sidechain=torch.tensor(
            [
                (not is_backbone_atom(n)) and (not is_cap_atom(n)) and e != "H"
                for n, e in zip(batch_names, batch_elements)
            ]
        ),
        is_cap=torch.tensor([is_cap_atom(n) for n in batch_names]),
        vdw_radius=torch.tensor(
            [BONDI_VDW_RADII[e] for e in batch_elements], dtype=torch.float64
        ),
        residue_index=torch.as_tensor(a2r[keep], dtype=torch.int64),
        residue_index_raw=torch.as_tensor(a2r, dtype=torch.int64),
        chi=chi,
        dssp=dssp,
        rmsf=rmsf,
    )


def clear_cache() -> None:
    """Drop the per-domain cache. Used by tests that vary the scope."""
    load_domain_topology.cache_clear()
