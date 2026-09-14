"""Molecular topology derived from the PSF the simulation actually used.

Nothing here is inferred from distance. mdCATH embeds the CHARMM PSF of the
solvated system, and ``data/psf.py`` extracts the protein-internal records of any
section; this module turns those into the four things the heavy-atom work needs:

1. **nonbonded exclusions** for a clash metric -- 1-2 from ``NBOND``, 1-3 from
   ``NTHETA``, 1-4 from ``NPHI``, read from the force field rather than derived
   by walking the bond graph. For a ring the two disagree, and a clash metric
   that disagrees with the force field about which pairs are bonded is measuring
   something the simulation never saw.
2. **the hydrogen -> parent heavy atom map** for H2's effective-force
   aggregation. Every hydrogen has exactly one heavy neighbour in ``NBOND``; that
   is asserted, not assumed.
3. **disulfides**, as explicit ``SG-SG`` bond records. The brief forbids
   inferring them from distance and this makes that unnecessary.
4. **chi rotatability**, decided per residue *instance* by asking the bond graph
   whether cutting the rotation bond actually separates the distal atoms from the
   backbone. This one check subsumes two special cases that would otherwise need
   hand-coding:

   * **PRO** -- its ring keeps CG connected to N through CD, so cutting CB-CG
     separates nothing and the torsion is correctly marked non-rotatable;
   * **disulfide-bonded CYS** -- cutting CA-CB leaves SG connected to the *other*
     chain through S-S, so the distal component reaches another residue's
     backbone and the torsion is correctly marked non-rotatable.

   Neither is special-cased by name. Both fall out of asking the real topology.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..data.psf import parse_psf_section
from .chemistry import ChiDefinition, chi_definitions_for

__all__ = [
    "MolecularTopology",
    "ChiInstance",
    "build_topology",
    "bonded_exclusion_mask",
]


@dataclass
class MolecularTopology:
    """Protein-internal topology of one domain, in **protein atom indexing**.

    All index tensors address the domain's own protein atom array (the one the
    coordinate dataset uses), 0-based, before any heavy-atom filtering.

    Args:
        bonds / angles / dihedrals / impropers: ``[2|3|4, N]`` int64 from the PSF.
        num_atoms: length of the protein atom array these index into.
        hydrogen_parent: ``[num_atoms]`` int64. For a hydrogen, the index of its
            single bonded heavy atom; ``-1`` for every heavy atom and for any
            hydrogen without exactly one heavy neighbour (which is reported, not
            silently bridged).
        disulfides: ``[2, N]`` int64 SG-SG bonds between different residues.
        is_hydrogen: ``[num_atoms]`` bool.
    """

    bonds: Tensor
    angles: Tensor
    dihedrals: Tensor
    impropers: Tensor
    num_atoms: int
    hydrogen_parent: Tensor
    disulfides: Tensor
    is_hydrogen: Tensor

    @property
    def num_bonds(self) -> int:
        return int(self.bonds.shape[1])

    def neighbours(self) -> list[list[int]]:
        """Adjacency list over ``bonds``. Built on demand, not cached in a field."""
        adjacency: list[list[int]] = [[] for _ in range(self.num_atoms)]
        for a, b in self.bonds.t().tolist():
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency


@dataclass(frozen=True)
class ChiInstance:
    """One chi torsion of one concrete residue, with its support decided.

    Args:
        residue_index: row in the residue arrays.
        residue_name: canonical residue name.
        chi_index: 1-based.
        atom_indices: ``(a, b, c, d)`` protein atom indices of the four atoms.
        moving: atom indices that rotate when the torsion changes -- the
            connected component of ``d`` after cutting the ``b-c`` bond.
        supported: whether H1 may rotate this torsion.
        reason: why not, when ``supported`` is False. Always set in that case.
        periodicity / ambiguous: carried from the definition.
    """

    residue_index: int
    residue_name: str
    chi_index: int
    atom_indices: tuple[int, int, int, int]
    moving: tuple[int, ...]
    supported: bool
    reason: Optional[str]
    periodicity: int
    ambiguous: bool


def build_topology(
    psf_text: str,
    num_protein_atoms: int,
    atomic_number: Tensor,
) -> MolecularTopology:
    """Parse every needed PSF section and derive the hydrogen map and disulfides.

    Args:
        psf_text: the shard's ``psf`` dataset, decoded.
        num_protein_atoms: length of the protein coordinate array.
        atomic_number: ``[num_protein_atoms]`` int64, to identify hydrogens.

    Raises:
        ValueError: on a length mismatch, or if a hydrogen has other than exactly
            one heavy neighbour -- which would make the H2 aggregation in §6.6
            either lossy or double-counting, and is worth stopping for.
    """
    if int(atomic_number.shape[0]) != num_protein_atoms:
        raise ValueError(
            f"atomic_number has {int(atomic_number.shape[0])} entries for "
            f"{num_protein_atoms} protein atoms"
        )
    bonds = parse_psf_section(psf_text, "NBOND", num_protein_atoms)
    angles = parse_psf_section(psf_text, "NTHETA", num_protein_atoms)
    dihedrals = parse_psf_section(psf_text, "NPHI", num_protein_atoms)
    impropers = parse_psf_section(psf_text, "NIMPHI", num_protein_atoms)

    is_hydrogen = atomic_number == 1
    heavy_neighbours: list[list[int]] = [[] for _ in range(num_protein_atoms)]
    for a, b in bonds.t().tolist():
        if is_hydrogen[a] and not is_hydrogen[b]:
            heavy_neighbours[a].append(b)
        elif is_hydrogen[b] and not is_hydrogen[a]:
            heavy_neighbours[b].append(a)

    parent = torch.full((num_protein_atoms,), -1, dtype=torch.int64)
    orphans = []
    for index in range(num_protein_atoms):
        if not bool(is_hydrogen[index]):
            continue
        candidates = heavy_neighbours[index]
        if len(candidates) == 1:
            parent[index] = candidates[0]
        else:
            orphans.append((index, len(candidates)))
    if orphans:
        raise ValueError(
            f"{len(orphans)} hydrogen(s) do not have exactly one bonded heavy "
            f"atom, e.g. {orphans[:5]} as (atom_index, n_heavy_neighbours). "
            "Aggregating their force onto a parent would either lose it or count "
            "it twice; the topology must be fixed rather than worked around."
        )

    sulfur = atomic_number == 16
    pairs = bonds.t()
    disulfide_mask = sulfur[pairs[:, 0]] & sulfur[pairs[:, 1]]
    disulfides = pairs[disulfide_mask].t().contiguous()

    return MolecularTopology(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        num_atoms=num_protein_atoms,
        hydrogen_parent=parent,
        disulfides=disulfides,
        is_hydrogen=is_hydrogen,
    )


def bonded_exclusion_mask(
    topology: MolecularTopology,
    keep: Tensor,
    *,
    exclude_1_4: bool = False,
) -> tuple[Tensor, Tensor]:
    """``(excluded, is_1_4)``: ``[n, n]`` bool over the atoms selected by ``keep``.

    ``excluded`` marks pairs a nonbonded clash metric must skip: 1-2 from the
    bond list and 1-3 from the angle list, always; 1-4 from the dihedral list
    only when ``exclude_1_4``.

    The brief's primary counts 1-4 pairs **in** the nonbonded result and reports
    them separately as well, so the default is ``False`` and the second return
    value carries the 1-4 pairs for that separate tally. A 1-4 contact at
    van der Waals distance is a real eclipsed torsion, not a bookkeeping
    artefact, which is why it is not silently removed.

    Args:
        keep: ``[num_atoms]`` bool selecting the atoms to score (typically the
            heavy atoms). Output is indexed in the compacted numbering, i.e. row
            ``i`` is the ``i``-th True entry of ``keep``.
    """
    index = keep.nonzero(as_tuple=True)[0]
    n = int(index.numel())
    remap = torch.full((topology.num_atoms,), -1, dtype=torch.int64)
    remap[index] = torch.arange(n)

    excluded = torch.zeros((n, n), dtype=torch.bool)
    is_1_4 = torch.zeros((n, n), dtype=torch.bool)

    def mark(target: Tensor, a: Tensor, b: Tensor) -> None:
        ra, rb = remap[a], remap[b]
        ok = (ra >= 0) & (rb >= 0)
        target[ra[ok], rb[ok]] = True
        target[rb[ok], ra[ok]] = True

    mark(excluded, topology.bonds[0], topology.bonds[1])
    mark(excluded, topology.angles[0], topology.angles[2])
    mark(is_1_4, topology.dihedrals[0], topology.dihedrals[3])
    if exclude_1_4:
        excluded |= is_1_4
    # A 1-4 pair that is also 1-2 or 1-3 (rings do this) is not a 1-4 contact for
    # reporting purposes; the closer relation wins.
    is_1_4 = is_1_4 & ~excluded
    excluded.fill_diagonal_(True)
    return excluded, is_1_4


def _component_after_cut(
    adjacency: Sequence[Sequence[int]], start: int, cut: tuple[int, int]
) -> set[int]:
    """Atoms reachable from ``start`` once the undirected bond ``cut`` is removed."""
    blocked = {cut, (cut[1], cut[0])}
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if (current, neighbour) in blocked or neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen


def chi_instances(
    topology: MolecularTopology,
    *,
    residue_names: Sequence[str],
    atom_to_residue: Tensor,
    atom_names: Sequence[str],
    backbone_atom_indices: Tensor,
) -> list[ChiInstance]:
    """Resolve every chi of every residue against the real bond graph.

    This is where a chi definition earns the right to be used. For each torsion
    the function checks, in order:

    1. all four named atoms exist in this residue;
    2. ``A-B``, ``B-C`` and ``C-D`` are all real bonds in the PSF;
    3. cutting ``B-C`` separates ``D``'s component from ``A``'s -- if it does
       not, the torsion is inside a ring and is **not rotatable**;
    4. ``D``'s component contains no backbone atom -- if it does, rotating would
       move the backbone, which for a disulfide-bonded cysteine it would.

    A torsion failing any check is returned with ``supported=False`` and a
    reason. It is never silently dropped, because "this residue has no chi2" and
    "this residue's chi2 was rejected" are different facts and the report needs
    both.

    Args:
        backbone_atom_indices: atom indices that must not move. Passing the
            backbone of the **whole domain** (not just this residue) is what
            makes the disulfide case fail correctly.
    """
    adjacency = topology.neighbours()
    bond_set = {
        (a, b) for a, b in topology.bonds.t().tolist()
    } | {(b, a) for a, b in topology.bonds.t().tolist()}
    backbone = set(backbone_atom_indices.tolist())

    by_residue: dict[int, dict[str, int]] = {}
    for atom_index, residue_index in enumerate(atom_to_residue.tolist()):
        by_residue.setdefault(residue_index, {})[atom_names[atom_index]] = atom_index

    residue_of = {i: int(r) for i, r in enumerate(atom_to_residue.tolist())}
    out: list[ChiInstance] = []
    for residue_index, name in enumerate(residue_names):
        lookup = by_residue.get(residue_index, {})
        for definition in chi_definitions_for(name):
            out.append(
                _resolve_chi(
                    definition, residue_index, name, lookup, adjacency,
                    bond_set, backbone, residue_of,
                )
            )
    return out


def _resolve_chi(
    definition: ChiDefinition,
    residue_index: int,
    residue_name: str,
    lookup: dict[str, int],
    adjacency: Sequence[Sequence[int]],
    bond_set: set[tuple[int, int]],
    backbone: set[int],
    residue_of: dict[int, int],
) -> ChiInstance:
    def unsupported(reason: str, indices=(-1, -1, -1, -1)) -> ChiInstance:
        return ChiInstance(
            residue_index=residue_index, residue_name=residue_name,
            chi_index=definition.index, atom_indices=indices, moving=(),
            supported=False, reason=reason,
            periodicity=definition.periodicity, ambiguous=definition.ambiguous,
        )

    missing = [n for n in definition.atoms if n not in lookup]
    if missing:
        return unsupported(f"missing atom(s) {missing} in this residue")
    a, b, c, d = (lookup[n] for n in definition.atoms)

    for left, right in ((a, b), (b, c), (c, d)):
        if (left, right) not in bond_set:
            return unsupported(
                f"{definition.atoms} is not a bond path in the PSF: "
                f"atoms {left}-{right} are not bonded",
                (a, b, c, d),
            )

    distal = _component_after_cut(adjacency, d, (b, c))
    contaminated = sorted(distal & backbone)
    if contaminated or a in distal:
        # Cutting the bond separated nothing. Two different molecules produce
        # that, and naming the wrong one in the report is worse than not naming
        # either -- but they are easy to tell apart: an intra-residue ring closes
        # *within this residue's own atoms*, while a cross-link has to leave the
        # residue to get back. So re-run the traversal confined to this residue.
        # PRO closes; a disulfide-bonded CYS does not.
        own = {i for i, r in residue_of.items() if r == residue_index}
        confined = [
            [n for n in adjacency[i] if n in own] if i in own else []
            for i in range(len(adjacency))
        ]
        if a in _component_after_cut(confined, d, (b, c)):
            return unsupported(
                "cutting the rotation bond does not separate the distal atoms, and "
                "the path back closes inside this residue — the torsion is in a "
                "ring and rotating it would break the ring",
                (a, b, c, d),
            )
        return unsupported(
            f"the distal component reaches {len(contaminated)} backbone atom(s) "
            "and the path back leaves this residue — the side chain is "
            "cross-linked (a disulfide), so rotating would move another residue",
            (a, b, c, d),
        )

    return ChiInstance(
        residue_index=residue_index, residue_name=residue_name,
        chi_index=definition.index, atom_indices=(a, b, c, d),
        moving=tuple(sorted(distal)), supported=True, reason=None,
        periodicity=definition.periodicity, ambiguous=definition.ambiguous,
    )
