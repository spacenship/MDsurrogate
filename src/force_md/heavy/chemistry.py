"""Chemistry constants for the heavy-atom extension, with their sources.

Everything here is a **published constant or a naming convention**, fixed before
any H0/H1/H2 result was produced. Nothing in this file was chosen after seeing a
number, and nothing in it is an approximation standing in for a table the
repository does not have.

Two things live here and they are different in kind.

*Van der Waals radii* are a measurement someone else made. Bondi's 1964 values
are the ones AlphaFold, OpenFold and most structure-validation tools use, so a
clash rate computed with them is comparable with published ones.

*Chi definitions* are a **convention**: which four atoms name a torsion. They
cannot be measured, so they are written out. What can be checked -- and is, in
:mod:`force_md.heavy.topology` -- is that each named path is a real bond path in
the PSF the simulation actually used, and that rotating about its central bond
moves no backbone atom. A table with a transposed atom pair fails that check
rather than silently mislabelling every side chain.

**What is deliberately absent.** There are no ideal bond lengths, no ideal bond
angles and no rigid-group definitions, because this repository has none and
inventing them was forbidden. The H1 design does not need them: it rotates a
residue's *own* current side chain about its *own* measured chi axes, which
preserves every bond length and every bond angle by construction. See
:mod:`force_md.heavy.builder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

__all__ = [
    "BONDI_VDW_RADII",
    "VDW_RADIUS_SOURCE",
    "OVERLAP_THRESHOLDS",
    "SERIOUS_OVERLAP_ANGSTROM",
    "ChiDefinition",
    "CHI_DEFINITIONS",
    "CHI_PI_PERIODIC",
    "CHI_AMBIGUOUS",
    "RING_RESIDUES",
    "chi_definitions_for",
    "vdw_radius_table",
]

# --------------------------------------------------------------------------
# van der Waals radii
# --------------------------------------------------------------------------

VDW_RADIUS_SOURCE: Final[str] = (
    "Bondi, A. (1964) 'van der Waals Volumes and Radii', J. Phys. Chem. 68(3) "
    "441-451. The same values AlphaFold/OpenFold use for their steric-clash "
    "term, so a rate computed here is comparable with published ones."
)

#: Element symbol -> van der Waals radius in angstrom. Only the five elements
#: mdCATH contains (see ``residue_constants.ELEMENT_TO_Z``); an element outside
#: this table is a data error and is raised on rather than defaulted.
BONDI_VDW_RADII: Final[dict[str, float]] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}

#: Overlap depth thresholds, in angstrom, for
#: ``o_ij = r_i + r_j - d_ij``.
#:
#: ``serious`` at 0.4 A is the primary and is **fixed here**, before any result:
#: it is the threshold MolProbity uses for a reportable clash and the one the
#: brief names. The other two are a sensitivity band, reported in an appendix and
#: never used to choose an arm.
OVERLAP_THRESHOLDS: Final[dict[str, float]] = {
    "any": 0.0,
    "mild": 0.2,
    "serious": 0.4,
}

SERIOUS_OVERLAP_ANGSTROM: Final[float] = OVERLAP_THRESHOLDS["serious"]


def vdw_radius_table(elements: tuple[str, ...]) -> list[float]:
    """Radii for a tuple of element symbols, raising on anything unknown."""
    missing = sorted({e for e in elements if e not in BONDI_VDW_RADII})
    if missing:
        raise KeyError(
            f"no Bondi van der Waals radius for element(s) {missing}. "
            "Guessing one would put an invented number into a clash rate; add "
            "the published value to BONDI_VDW_RADII with its source instead."
        )
    return [BONDI_VDW_RADII[e] for e in elements]


# --------------------------------------------------------------------------
# chi torsions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChiDefinition:
    """One side-chain torsion, named by the four atoms that define it.

    Args:
        residue: canonical CHARMM residue name, as
            ``residue_constants.canonical_resname`` returns.
        index: 1-based chi number.
        atoms: ``(A, B, C, D)`` atom names. The torsion is the dihedral
            ``A-B-C-D`` and the rotation is about the **B-C** bond, which is
            :attr:`rotation_bond`.
        periodicity: 1 for an ordinary torsion, 2 where the terminal group is
            symmetric under a half turn and the two states are the *same
            structure* (ASP chi2, GLU chi3, PHE chi2, TYR chi2). A periodicity-2
            torsion must be compared modulo 180 degrees or the error is
            meaningless.
        ambiguous: the terminal atoms are chemically distinct but routinely
            indistinguishable in a structure (ASN chi2, GLN chi3, HIS chi2).
            Recorded rather than acted on: mdCATH is simulated, so the assignment
            is whatever CHARMM used and is self-consistent.
        source: where the convention comes from.
    """

    residue: str
    index: int
    atoms: tuple[str, str, str, str]
    periodicity: int = 1
    ambiguous: bool = False
    source: str = "IUPAC-IUB (1970) side-chain torsion convention, CHARMM naming"

    @property
    def rotation_bond(self) -> tuple[str, str]:
        """The bond the torsion rotates about."""
        return self.atoms[1], self.atoms[2]


_PI_PERIODIC: Final[set[tuple[str, int]]] = {
    ("ASP", 2),   # OD1 / OD2 are equivalent
    ("GLU", 3),   # OE1 / OE2
    ("PHE", 2),   # CD1/CD2 and CE1/CE2 -- the ring is symmetric about CB-CG
    ("TYR", 2),   # same ring symmetry
}

_AMBIGUOUS: Final[set[tuple[str, int]]] = {
    ("ASN", 2),   # OD1 vs ND2 -- different elements, routinely swapped in X-ray
    ("GLN", 3),   # OE1 vs NE2
    ("HIS", 2),   # ND1 vs CD2, and the protonation state
}

#: Residues whose side chain contains a ring. Listed for documentation and for
#: the ring-planarity sanity check; the *rotatability* of a torsion is decided by
#: the PSF bond graph, not by this set, because PRO's ring makes its chi
#: non-rotatable while HIS/PHE/TRP/TYR rings sit distal to their chi bonds and
#: rotate perfectly well.
RING_RESIDUES: Final[frozenset[str]] = frozenset(
    {"PRO", "HIS", "PHE", "TRP", "TYR"}
)


def _chi(residue: str, index: int, *atoms: str) -> ChiDefinition:
    return ChiDefinition(
        residue=residue,
        index=index,
        atoms=(atoms[0], atoms[1], atoms[2], atoms[3]),
        periodicity=2 if (residue, index) in _PI_PERIODIC else 1,
        ambiguous=(residue, index) in _AMBIGUOUS,
    )


#: Chi torsions per residue, in CHARMM atom naming.
#:
#: Two CHARMM-specific points, both verified against ``SIDECHAIN_HEAVY_ATOMS``
#: and against a real shard rather than carried over from PDB convention:
#:
#: * **ILE chi2 ends at ``CD``**, not ``CD1``. CHARMM names isoleucine's terminal
#:   carbon ``CD``; using the PDB's ``CD1`` would silently drop every ILE chi2.
#: * **Histidine is ``HSD``** in mdCATH and canonicalises to ``HIS`` through
#:   ``CHARMM_RESNAME_ALIASES`` before lookup here.
#:
#: ALA and GLY have no rotatable side chain and are absent by design, not by
#: omission. PRO is listed because its torsions exist and are measurable; the
#: topology check will find that they are **not rotatable** (the ring keeps the
#: distal atoms connected to the backbone) and mark them unsupported for H1.
CHI_DEFINITIONS: Final[dict[str, tuple[ChiDefinition, ...]]] = {
    "ARG": (
        _chi("ARG", 1, "N", "CA", "CB", "CG"),
        _chi("ARG", 2, "CA", "CB", "CG", "CD"),
        _chi("ARG", 3, "CB", "CG", "CD", "NE"),
        _chi("ARG", 4, "CG", "CD", "NE", "CZ"),
    ),
    "ASN": (
        _chi("ASN", 1, "N", "CA", "CB", "CG"),
        _chi("ASN", 2, "CA", "CB", "CG", "OD1"),
    ),
    "ASP": (
        _chi("ASP", 1, "N", "CA", "CB", "CG"),
        _chi("ASP", 2, "CA", "CB", "CG", "OD1"),
    ),
    "CYS": (
        _chi("CYS", 1, "N", "CA", "CB", "SG"),
    ),
    "GLN": (
        _chi("GLN", 1, "N", "CA", "CB", "CG"),
        _chi("GLN", 2, "CA", "CB", "CG", "CD"),
        _chi("GLN", 3, "CB", "CG", "CD", "OE1"),
    ),
    "GLU": (
        _chi("GLU", 1, "N", "CA", "CB", "CG"),
        _chi("GLU", 2, "CA", "CB", "CG", "CD"),
        _chi("GLU", 3, "CB", "CG", "CD", "OE1"),
    ),
    "HIS": (
        _chi("HIS", 1, "N", "CA", "CB", "CG"),
        _chi("HIS", 2, "CA", "CB", "CG", "ND1"),
    ),
    "ILE": (
        _chi("ILE", 1, "N", "CA", "CB", "CG1"),
        _chi("ILE", 2, "CA", "CB", "CG1", "CD"),   # CHARMM CD, not PDB CD1
    ),
    "LEU": (
        _chi("LEU", 1, "N", "CA", "CB", "CG"),
        _chi("LEU", 2, "CA", "CB", "CG", "CD1"),
    ),
    "LYS": (
        _chi("LYS", 1, "N", "CA", "CB", "CG"),
        _chi("LYS", 2, "CA", "CB", "CG", "CD"),
        _chi("LYS", 3, "CB", "CG", "CD", "CE"),
        _chi("LYS", 4, "CG", "CD", "CE", "NZ"),
    ),
    "MET": (
        _chi("MET", 1, "N", "CA", "CB", "CG"),
        _chi("MET", 2, "CA", "CB", "CG", "SD"),
        _chi("MET", 3, "CB", "CG", "SD", "CE"),
    ),
    "PHE": (
        _chi("PHE", 1, "N", "CA", "CB", "CG"),
        _chi("PHE", 2, "CA", "CB", "CG", "CD1"),
    ),
    "PRO": (
        _chi("PRO", 1, "N", "CA", "CB", "CG"),
        _chi("PRO", 2, "CA", "CB", "CG", "CD"),
    ),
    "SER": (
        _chi("SER", 1, "N", "CA", "CB", "OG"),
    ),
    "THR": (
        _chi("THR", 1, "N", "CA", "CB", "OG1"),
    ),
    "TRP": (
        _chi("TRP", 1, "N", "CA", "CB", "CG"),
        _chi("TRP", 2, "CA", "CB", "CG", "CD1"),
    ),
    "TYR": (
        _chi("TYR", 1, "N", "CA", "CB", "CG"),
        _chi("TYR", 2, "CA", "CB", "CG", "CD1"),
    ),
    "VAL": (
        _chi("VAL", 1, "N", "CA", "CB", "CG1"),
    ),
}

#: ``{(residue, chi_index)}`` where the torsion is symmetric under 180 degrees.
CHI_PI_PERIODIC: Final[frozenset[tuple[str, int]]] = frozenset(_PI_PERIODIC)

#: ``{(residue, chi_index)}`` where the terminal atoms are routinely swapped.
CHI_AMBIGUOUS: Final[frozenset[tuple[str, int]]] = frozenset(_AMBIGUOUS)

#: Largest chi index in the table, so a fixed-width tensor can be allocated.
MAX_CHI: Final[int] = max(len(v) for v in CHI_DEFINITIONS.values())


def chi_definitions_for(residue: str) -> tuple[ChiDefinition, ...]:
    """Chi torsions of a canonical residue name; empty for ALA, GLY and UNK.

    ``UNK`` returns empty rather than guessing: an unknown residue has no known
    side-chain topology, and rotating an unknown side chain about a guessed axis
    is exactly the invention this module exists to avoid.
    """
    return CHI_DEFINITIONS.get(residue, ())
