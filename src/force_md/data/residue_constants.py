"""Residue and atom vocabularies, in the CHARMM naming that mdCATH actually uses.

Verified against the shipped topology (``top_all22star_prot``) by reading the
per-atom ``resname``/``element`` arrays and the ``pdbProteinAtoms`` text of real
shards. Three CHARMM-specific facts drive this module, and each of them would be
a silent bug if the PDB naming were assumed instead:

1. Histidine is stored by protonation state: ``HSD``/``HSE``/``HSP``, never
   ``HIS``. A PDB-only table maps every histidine to "unknown".
2. Terminal capping groups are merged into the first/last residue: the N-terminal
   acetyl contributes ``CAY HY1 HY2 HY3 CY OY`` and the C-terminal N-methylamide
   contributes ``NT HNT CAT HT1 HT2 HT3``. ``CAY`` is *not* the alpha carbon --
   selecting the backbone by element would pick it up and corrupt the residue
   frame.
3. Isoleucine's delta carbon is ``CD``, not the PDB's ``CD1``.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Residue types
# --------------------------------------------------------------------------

#: Canonical residue alphabet. Index into this list is the ``residue_type`` id.
RESIDUE_TYPES: Final[tuple[str, ...]] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
)
RESIDUE_TYPE_TO_ID: Final[dict[str, int]] = {r: i for i, r in enumerate(RESIDUE_TYPES)}
UNK_RESIDUE_ID: Final[int] = RESIDUE_TYPE_TO_ID["UNK"]
NUM_RESIDUE_TYPES: Final[int] = len(RESIDUE_TYPES)

#: CHARMM residue name -> canonical residue name. Protonation/tautomer variants
#: collapse onto the canonical residue; the distinction is not modelled.
CHARMM_RESNAME_ALIASES: Final[dict[str, str]] = {
    "HSD": "HIS",  # neutral, proton on ND1 (CHARMM default)
    "HSE": "HIS",  # neutral, proton on NE2
    "HSP": "HIS",  # doubly protonated, +1
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",  # AMBER spellings, for robustness
    "CYX": "CYS",  # disulfide-bonded cysteine
    "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}

RESNAME_TO_ONE_LETTER: Final[dict[str, str]] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
}


def canonical_resname(resname: str) -> str:
    """Map a raw (CHARMM) residue name onto the canonical alphabet.

    Unrecognised residues become ``"UNK"`` rather than raising: mdCATH contains
    only standard protein residues, but a nonstandard residue must degrade to a
    masked, typed node instead of crashing the pipeline.
    """
    key = resname.strip().upper()
    key = CHARMM_RESNAME_ALIASES.get(key, key)
    return key if key in RESIDUE_TYPE_TO_ID else "UNK"


def residue_type_id(resname: str) -> int:
    """Canonical residue-type index of a raw residue name."""
    return RESIDUE_TYPE_TO_ID[canonical_resname(resname)]


def one_letter(resname: str) -> str:
    """One-letter code used to build the ESM-2 input sequence."""
    return RESNAME_TO_ONE_LETTER[canonical_resname(resname)]


# --------------------------------------------------------------------------
# Atoms
# --------------------------------------------------------------------------

#: Backbone atoms that define the residue frame plus the carbonyl oxygen.
#: ``N``, ``CA`` and ``C`` are required for a valid frame; ``O`` is not.
FRAME_ATOM_NAMES: Final[tuple[str, str, str]] = ("N", "CA", "C")
BACKBONE_ATOM_NAMES: Final[frozenset[str]] = frozenset({"N", "CA", "C", "O", "OXT"})

#: CHARMM terminal patch atoms, merged into the first/last residue in mdCATH.
#: They belong to no standard residue template and are neither backbone nor
#: side chain, so they are flagged separately.
NTERM_CAP_ATOM_NAMES: Final[frozenset[str]] = frozenset(
    {"CAY", "HY1", "HY2", "HY3", "CY", "OY"}
)
CTERM_CAP_ATOM_NAMES: Final[frozenset[str]] = frozenset(
    {"NT", "HNT", "CAT", "HT1", "HT2", "HT3"}
)
CAP_ATOM_NAMES: Final[frozenset[str]] = NTERM_CAP_ATOM_NAMES | CTERM_CAP_ATOM_NAMES

#: Side-chain heavy atoms per canonical residue, in CHARMM naming.
SIDECHAIN_HEAVY_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD"),  # CHARMM: CD, not the PDB's CD1
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
    "UNK": ("CB",),
}

#: Element symbol -> atomic number, for the elements mdCATH contains.
ELEMENT_TO_Z: Final[dict[str, int]] = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
Z_TO_ELEMENT: Final[dict[int, str]] = {v: k for k, v in ELEMENT_TO_Z.items()}

#: Atomic numbers considered "heavy" (i.e. everything except hydrogen).
HYDROGEN_Z: Final[int] = 1


def _build_atom_name_vocabulary() -> tuple[str, ...]:
    """Deterministic, sorted atom-name vocabulary with an explicit UNK at 0."""
    names: set[str] = set(BACKBONE_ATOM_NAMES) | set(CAP_ATOM_NAMES)
    for atoms in SIDECHAIN_HEAVY_ATOMS.values():
        names.update(atoms)
    # Polar/aliphatic hydrogens that appear in the all-atom representation.
    names.update({"HN", "HA", "HA1", "HA2"})
    for stem in ("HB", "HG", "HD", "HE", "HZ", "HH"):
        for suffix in ("", "1", "2", "3", "11", "12", "13", "21", "22", "23"):
            names.add(stem + suffix)
    return ("<unk>",) + tuple(sorted(names))


ATOM_NAMES: Final[tuple[str, ...]] = _build_atom_name_vocabulary()
ATOM_NAME_TO_ID: Final[dict[str, int]] = {n: i for i, n in enumerate(ATOM_NAMES)}
UNK_ATOM_NAME_ID: Final[int] = 0
NUM_ATOM_NAMES: Final[int] = len(ATOM_NAMES)


def atom_name_id(name: str) -> int:
    """Vocabulary index of an atom name, ``0`` if unseen."""
    return ATOM_NAME_TO_ID.get(name.strip().upper(), UNK_ATOM_NAME_ID)


def is_cap_atom(name: str) -> bool:
    """True for CHARMM terminal-patch atoms merged into a terminal residue."""
    return name.strip().upper() in CAP_ATOM_NAMES


def is_backbone_atom(name: str) -> bool:
    """True for backbone atoms. Excludes cap atoms such as ``CAY``/``CAT``."""
    return name.strip().upper() in BACKBONE_ATOM_NAMES
