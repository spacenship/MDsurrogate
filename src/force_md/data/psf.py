"""CHARMM PSF bond parsing.

mdCATH embeds the PSF of the **full solvated system**, not of the protein-only
coordinate array it ships. On a domain with 1462 protein atoms the PSF declares
41926 atoms and 28397 bonds, so PSF indices cannot be used directly. Verified on
real shards: the protein atoms are the *leading* block of the PSF, and the bond
records for the first residue reproduce the CHARMM cap ordering
(``CAY HY1 HY2 HY3 CY ...``) exactly. Restricting to bonds whose two endpoints
are both below ``num_protein_atoms`` therefore yields the protein bond graph.

This is the authoritative bond source and should be preferred over the
distance heuristic in :func:`force_md.graph.edges.build_covalent_bonds`.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

__all__ = ["parse_psf_bonds", "parse_psf_section", "PSF_SECTION_ARITY"]

_SECTION = re.compile(r"^\s*(\d+)\s*!(\w+)")

#: Indices per record, per CHARMM PSF section. Measured on a real mdCATH shard
#: (``1ad3A02``): ``NBOND 22402``, ``NTHETA 15266``, ``NPHI 8222``,
#: ``NIMPHI 523``, ``NCRTERM 24``.
#:
#: ``NTHETA`` and ``NPHI`` matter because they are the **authoritative** 1-3 and
#: 1-4 relations. Deriving those by walking the bond graph gives the same answer
#: for a well-formed molecule and a subtly different one for a ring, and a clash
#: metric that disagrees with the force field about which pairs are bonded is
#: measuring the wrong thing.
PSF_SECTION_ARITY: dict[str, int] = {
    "NBOND": 2,
    "NTHETA": 3,
    "NPHI": 4,
    "NIMPHI": 4,
}


def parse_psf_bonds(psf_text: str, num_protein_atoms: int) -> Tensor:
    """Extract protein-internal bonds from a CHARMM PSF.

    Args:
        psf_text: the ``psf`` dataset of an mdCATH shard, decoded to ``str``.
        num_protein_atoms: length of the protein-only coordinate array. Bonds
            touching any atom at or beyond this index belong to solvent/ions and
            are dropped.

    Returns:
        ``[2, E]`` int64, **0-based**, each undirected bond listed once.

    Raises:
        ValueError: if no ``!NBOND`` section exists or it is truncated. Failing
            loudly matters: a silently empty bond list would train a model with
            no covalent topology at all.
    """
    lines = psf_text.splitlines()
    start = count = None
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m and m.group(2).upper().startswith("NBOND"):
            count = int(m.group(1))
            start = i + 1
            break
    if start is None:
        raise ValueError("PSF contains no !NBOND section")

    need = 2 * count
    values: list[int] = []
    for line in lines[start:]:
        if len(values) >= need:
            break
        if "!" in line:  # next section began early
            break
        values.extend(int(tok) for tok in line.split())
    if len(values) < need:
        raise ValueError(
            f"PSF !NBOND declares {count} bonds ({need} indices) but only "
            f"{len(values)} were readable"
        )

    pairs = torch.tensor(values[:need], dtype=torch.int64).view(count, 2) - 1  # 1- -> 0-based
    keep = (pairs >= 0).all(1) & (pairs < num_protein_atoms).all(1)
    return pairs[keep].t().contiguous()


def parse_psf_section(
    psf_text: str, section: str, num_protein_atoms: int
) -> Tensor:
    """Extract one PSF section, restricted to protein-internal records.

    Generalises :func:`parse_psf_bonds` to the sections Phase 1.6's heavy-atom
    work needs: ``NTHETA`` (angles, the 1-3 relations) and ``NPHI`` (dihedrals,
    the 1-4 relations) are what a nonbonded clash metric must exclude, and
    ``NIMPHI`` (impropers) is the force field's own statement of which centres
    are chiral or planar.

    The same index caveat as ``parse_psf_bonds`` applies and for the same
    reason: mdCATH embeds the PSF of the **full solvated system**, so a record
    touching any atom at or beyond ``num_protein_atoms`` belongs to solvent or
    ions and is dropped.

    Args:
        psf_text: the ``psf`` dataset of an mdCATH shard, decoded to ``str``.
        section: one of :data:`PSF_SECTION_ARITY`, with or without a trailing
            colon (CHARMM writes ``!NBOND:`` in some versions and ``!NBOND`` in
            others; both are accepted).
        num_protein_atoms: length of the protein-only coordinate array.

    Returns:
        ``[arity, N]`` int64, **0-based**.

    Raises:
        KeyError: for an unknown section name.
        ValueError: if the section is absent or truncated. Absent is an error
            rather than an empty tensor: a silently empty angle list would
            remove every 1-3 exclusion and report a protein as one enormous
            clash.
    """
    key = section.rstrip(":").upper()
    if key not in PSF_SECTION_ARITY:
        raise KeyError(
            f"unknown PSF section {section!r}; known: {sorted(PSF_SECTION_ARITY)}"
        )
    arity = PSF_SECTION_ARITY[key]

    lines = psf_text.splitlines()
    start = count = None
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m and m.group(2).upper().rstrip(":").startswith(key):
            count = int(m.group(1))
            start = i + 1
            break
    if start is None:
        raise ValueError(f"PSF contains no !{key} section")

    need = arity * count
    values: list[int] = []
    for line in lines[start:]:
        if len(values) >= need:
            break
        if "!" in line:  # the next section began early
            break
        values.extend(int(tok) for tok in line.split())
    if len(values) < need:
        raise ValueError(
            f"PSF !{key} declares {count} records ({need} indices) but only "
            f"{len(values)} were readable"
        )

    records = torch.tensor(values[:need], dtype=torch.int64).view(count, arity) - 1
    keep = (records >= 0).all(1) & (records < num_protein_atoms).all(1)
    return records[keep].t().contiguous()
