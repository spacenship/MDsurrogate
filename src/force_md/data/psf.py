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

__all__ = ["parse_psf_bonds"]

_SECTION = re.compile(r"^\s*(\d+)\s*!(\w+)")


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
