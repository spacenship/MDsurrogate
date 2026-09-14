"""Residue identity follows first appearance in the source atom topology.

Residue numbers are labels, not sequence positions (e.g. 72, 860, 73).
Chain is part of the identity; stable grouping also supports interleaved atoms.
"""
import numpy as np


def residue_order(resid, chain):
    """Return original residue labels, first atom indices and atom-to-residue IDs."""
    resid, chain = np.asarray(resid), np.asarray(chain)
    if resid.ndim != 1 or chain.shape != resid.shape:
        raise ValueError("resid and chain must be matching 1-D arrays")
    lookup, first = {}, []
    inverse = np.empty(len(resid), dtype=np.int64)
    for atom, (ch, label) in enumerate(zip(chain.tolist(), resid.tolist())):
        key = (ch, label)
        if key not in lookup:
            lookup[key] = len(first)
            first.append(atom)
        inverse[atom] = lookup[key]
    first = np.asarray(first, dtype=np.int64)
    return resid[first], first, inverse
