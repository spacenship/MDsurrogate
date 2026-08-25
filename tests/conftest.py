"""Shared test utilities.

The dihedral sign convention lives here, with a self-test, because getting it
backwards silently turns every chirality assertion into its mirror image -- and
protein chirality is a modelling requirement of this project, not a symmetry we
are free to average over.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> float:
    """Signed dihedral p0-p1-p2-p3 in degrees, IUPAC convention.

    ``b0`` points from ``p1`` to ``p0``; flipping that sign rotates the result by
    180 degrees and inverts apparent handedness. See :func:`test_dihedral_convention`.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / torch.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2(float(torch.linalg.cross(b1, v) @ w), float(v @ w)))


def ca_pseudo_torsions(ca: Tensor) -> list[float]:
    """Virtual CA(i)..CA(i+3) torsions. ~ +50 deg for a right-handed alpha helix."""
    return [dihedral(ca[i], ca[i + 1], ca[i + 2], ca[i + 3]) for i in range(len(ca) - 3)]


def reference_dihedral_case(want_degrees: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Four points whose dihedral is analytically ``want_degrees``."""
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 0.0, 1.0])
    p0 = p1 + torch.tensor([1.0, 0.0, 0.0])
    r = math.radians(want_degrees)
    p3 = p2 + torch.tensor([math.cos(r), math.sin(r), 0.0])
    return p0, p1, p2, p3
