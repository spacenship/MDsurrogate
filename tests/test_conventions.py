"""Pin the geometric conventions the rest of the project asserts against.

These guard the *test harness itself*. A dihedral helper with an inverted sign
reports a mirrored structure as correct, so every downstream chirality test
would pass on a left-handed protein.
"""

from __future__ import annotations

import pytest

from conftest import dihedral, reference_dihedral_case


@pytest.mark.parametrize("want", [0.0, 30.0, -30.0, 90.0, -120.0, 180.0])
def test_dihedral_matches_analytic_value(want):
    got = dihedral(*reference_dihedral_case(want))
    assert abs((got - want + 180.0) % 360.0 - 180.0) < 1e-4, f"{got} != {want}"


def test_dihedral_is_antisymmetric_under_reflection():
    """Mirroring must flip the sign; this is what makes chirality testable."""
    p0, p1, p2, p3 = reference_dihedral_case(47.0)
    mirror = lambda p: p * p.new_tensor([1.0, -1.0, 1.0])  # noqa: E731
    assert abs(dihedral(p0, p1, p2, p3) + dihedral(*map(mirror, (p0, p1, p2, p3)))) < 1e-4
