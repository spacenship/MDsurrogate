"""Side-chain construction by rotating a residue's **own** geometry.

This is the module that replaces the brief's `HeavyAtomKinematicBuilder`, and the
reason it is different is worth stating once, here, where it applies.

A canonical builder places atoms from residue type + frame + torsions using ideal
bond lengths, ideal bond angles and a rigid-group tree. This repository has none
of those tables, and inventing them was ruled out. So instead of *building* a
side chain, this module **rotates the one the residue already has**: the atoms
distal to each chi bond are turned about that bond's measured axis.

What that buys, and it is not a small thing:

* every bond length and every bond angle in the residue is preserved **exactly**,
  because a rotation about a bond axis is a rigid motion of the distal fragment
  and cannot change either. Chirality and ring planarity likewise.
* no ideal-geometry constant is needed, so nothing is invented.
* the residue's own CHARMM equilibrium geometry is carried through, which is the
  geometry the simulation actually sampled.

What it costs, stated so no table can misread it: bond lengths, bond angles,
chirality and ring planarity are **construction-invariant** here. H1 does not get
to claim them. What H1 can move is chi accuracy, side-chain RMSD, packing and
steric overlap.

**Ordering matters and is handled.** Rotating chi1 must carry chi2's axis with
it. The moving set of chi1 contains the whole of chi2's, so applying torsions in
increasing chi index and re-reading each axis from the *already rotated*
coordinates gives the correct nested behaviour. ``test_rotating_chi1_moves_the_
chi2_axis`` pins it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..geometry.torsions import dihedral_angle, wrap_to_pi
from .topology import ChiInstance

__all__ = [
    "chi_values",
    "rotate_about_axis",
    "apply_chi_deltas",
    "set_chi_values",
    "supported_chi",
    "chi_delta_to_target",
]


def supported_chi(instances: Sequence[ChiInstance]) -> list[ChiInstance]:
    """The rotatable subset, in the order rotations must be applied.

    Sorted by ``(residue_index, chi_index)`` so a residue's chi1 is always
    applied before its chi2.
    """
    return sorted(
        (c for c in instances if c.supported),
        key=lambda c: (c.residue_index, c.chi_index),
    )


def chi_values(positions: Tensor, instances: Sequence[ChiInstance]) -> Tensor:
    """``[len(instances)]`` torsion angles in radians, from real coordinates.

    Unsupported instances get NaN rather than 0: a torsion that cannot be
    rotated still *has* a value, but a caller that reads it as a prediction
    target should be stopped by the NaN rather than handed a plausible zero.
    """
    if not instances:
        return positions.new_zeros(0)
    out = positions.new_full((len(instances),), float("nan"))
    usable = [i for i, c in enumerate(instances) if c.supported]
    if not usable:
        return out
    index = torch.tensor(
        [instances[i].atom_indices for i in usable],
        dtype=torch.int64, device=positions.device,
    )
    out[torch.tensor(usable, device=positions.device)] = dihedral_angle(
        positions[index[:, 0]], positions[index[:, 1]],
        positions[index[:, 2]], positions[index[:, 3]],
    )
    return out


def rotate_about_axis(
    points: Tensor, origin: Tensor, axis: Tensor, angle: Tensor, *, eps: float = 1e-12
) -> Tensor:
    """Rodrigues rotation of ``[N, 3]`` points about per-point axes.

    Args:
        points / origin / axis: ``[N, 3]``. ``axis`` need not be normalised.
        angle: ``[N]`` radians.

    Rodrigues rather than building a matrix per point: it is the same arithmetic
    with a third of the memory traffic, and it degrades gracefully when the axis
    is near zero-length (a degenerate bond), where the ``clamp`` leaves the point
    where it was instead of producing NaN.
    """
    unit = axis / axis.norm(dim=-1, keepdim=True).clamp(min=eps)
    relative = points - origin
    # Take the angle to the coordinates' dtype *before* the trigonometry. A
    # float32 angle against float64 coordinates loses seven digits in cos/sin and
    # then promotes, so the error survives into the result looking like float64
    # precision: cos(pi/2) comes back as -4.4e-8 rather than -6.1e-17, and a
    # 90-degree rotation misses by 4e-8 A per angstrom of arm.
    angle = angle.to(points.dtype)
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    dot = (unit * relative).sum(-1, keepdim=True)
    rotated = (
        relative * cos
        + torch.linalg.cross(unit, relative, dim=-1) * sin
        + unit * dot * (1.0 - cos)
    )
    return origin + rotated


def _apply(
    positions: Tensor,
    instances: Sequence[ChiInstance],
    angles: Tensor,
    *,
    absolute: bool,
) -> Tensor:
    """Shared core of :func:`apply_chi_deltas` and :func:`set_chi_values`."""
    if len(instances) != int(angles.shape[0]):
        raise ValueError(
            f"{len(instances)} chi instances but {int(angles.shape[0])} angles"
        )
    out = positions.clone()
    order = sorted(
        (i for i, c in enumerate(instances) if c.supported),
        key=lambda i: (instances[i].residue_index, instances[i].chi_index),
    )
    # Torsions of the same chi index touch disjoint residues, so a whole level
    # goes in one vectorised step; levels must stay ordered because chi1's moving
    # set contains chi2's.
    by_level: dict[int, list[int]] = defaultdict(list)
    for i in order:
        by_level[instances[i].chi_index].append(i)

    for level in sorted(by_level):
        members = by_level[level]
        rows, per_atom_slot = [], []
        for slot, i in enumerate(members):
            moving = instances[i].moving
            rows.extend(moving)
            per_atom_slot.extend([slot] * len(moving))
        if not rows:
            continue
        atom_index = torch.tensor(rows, dtype=torch.int64, device=out.device)
        slot_index = torch.tensor(per_atom_slot, dtype=torch.int64, device=out.device)
        b = torch.tensor(
            [instances[i].atom_indices[1] for i in members],
            dtype=torch.int64, device=out.device,
        )
        c = torch.tensor(
            [instances[i].atom_indices[2] for i in members],
            dtype=torch.int64, device=out.device,
        )
        # Read the axis from the *current* coordinates, so a rotation applied at
        # an earlier level has already moved it.
        axis = (out[c] - out[b])[slot_index]
        origin = out[c][slot_index]
        if absolute:
            current = chi_values(out, [instances[i] for i in members])
            step = wrap_to_pi(angles[torch.tensor(members, device=out.device)] - current)
        else:
            step = angles[torch.tensor(members, device=out.device)]
        out[atom_index] = rotate_about_axis(
            out[atom_index], origin, axis, step[slot_index]
        )
    return out


def apply_chi_deltas(
    positions: Tensor, instances: Sequence[ChiInstance], deltas: Tensor
) -> Tensor:
    """Rotate each supported torsion **by** ``deltas`` radians.

    Unsupported instances are skipped; their entry in ``deltas`` is ignored
    rather than silently applied to some other axis.
    """
    return _apply(positions, instances, deltas, absolute=False)


def set_chi_values(
    positions: Tensor, instances: Sequence[ChiInstance], targets: Tensor
) -> Tensor:
    """Rotate each supported torsion **to** ``targets`` radians.

    Used by the H1 sanity check that setting the ground-truth chi reduces
    side-chain RMSD, and by the round-trip tests. The step is taken through
    :func:`wrap_to_pi`, so asking for ``-179`` from ``+179`` turns two degrees
    rather than 358.
    """
    return _apply(positions, instances, targets, absolute=True)


def chi_delta_to_target(
    current: Tensor, target: Tensor, periodicity: Tensor
) -> Tensor:
    """Signed shortest rotation from ``current`` to ``target``, symmetry-aware.

    For a periodicity-2 torsion (ASP chi2, GLU chi3, PHE chi2, TYR chi2) the two
    states half a turn apart are the *same structure*, so the shortest step is
    taken modulo pi. Using the full circle there would train the head to chase a
    flip that changes nothing, and would report a 180-degree error for a
    perfect prediction.
    """
    period = 2.0 * torch.pi / periodicity.to(current.dtype)
    difference = target - current
    return difference - period * torch.round(difference / period)
