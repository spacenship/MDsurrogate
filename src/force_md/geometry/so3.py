"""SO(3) utilities: relative rotations, geodesic distance, log/exp map, 6D charts.

Phase 1 had no need for these -- it predicts forces, which are plain vectors. A
transition between two structures is a rotation, so Phase 1.5 needs them, and
every one of them has a numerically delicate region that a naive implementation
gets wrong exactly where a good model lives.

**Convention, fixed here and used everywhere downstream.** A residue frame
``R_i`` has the *local axes as its columns* (Phase 1's convention, see
:mod:`force_md.geometry.frames`), so ``R_i`` maps local vectors to global ones.
The relative rotation from frame ``A`` to frame ``B`` is therefore

    R_rel = R_A^T R_B                    (right multiplication)
    R_B   = R_A R_rel

and ``R_rel`` is expressed **in A's local frame**. Getting this backwards
transposes every rotation target and produces a model that trains to a mirror of
the intended objective, so :func:`relative_rotation` is the only place the
product is written and the ordering is asserted by test.

**Why the geodesic angle uses ``atan2``.** The angle of a rotation satisfies
``cos(theta) = (tr(R) - 1) / 2``, so ``arccos`` is the obvious implementation and
the wrong one: near ``theta = 0`` -- where an accurate prediction sits --
``arccos`` has an infinite derivative and float32 rounding alone turns a perfect
rotation into a fraction of a degree. Writing ``theta = atan2(|v|, cos theta)``
with ``v = vee((R - R^T)/2)``, ``|v| = sin(theta)``, is well conditioned over the
whole range including both endpoints. This is the same argument the Phase 1
metrics module makes for angular error between vectors.

**Why the rotation target is a matrix, not a chart.** Targets and losses use
``R_rel`` itself and the chart-free geodesic angle. Charts (axis-angle, 6D,
quaternion) are needed only where a *network* must emit a rotation, which is a
Checkpoint 5 question; the target definition does not commit to one. Both charts
are provided here because the probe head needs one: 6D
(Zhou et al. 2019) is continuous everywhere and is the recommended choice for a
regression head, while the log map is the natural chart for reporting and for a
small-angle prior but is discontinuous at ``theta = pi``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "relative_rotation",
    "rotation_geodesic_angle",
    "so3_log_map",
    "so3_exp_map",
    "rotation_to_6d",
    "rotation_from_6d",
    "is_proper_rotation",
    "random_rotation_of_angle",
]

#: Below this angle (radians) ``sin(theta)/theta`` is replaced by its series, and
#: within this of ``pi`` the log map switches to the symmetric branch. 1e-6 rad is
#: 6e-5 degrees -- far below any angle this project measures, and far above
#: float32 noise on a rotation matrix.
_SMALL_ANGLE = 1e-6


def _vee(skew: Tensor) -> Tensor:
    """Inverse of the hat map: ``[..., 3, 3]`` skew matrix -> ``[..., 3]``."""
    return torch.stack(
        [
            skew[..., 2, 1] - skew[..., 1, 2],
            skew[..., 0, 2] - skew[..., 2, 0],
            skew[..., 1, 0] - skew[..., 0, 1],
        ],
        dim=-1,
    ) * 0.5


def _axis_sine(rotation: Tensor) -> Tensor:
    """``vee((R - R^T)/2)``, whose norm is ``sin(theta)``.

    The factor of one half is the whole content of this helper and is the reason
    it exists rather than being written inline twice: ``_vee`` is the inverse of
    ``_hat`` and already halves, so ``_vee(R - R^T)`` is *twice* the intended
    vector. That doubling is silent -- it reports a 30 degree rotation as 49.1
    degrees and is exact at 0, 90 and 180 degrees, so the obvious spot checks all
    pass.
    """
    return _vee((rotation - rotation.transpose(-1, -2)) * 0.5)


def _hat(vector: Tensor) -> Tensor:
    """``[..., 3]`` -> ``[..., 3, 3]`` skew-symmetric matrix."""
    zero = torch.zeros_like(vector[..., 0])
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def relative_rotation(source: Tensor, destination: Tensor) -> Tensor:
    """``R_rel = source^T destination``, expressed in ``source``'s local frame.

    Args:
        source / destination: ``[..., 3, 3]`` rotations whose **columns** are the
            local axes in global coordinates.

    Returns:
        ``[..., 3, 3]`` with ``destination = source @ relative_rotation(...)``.
    """
    return source.transpose(-1, -2) @ destination


def rotation_geodesic_angle(a: Tensor, b: Tensor | None = None) -> Tensor:
    """Angle in radians of ``a`` (or between ``a`` and ``b``), in ``[0, pi]``.

    Args:
        a: ``[..., 3, 3]`` rotation.
        b: optional second rotation; when given the angle of ``a^T b`` is
            returned, i.e. the geodesic distance on SO(3).

    Numerically stable at both ends: exactly 0 for the identity and exactly
    ``pi`` for a half turn, with no ``arccos`` anywhere.
    """
    rotation = a if b is None else relative_rotation(a, b)
    cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    sine = torch.linalg.norm(_axis_sine(rotation), dim=-1)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def so3_log_map(rotation: Tensor) -> Tensor:
    """Rotation matrix -> rotation vector ``theta * axis``, ``[..., 3]``.

    Three regimes, because one formula does not cover SO(3):

    * ``theta`` near 0: ``theta / sin(theta) -> 1``, so the antisymmetric part is
      the answer directly. Taking the ratio there divides ~0 by ~0.
    * generic ``theta``: ``w = theta / sin(theta) * vee((R - R^T)/2)``.
    * ``theta`` near ``pi``: ``sin(theta) -> 0`` again but the axis is *not*
      recoverable from the antisymmetric part, which vanishes. The axis is taken
      from the symmetric part ``(R + I)/2``, whose diagonal is ``axis**2``, with
      the signs fixed by its largest row. Skipping this branch produces NaN or a
      silently wrong axis for any near-half-turn.

    The map is discontinuous at ``theta = pi`` (``w`` and ``-w`` describe the same
    rotation); that is a property of the chart, not of this implementation, and is
    why the *loss* uses :func:`rotation_geodesic_angle` instead.
    """
    theta = rotation_geodesic_angle(rotation)
    axis_sine = _axis_sine(rotation)  # = sin(theta) * axis

    sin_theta = torch.sin(theta)
    # Generic and small-angle branches share one expression: the denominator is
    # replaced by 1 where it would be ~0 and the small-angle case is selected
    # afterwards, so no NaN is produced in a branch that is not taken -- a NaN in
    # an unselected branch still poisons the backward pass, the same rule Phase 1's
    # frames module follows.
    safe_sin = torch.where(sin_theta.abs() < _SMALL_ANGLE,
                           torch.ones_like(sin_theta), sin_theta)
    generic = axis_sine * (theta / safe_sin).unsqueeze(-1)
    small = axis_sine  # theta / sin(theta) -> 1

    # Near a half turn the antisymmetric part vanishes and carries no axis. There
    # (R + I)/2 -> a a^T, so any row k is a_k * a: dividing row k by sqrt of its
    # diagonal recovers the axis, and the best-conditioned k is the largest
    # diagonal entry. The overall sign is free -- w and -w are the same half turn.
    symmetric = (rotation + torch.eye(3, dtype=rotation.dtype, device=rotation.device)) * 0.5
    diagonal = symmetric.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    lead = diagonal.argmax(dim=-1, keepdim=True)
    lead_row = torch.gather(
        symmetric, -2, lead.unsqueeze(-1).expand(*lead.shape, 3)
    ).squeeze(-2)
    axis_pi = lead_row / torch.sqrt(
        torch.gather(diagonal, -1, lead)
    ).clamp(min=_SMALL_ANGLE)
    axis_pi = axis_pi / torch.linalg.norm(axis_pi, dim=-1, keepdim=True).clamp(min=_SMALL_ANGLE)

    near_pi = theta > (torch.pi - 1e-3)
    result = torch.where((theta.abs() < _SMALL_ANGLE).unsqueeze(-1), small, generic)
    return torch.where(near_pi.unsqueeze(-1), axis_pi * theta.unsqueeze(-1), result)


def so3_exp_map(vector: Tensor) -> Tensor:
    """Rotation vector ``[..., 3]`` -> rotation matrix, by Rodrigues' formula.

    ``R = I + sin(t)/t * K + (1 - cos(t))/t^2 * K^2`` with ``K = hat(w)``. Both
    coefficients are replaced by their Taylor series below
    :data:`_SMALL_ANGLE`, where the closed form is 0/0.
    """
    theta = torch.linalg.norm(vector, dim=-1)
    K = _hat(vector)
    K2 = K @ K
    small = theta < _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(safe)) / safe**2)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(K)
    return eye + a[..., None, None] * K + b[..., None, None] * K2


def rotation_to_6d(rotation: Tensor) -> Tensor:
    """Rotation -> its first two columns, ``[..., 6]`` (Zhou et al. 2019).

    Columns, not rows: the columns of ``R`` are this project's local axes, so the
    6D vector is "the local x and y axes in global coordinates", which is a
    statement about the molecule rather than about a storage layout.
    """
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_from_6d(vector: Tensor) -> Tensor:
    """``[..., 6]`` -> the nearest rotation, by Gram-Schmidt on the two columns.

    Continuous and surjective onto SO(3), which is what makes it a safe output
    representation for a regression head: unlike axis-angle or quaternions there
    is no antipodal identification and no wrap-around for the network to learn.
    ``det = +1`` by construction, because the third axis is a cross product.
    """
    a1 = vector[..., 0:3]
    a2 = vector[..., 3:6]
    e1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
    a2 = a2 - (a2 * e1).sum(-1, keepdim=True) * e1
    e2 = a2 / torch.linalg.norm(a2, dim=-1, keepdim=True).clamp(min=1e-8)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def is_proper_rotation(rotation: Tensor, *, tolerance: float = 1e-5) -> Tensor:
    """``[...]`` bool: orthonormal **and** ``det = +1``.

    Reflection is not a symmetry of a chiral molecule, so "orthogonal" is not the
    check that matters here; a mirrored frame passes it and would silently invert
    the handedness of every target built from it.
    """
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthonormal = (
        (rotation.transpose(-1, -2) @ rotation - eye).abs().amax(dim=(-2, -1)) < tolerance
    )
    return orthonormal & ((torch.linalg.det(rotation) - 1.0).abs() < tolerance)


def random_rotation_of_angle(
    angle_radians: float,
    *,
    axis: Tensor | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """A rotation of exactly ``angle_radians`` about ``axis`` (random if omitted).

    Exists for the tests: a geodesic-angle implementation must be checked against
    a rotation whose angle is known analytically, not against another
    implementation of the same formula.
    """
    if axis is None:
        axis = torch.empty(3, dtype=torch.float64)
        axis.normal_(generator=generator)
    axis = axis.to(dtype=dtype)
    axis = axis / torch.linalg.norm(axis)
    return so3_exp_map(axis * angle_radians)
