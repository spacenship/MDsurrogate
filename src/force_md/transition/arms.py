"""Canonical Phase 1.6 arm names, and what each one is allowed to see.

Phase 1.5 named its arms after what they condition on (``physics_latent``,
``oracle_force``). The Phase 1.6 plan names them by role (``S2_node_physics_152d``,
``O_current_gt_force_oracle``). Both names refer to the *same* registered
conditioner: this module is the mapping, not a second set of arms.

Every results row and every report table carries both names, so a Phase 1.5
number and a Phase 1.6 number can be put side by side without anyone having to
remember which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .conditioners import CONDITIONER_ARMS

__all__ = [
    "ArmSpec",
    "CANONICAL_ARMS",
    "CANONICAL_BY_IMPLEMENTATION",
    "SCREENING_ARMS",
    "MECHANISM_ARMS",
    "canonical_name",
    "implementation_name",
    "resolve_arm",
    "arm_spec",
]


@dataclass(frozen=True)
class ArmSpec:
    """One canonical arm.

    Args:
        canonical: the plan's name.
        implementation: the registered conditioner name (``probe_config.arm``).
        oracle: reads a ground-truth label. Recorded in every checkpoint and row.
        needs_pair_features: needs Phase 1's residue-pair messages extracted.
        future_physics: attaches the ``P4`` auxiliary head.
        stage: which experiment stage this arm belongs to.
        summary: one line, for the report's arm table.
    """

    canonical: str
    implementation: str
    oracle: bool
    needs_pair_features: bool
    future_physics: bool
    stage: str
    summary: str


CANONICAL_ARMS: dict[str, ArmSpec] = {
    spec.canonical: spec
    for spec in (
        ArmSpec(
            "S0_structure_history", "structure_only", False, False, False, "screening",
            "structure, history, PLM, temperature, lag. No physics. The baseline.",
        ),
        ArmSpec(
            "S1_residue_wrench", "force_torque", False, False, False, "screening",
            "S0 plus Phase 1's predicted residue net force and torque.",
        ),
        ArmSpec(
            "S2_node_physics_152d", "physics_latent", False, False, False, "screening",
            "S0 plus the full 152-d node physics latent, local frame.",
        ),
        ArmSpec(
            "O_current_gt_force_oracle", "oracle_force", True, False, False, "screening",
            "Ground-truth atom forces at t. Upper bound, never deployable.",
        ),
        ArmSpec(
            "P0_pair_geometry_control", "pair_geometry", False, True, False, "screening",
            "Pair architecture on current pair geometry only, capacity-matched to P1.",
        ),
        ArmSpec(
            "P1_pair_physics_frozen", "pair_physics", False, True, False, "screening",
            "Phase 1's residue-pair messages, local frame, pooled per residue.",
        ),
        ArmSpec(
            "P2_pair_physics_moments", "pair_physics_moments", False, True, False,
            "screening",
            "P1 plus predicted-force moments F, tau and the traceless stress S.",
        ),
        ArmSpec(
            "P3_pair_physics_uncertainty", "pair_physics_uncertainty", False, True,
            False, "mechanism",
            "P2 with a sigmoid gate on (log sigma, T, lag).",
        ),
        ArmSpec(
            "P4_future_physics_consistency", "pair_physics_future", False, True, True,
            "mechanism",
            "P2 plus an auxiliary head predicting the future physics latent.",
        ),
        ArmSpec(
            "P5_pair_physics_partial_e2e", "pair_physics_moments", False, True, False,
            "gated",
            "P2 with Phase 1's last block unfrozen. Config only; never auto-run.",
        ),
    )
}

#: Reverse lookup. ``P5`` deliberately shares ``P2``'s conditioner -- it differs by
#: what is frozen, not by what it reads -- so the implementation name maps back to
#: ``P2`` and ``P5`` must be named explicitly.
CANONICAL_BY_IMPLEMENTATION: dict[str, str] = {}
for _spec in CANONICAL_ARMS.values():
    CANONICAL_BY_IMPLEMENTATION.setdefault(_spec.implementation, _spec.canonical)

#: Legacy Phase 1.5 arm with no canonical Phase 1.6 role. Reported as legacy
#: rather than renamed, so the Phase 1.5 report stays readable against this code.
CANONICAL_BY_IMPLEMENTATION.setdefault("force_pattern_shape", "legacy_force_pattern_shape")

#: Stage B: everything the screening runs, in report order.
SCREENING_ARMS: tuple[str, ...] = tuple(
    name for name, spec in CANONICAL_ARMS.items() if spec.stage == "screening"
)

#: Stage D: only run if Stage C found a pair-physics signal.
MECHANISM_ARMS: tuple[str, ...] = tuple(
    name for name, spec in CANONICAL_ARMS.items() if spec.stage == "mechanism"
)


def arm_spec(name: str) -> ArmSpec:
    """Look an arm up by either name.

    Raises:
        KeyError: with both vocabularies listed, because "unknown arm" with no
            list is the error message that costs ten minutes.
    """
    if name in CANONICAL_ARMS:
        return CANONICAL_ARMS[name]
    for spec in CANONICAL_ARMS.values():
        if spec.implementation == name and spec.stage != "gated":
            return spec
    raise KeyError(
        f"unknown arm {name!r}. Canonical: {sorted(CANONICAL_ARMS)}. "
        f"Implementation: {sorted(CONDITIONER_ARMS)}."
    )


def resolve_arm(name: str) -> ArmSpec:
    """Alias of :func:`arm_spec`, for call sites that read better this way."""
    return arm_spec(name)


def canonical_name(implementation: str) -> str:
    return CANONICAL_BY_IMPLEMENTATION.get(implementation, implementation)


def implementation_name(canonical: str) -> str:
    return arm_spec(canonical).implementation
