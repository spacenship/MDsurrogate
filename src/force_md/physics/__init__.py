"""Physics: force projection, output heads, energy, losses."""

from .energy import InvariantEnergyHead, conservative_force
from .heads import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    AtomicEffectiveForceHead,
    AtomicForceOutput,
    ResiduePhysicsHead,
    ResiduePhysicsOutput,
)
from .losses import (
    LossWeights,
    TargetNormalizer,
    masked_gaussian_nll,
    masked_mse,
    phase1_loss,
)
from .outputs import Phase1Output
from .projection import (
    ResidueForceTargets,
    ResidueSumProjector,
    TargetScope,
    omitted_atom_residual,
    shift_torque_origin,
)

__all__ = [
    # projection
    "ResidueSumProjector",
    "ResidueForceTargets",
    "TargetScope",
    "omitted_atom_residual",
    "shift_torque_origin",
    # heads
    "AtomicEffectiveForceHead",
    "AtomicForceOutput",
    "ResiduePhysicsHead",
    "ResiduePhysicsOutput",
    "LOGVAR_MIN",
    "LOGVAR_MAX",
    # energy
    "InvariantEnergyHead",
    "conservative_force",
    # outputs and losses
    "Phase1Output",
    "LossWeights",
    "TargetNormalizer",
    "masked_gaussian_nll",
    "masked_mse",
    "phase1_loss",
]
