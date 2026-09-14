"""H2 -- hydrogen force aggregation, the force predictor, and the four arms.

Three separate concerns, kept in one module because they share one contract: what
"the force on a heavy atom" means in a simulation that also had hydrogens.

**Aggregation (§6.6).** mdCATH stores forces on every atom, hydrogens included,
but the model represents heavy atoms only. The effective force on a heavy atom is
its own plus its hydrogens':

    F_A^eff = f_A + sum_{h in H(A)} f_h

and the hydrogens' first moment about the parent is also available:

    tau_A^H = sum_{h in H(A)} (x_h - x_A) x f_h

Every hydrogen has exactly one parent in the PSF -- asserted in
:mod:`force_md.heavy.topology` -- so the sum is a partition, and total force is
conserved exactly. Nothing is inferred from distance.

**The force is a vector.** Under ``x -> Qx + t`` it transforms as ``f -> Qf``,
so it is never concatenated into an invariant scalar channel. The conditioner
takes it in the residue-local frame, ``R_i^T f``, which is invariant to a global
rotation *and* keeps the directional information a norm would throw away.

**The four arms differ only in what enters the force encoder.** Same
architecture, same parameter count, asserted at construction. That is the whole
point: a difference between them is a difference in force information, not in
capacity.

* ``H2G`` geometry control -- zeros in, ``has_force = 0``.
* ``H2P`` predicted force -- its **own** predictions, at training and inference
  alike. No teacher forcing: a branch trained on ground truth and evaluated on
  its own output is measuring something it will never do.
* ``H2O`` oracle -- ground-truth force at time ``t`` only. Never deployable.
* ``H2S`` shuffled control -- real forces permuted **within a stratum**, so the
  magnitude distribution survives and the correspondence to structure does not.

``H2O - H2S`` is the comparison that isolates whether force *correspondence*
matters, as opposed to force magnitude statistics. If they tie, the oracle's
advantage was never about this structure's forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "aggregate_hydrogen_forces",
    "HydrogenAggregation",
    "AtomForcePredictor",
    "AtomForceConditionerConfig",
    "AtomForceConditioner",
    "H2_ARMS",
    "shuffle_forces_within_strata",
]

#: The four H2 arms and what each is allowed to read. ``deployable`` is False
#: wherever the arm needs a label at inference time.
H2_ARMS: dict[str, dict] = {
    "H2G_atom_geometry_control": {
        "force_source": "zeros",
        "deployable": True,
        "summary": "atom graph with the force channel held at zero",
    },
    "H2P_predicted_atom_force": {
        "force_source": "predicted",
        "deployable": True,
        "summary": "its own predicted effective force and uncertainty",
    },
    "H2O_current_gt_atom_force_oracle": {
        "force_source": "ground_truth_current",
        "deployable": False,
        "summary": "ground-truth effective force at time t. Upper bound.",
    },
    "H2S_shuffled_force_control": {
        "force_source": "shuffled",
        "deployable": False,
        "summary": "real forces permuted within a stratum. Negative control.",
    },
}


@dataclass
class HydrogenAggregation:
    """Effective per-heavy-atom force, and the hydrogens' moment about it.

    Args:
        force: ``[N_heavy, 3]`` in the batch's force unit (kcal/mol/A).
        torque: ``[N_heavy, 3]`` or None, ``sum (x_h - x_A) x f_h``. Unit is
            length x force = kcal/mol.
        n_hydrogens: ``[N_heavy]`` int64, how many hydrogens were folded in.
        mode: which of the brief's three modes produced this.
    """

    force: Tensor
    torque: Optional[Tensor]
    n_hydrogens: Tensor
    mode: str


def aggregate_hydrogen_forces(
    forces: Tensor,
    positions: Tensor,
    hydrogen_parent: Tensor,
    raw_to_batch: Tensor,
    *,
    mode: str = "heavy_plus_bonded_hydrogen",
) -> HydrogenAggregation:
    """Fold hydrogen forces onto their bonded heavy atoms.

    Args:
        forces / positions: ``[N_raw, 3]`` over **all** protein atoms, hydrogens
            included -- i.e. the ``all_atom`` load path.
        hydrogen_parent: ``[N_raw]``, the heavy parent of each hydrogen, ``-1``
            for heavy atoms.
        raw_to_batch: ``[N_raw]``, row in the heavy-atom array, ``-1`` for atoms
            the heavy representation drops.
        mode: one of ``heavy_only``, ``heavy_plus_bonded_hydrogen``,
            ``heavy_plus_bonded_hydrogen_and_torque``.

    Total force is conserved exactly in the two aggregating modes, which
    ``test_hydrogen_aggregation_conserves_total_force`` checks rather than
    assumes.
    """
    allowed = {
        "heavy_only",
        "heavy_plus_bonded_hydrogen",
        "heavy_plus_bonded_hydrogen_and_torque",
    }
    if mode not in allowed:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(allowed)}")

    heavy_rows = (raw_to_batch >= 0).nonzero(as_tuple=True)[0]
    n_heavy = int(raw_to_batch.max()) + 1 if heavy_rows.numel() else 0
    out = forces.new_zeros((n_heavy, 3))
    out[raw_to_batch[heavy_rows]] = forces[heavy_rows]
    counts = torch.zeros(n_heavy, dtype=torch.int64, device=forces.device)

    if mode == "heavy_only":
        return HydrogenAggregation(out, None, counts, mode)

    hydrogens = (hydrogen_parent >= 0).nonzero(as_tuple=True)[0]
    if hydrogens.numel():
        parents = raw_to_batch[hydrogen_parent[hydrogens]]
        if bool((parents < 0).any()):
            raise ValueError(
                "a hydrogen's parent heavy atom is not in the heavy-atom "
                "representation; its force would be dropped silently"
            )
        out.index_add_(0, parents, forces[hydrogens])
        counts.index_add_(
            0, parents, torch.ones_like(parents, dtype=torch.int64)
        )

    torque = None
    if mode.endswith("torque") and hydrogens.numel():
        parents = raw_to_batch[hydrogen_parent[hydrogens]]
        lever = positions[hydrogens] - positions[hydrogen_parent[hydrogens]]
        torque = forces.new_zeros((n_heavy, 3))
        torque.index_add_(0, parents, torch.linalg.cross(lever, forces[hydrogens]))
    return HydrogenAggregation(out, torque, counts, mode)


def shuffle_forces_within_strata(
    forces: Tensor, strata: Tensor, *, seed: int
) -> Tensor:
    """Permute force **vectors** within each stratum, with a fixed seed.

    The negative control. Within a stratum -- element crossed with temperature,
    say -- the multiset of force vectors is unchanged, so magnitude and
    directional statistics survive; what is destroyed is which atom had which
    force. An arm that gains as much from this as from the real thing was never
    using the correspondence.

    A fixed seed because a control that changes between runs is not a control.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = forces.clone()
    for value in torch.unique(strata):
        rows = (strata == value).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        permutation = torch.randperm(rows.numel(), generator=generator)
        out[rows] = forces[rows[permutation.to(rows.device)]]
    return out


class AtomForcePredictor(nn.Module):
    """Predicts the effective force on each heavy atom, with uncertainty.

    Outputs a mean vector and a log-variance. The vector is produced in the
    **residue-local frame** and rotated out by the caller, which is what makes
    the whole thing equivariant without an equivariant network: local components
    are invariants, and the frame carries the covariance.

    Trained with a heteroscedastic Gaussian NLL so the head can say when it does
    not know -- which matters here, because ``H2P`` feeds its own uncertainty
    into the conditioner and a confident wrong force is worse than an honest
    unsure one.
    """

    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 3)
        self.log_variance = nn.Linear(hidden, 3)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.zeros_(self.log_variance.bias)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        hidden = self.trunk(features)
        return {
            "mean_local": self.mean(hidden),
            "log_variance": self.log_variance(hidden).clamp(-8.0, 8.0),
        }


def heteroscedastic_nll(
    mean: Tensor, log_variance: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    """Gaussian NLL with a predicted per-component variance."""
    if not bool(mask.any()):
        return mean.new_tensor(float("nan"))
    inverse = torch.exp(-log_variance)
    term = 0.5 * (log_variance + inverse * (mean - target).pow(2))
    return term[mask].mean()


@dataclass
class AtomForceConditionerConfig:
    """Shared by all four arms. Changing it changes all of them together."""

    arm: str = "H2G_atom_geometry_control"
    node_features: int = 64
    force_channels: int = 32
    hidden: int = 128
    use_torque: bool = False
    shuffle_seed: int = 20260828

    def __post_init__(self) -> None:
        if self.arm not in H2_ARMS:
            raise ValueError(
                f"unknown H2 arm {self.arm!r}; expected one of {sorted(H2_ARMS)}"
            )


class AtomForceConditioner(nn.Module):
    """The capacity-matched H2 arms.

    Every arm instantiates the **same** modules; only the tensor handed to the
    force encoder differs. ``parameter_count`` is therefore identical across
    arms by construction, and
    ``test_all_h2_arms_have_identical_parameter_counts`` proves it rather than
    trusting the construction.

    The force encoder exists even in ``H2G``, where it is fed zeros. Removing it
    there would make the geometry control a smaller model and turn the
    comparison into a capacity comparison.
    """

    def __init__(self, config: AtomForceConditionerConfig | None = None):
        super().__init__()
        self.config = config or AtomForceConditionerConfig()
        width = 3 + 3 + 1 + (3 if self.config.use_torque else 0)
        self.force_encoder = nn.Sequential(
            nn.Linear(width, self.config.force_channels), nn.SiLU(),
            nn.Linear(self.config.force_channels, self.config.force_channels),
        )
        self.mix = nn.Sequential(
            nn.Linear(
                self.config.node_features + self.config.force_channels,
                self.config.hidden,
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden, self.config.node_features),
        )

    @property
    def force_source(self) -> str:
        return H2_ARMS[self.config.arm]["force_source"]

    @property
    def deployable(self) -> bool:
        return H2_ARMS[self.config.arm]["deployable"]

    def force_input(
        self,
        *,
        local_force: Optional[Tensor],
        log_variance: Optional[Tensor],
        local_torque: Optional[Tensor],
        n_atoms: int,
        device,
        dtype,
        strata: Optional[Tensor] = None,
    ) -> Tensor:
        """Assemble the force channel for this arm. Zeros for the control.

        ``has_force`` is an explicit 0/1 channel rather than an implicit
        consequence of zeros, so the network can tell "no force information" from
        "a force that happens to be zero" -- which a buried atom's very nearly
        is.
        """
        source = self.force_source
        width = 3 + 3 + 1 + (3 if self.config.use_torque else 0)
        if source == "zeros":
            return torch.zeros((n_atoms, width), device=device, dtype=dtype)
        if local_force is None:
            raise ValueError(
                f"arm {self.config.arm!r} needs a force but none was supplied"
            )
        force = local_force
        if source == "shuffled":
            if strata is None:
                raise ValueError(
                    "the shuffled control needs strata; shuffling across the "
                    "whole batch would destroy the magnitude distribution the "
                    "control is supposed to preserve"
                )
            force = shuffle_forces_within_strata(
                force, strata, seed=self.config.shuffle_seed
            )
        variance = (
            log_variance
            if log_variance is not None
            else torch.zeros_like(force)
        )
        parts = [force, variance, torch.ones((n_atoms, 1), device=device, dtype=dtype)]
        if self.config.use_torque:
            parts.append(
                local_torque
                if local_torque is not None
                else torch.zeros_like(force)
            )
        return torch.cat(parts, dim=-1)

    def forward(self, node_features: Tensor, force_input: Tensor) -> Tensor:
        encoded = self.force_encoder(force_input)
        return self.mix(torch.cat([node_features, encoded], dim=-1))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
