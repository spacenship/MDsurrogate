"""Projecting atomic forces onto residue force and torque targets.

The baseline, and the only projector implemented here, is a plain sum::

    F_i   = sum_{a in i} f_a
    tau_i = sum_{a in i} (x_a - r_i) x f_a          with r_i = CA_i

``ResidueSumProjector`` is named for what it does. It is not a "constraint-aware
projection" and must not be described as one: mdCATH's forces come from a
constrained simulation (rigid bonds to hydrogen), and simply adding them does not
undo those constraints. A genuinely constraint-aware projector would need the
constraint Jacobian and is deliberately left unimplemented.

**What a heavy-atom force label already contains.** ``f_a`` is the *total* force
on atom ``a``: it already includes the pull of that atom's hydrogens and of the
surrounding water. So summing over heavy atoms gives the net force on the
heavy-atom subset, solvent coupling included. What it omits is the force acting
*on* the hydrogens themselves.

**Which makes the omitted-atom residual identifiable here.** mdCATH stores
hydrogens (audited: 558 of 1126 atoms in a typical domain), so both scopes can be
built from the same file and their difference is a real, measurable quantity::

    residual_i = F_i(all_atom) - F_i(heavy_atom) = sum_{h in i} f_h

The design assumed only a heavy-atom target would exist, in which case a separate
additive hidden force would be unidentifiable and had to be disabled. That
assumption does not hold for this dataset. Measured on one domain: mean per-residue
|F| is 44.7 kcal/mol/A heavy-only, and the mean residual magnitude is 30.4 --
comparable, not negligible.

**Solvent is still not identifiable, and is never faked.** Water and ion atoms are
absent from the file. Their effect enters only through the protein-atom force
labels above. This module never assigns a solvent atom to a residue, and the
unresolved solvent contribution is represented by predicted *uncertainty*, not by
a residual term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

from ..data.contracts import HierarchicalProteinBatch
from ..nn.irreps import scatter_sum

__all__ = [
    "TargetScope",
    "ResidueForceTargets",
    "ResidueSumProjector",
    "omitted_atom_residual",
    "shift_torque_origin",
]

TargetScope = Literal["heavy_atom", "all_atom"]


@dataclass
class ResidueForceTargets:
    """Residue-level force/torque produced by a projector.

    Args:
        force: ``[N_res, 3]`` global-frame net force, equivariant.
        torque: ``[N_res, 3]`` global-frame torque about ``origin``.
        origin: ``[N_res, 3]`` reference point the torque is taken about. Stored
            because a torque without its origin is meaningless -- see
            :func:`shift_torque_origin`.
        scope: which atoms contributed (``"heavy_atom"`` or ``"all_atom"``).
        valid: ``[N_res]`` bool, False where a contributing atom had no usable
            force label or the residue contributed no atoms at all.
        num_atoms: ``[N_res]`` count of contributing atoms.
    """

    force: Tensor
    torque: Tensor
    origin: Tensor
    scope: str
    valid: Tensor
    num_atoms: Tensor

    def to(self, device) -> "ResidueForceTargets":
        import dataclasses
        return dataclasses.replace(
            self,
            force=self.force.to(device), torque=self.torque.to(device),
            origin=self.origin.to(device), valid=self.valid.to(device),
            num_atoms=self.num_atoms.to(device),
        )


class ResidueSumProjector(torch.nn.Module):
    """Sum atomic vectors into residue force and torque. No learned parameters.

    The same instance is used for the ground-truth labels and for the model's
    *predicted* atom forces. Reusing one operator is what makes the
    atom-to-residue aggregation-consistency loss meaningful: if prediction and
    target went through different aggregations, the consistency term would be
    comparing two different quantities.

    Args:
        scope: ``"heavy_atom"`` uses non-hydrogen atoms only; ``"all_atom"`` uses
            every stored atom. Hydrogens are **not** reassigned to their bonded
            heavy atom: in mdCATH a hydrogen already carries its heavy atom's
            residue id, so reassignment would change only the torque lever arm,
            and doing that without a physical argument would quietly alter the
            target. It is not implemented rather than implemented and defaulted off.
    """

    def __init__(self, scope: TargetScope = "heavy_atom"):
        super().__init__()
        if scope not in ("heavy_atom", "all_atom"):
            raise ValueError(f"scope must be 'heavy_atom' or 'all_atom', got {scope!r}")
        self.scope = scope

    def atom_selection(self, batch: HierarchicalProteinBatch) -> Tensor:
        """``[N_atom]`` bool mask of the atoms this scope sums over."""
        if self.scope == "all_atom":
            return torch.ones_like(batch.atoms.is_heavy)
        return batch.atoms.is_heavy

    def forward(
        self,
        batch: HierarchicalProteinBatch,
        atom_vectors: Optional[Tensor] = None,
        *,
        origin: Optional[Tensor] = None,
        require_valid_forces: bool = True,
    ) -> ResidueForceTargets:
        """Project ``atom_vectors`` (default: the batch's force labels).

        Args:
            atom_vectors: ``[N_atom, 3]`` global-frame vectors. Defaults to
                ``batch.atoms.forces``; pass predicted forces to aggregate a
                prediction through the identical operator.
            origin: ``[N_res, 3]`` torque reference. Defaults to CA.
            require_valid_forces: mark a residue invalid if any contributing atom
                has ``force_valid == False``. mdCATH ships trajectories whose
                force array is a copy of the coordinates; those must not become
                a training target through a silent sum.

        Returns:
            :class:`ResidueForceTargets`.
        """
        atoms = batch.atoms
        if atom_vectors is None:
            if atoms.forces is None:
                raise ValueError(
                    "batch has no atom forces; pass atom_vectors explicitly to "
                    "project a prediction"
                )
            atom_vectors = atoms.forces

        n_res = batch.num_residues
        origin = batch.backbone.ca_positions if origin is None else origin

        select = self.atom_selection(batch)
        contributing = select.unsqueeze(-1).to(atom_vectors.dtype)
        vectors = atom_vectors * contributing

        force = scatter_sum(vectors, atoms.atom_to_residue, n_res)
        lever = atoms.positions - origin[atoms.atom_to_residue]
        torque = scatter_sum(
            torch.linalg.cross(lever, vectors, dim=-1), atoms.atom_to_residue, n_res
        )

        counts = scatter_sum(
            select.to(torch.int64).unsqueeze(-1), atoms.atom_to_residue, n_res
        ).squeeze(-1)
        valid = counts > 0
        if require_valid_forces and atoms.force_valid is not None:
            bad = (~atoms.force_valid) & select
            n_bad = scatter_sum(
                bad.to(torch.int64).unsqueeze(-1), atoms.atom_to_residue, n_res
            ).squeeze(-1)
            valid = valid & (n_bad == 0)

        return ResidueForceTargets(
            force=force, torque=torque, origin=origin, scope=self.scope,
            valid=valid, num_atoms=counts,
        )


def shift_torque_origin(
    targets: ResidueForceTargets, new_origin: Tensor
) -> ResidueForceTargets:
    """Re-express a torque about a different reference point.

    ``tau(o') = tau(o) + (o - o') x F``. A torque quoted without its origin is
    ambiguous, so this law is implemented once and tested rather than being
    reimplemented at each call site.
    """
    import dataclasses

    delta = targets.origin - new_origin
    return dataclasses.replace(
        targets,
        torque=targets.torque + torch.linalg.cross(delta, targets.force, dim=-1),
        origin=new_origin,
    )


def omitted_atom_residual(
    all_atom: ResidueForceTargets, heavy_atom: ResidueForceTargets
) -> tuple[Tensor, Tensor]:
    """The identifiable part of what a heavy-atom representation leaves out.

    Returns:
        ``(force_residual [N_res, 3], torque_residual [N_res, 3])``, the summed
        contribution of the atoms present in the all-atom scope but absent from
        the heavy-atom one -- in practice, the hydrogens.

    This is a real supervision target *only* because mdCATH stores hydrogens. It
    is not a model of solvent: water never appears in the file and no part of
    this residual represents it.

    Raises:
        ValueError: if the two targets use different torque origins, which would
            make their difference meaningless.
    """
    if all_atom.scope != "all_atom" or heavy_atom.scope != "heavy_atom":
        raise ValueError(
            f"expected (all_atom, heavy_atom) scopes, got "
            f"({all_atom.scope!r}, {heavy_atom.scope!r})"
        )
    if not torch.allclose(all_atom.origin, heavy_atom.origin):
        raise ValueError(
            "torque origins differ; residual of torques about different points "
            "is not a torque"
        )
    return (
        all_atom.force - heavy_atom.force,
        all_atom.torque - heavy_atom.torque,
    )
