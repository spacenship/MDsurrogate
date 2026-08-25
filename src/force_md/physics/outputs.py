"""The Phase 1 output contract.

Everything Phase 2 is allowed to depend on lives in :class:`Phase1Output`. Field
names and shapes are frozen here; adding fields is fine, renaming or repurposing
one is a breaking change for a Phase 2 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

__all__ = ["Phase1Output"]


@dataclass
class Phase1Output:
    """Structured output of :class:`force_md.models.LocalPhysicsModel`.

    Atom level:
        atom_force_mean: ``[N_atom, 3]`` global frame, ``residual + conservative``.
        atom_force_residual: ``[N_atom, 3]`` non-conservative part.
        atom_force_conservative: ``[N_atom, 3]`` ``-grad U``, or None.
        atom_force_logvar: ``[N_atom, 3]`` residue-local diagonal log-variance.

    Residue level:
        residue_explained_force: ``[N_res, 3]`` force of the represented atoms,
            predicted at residue level and tied to
            ``aggregated_atom_force`` by the consistency loss.
        residue_hidden_force: ``[N_res, 3]`` omitted-atom (hydrogen) residual, or
            None when the target scope does not identify it.
        residue_force_mean: ``[N_res, 3]`` explained + hidden.
        residue_torque_mean: ``[N_res, 3]`` about ``residue_torque_origin``.
        residue_torque_origin: ``[N_res, 3]`` (the CA positions).
        residue_force_logvar / residue_torque_logvar: ``[N_res, 3]`` local frame.
        aggregated_atom_force / aggregated_atom_torque: ``[N_res, 3]`` the
            predicted atom forces pushed through the *same* ``ResidueSumProjector``
            used to build the labels.

    Energy:
        energy: ``[B]`` invariant graph energy of the learned potential.
        residue_energy: ``[N_res]`` its per-residue decomposition (a latent
            decomposition, not a measurable per-residue energy).

    Phase 2 handoff:
        physics_latent: ``[N_res, D]`` residue irreps, aligned row-for-row with
            ``batch.residues``.
        physics_latent_irreps: the irreps string, e.g. ``"64x0e+16x1o+8x2e"``.
        target_scope: which atoms the residue targets sum over.
    """

    # atom level
    atom_force_mean: Tensor
    atom_force_residual: Tensor
    atom_force_logvar: Tensor
    atom_force_conservative: Optional[Tensor]

    # residue level
    residue_explained_force: Tensor
    residue_hidden_force: Optional[Tensor]
    residue_force_mean: Tensor
    residue_torque_mean: Tensor
    residue_torque_origin: Tensor
    residue_force_logvar: Tensor
    residue_torque_logvar: Tensor
    aggregated_atom_force: Tensor
    aggregated_atom_torque: Tensor

    # energy
    energy: Tensor
    residue_energy: Tensor

    # handoff
    physics_latent: Tensor
    physics_latent_irreps: str
    target_scope: str

    def to(self, device) -> "Phase1Output":
        import dataclasses

        from torch import Tensor as _T

        return dataclasses.replace(
            self,
            **{
                f.name: v.to(device)
                for f in dataclasses.fields(self)
                for v in (getattr(self, f.name),)
                if isinstance(v, _T)
            },
        )
