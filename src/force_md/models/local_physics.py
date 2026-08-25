"""``LocalPhysicsModel`` -- the Phase 1 model, assembled from the real modules.

Forward path::

    validate batch
      -> link backbone frames to the atom position tensor
      -> build the hierarchical graph (6 relations)
      -> residue frames and local coordinates
      -> HierarchicalPhysicsEncoder  (A -> R -> B -> R -> A, twice)
      -> InvariantEnergyHead -> conservative force
      -> AtomicEffectiveForceHead / ResiduePhysicsHead
      -> ResidueSumProjector on the *predicted* atom forces
      -> Phase1Output

Nothing here is new machinery: every component is the module written in
Checkpoints 2-7, wired together. Depth and width are configuration, so Phase 2
loads this same class.

**Ground-truth forces are never a forward input.** ``batch.atoms.forces`` is a
label; the forward pass reads positions, chemistry, sequence and temperature
only. ``test_forward_ignores_ground_truth_forces`` pins this down by running the
model with the labels deleted and with the labels randomised, and requiring bitwise
identical output.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from ..data.contracts import HierarchicalProteinBatch
from ..geometry.frames import (
    ResidueFrames,
    frames_from_batch,
    link_backbone_to_atom_positions,
)
from ..graph.hierarchy import GraphConfig, HierarchicalGraph, build_hierarchical_graph
from ..nn.hierarchical_encoder import EncoderConfig, HierarchicalPhysicsEncoder
from ..physics.energy import InvariantEnergyHead, conservative_force
from ..physics.heads import AtomicEffectiveForceHead, ResiduePhysicsHead
from ..physics.outputs import Phase1Output
from ..physics.projection import ResidueSumProjector, TargetScope

__all__ = ["LocalPhysicsConfig", "LocalPhysicsModel"]


def _detach_output(output: Phase1Output) -> Phase1Output:
    """Detach every tensor, so a locally-enabled graph never escapes."""
    return dataclasses.replace(
        output,
        **{
            f.name: v.detach()
            for f in dataclasses.fields(output)
            for v in (getattr(output, f.name),)
            if isinstance(v, torch.Tensor)
        },
    )


@dataclass(frozen=True)
class LocalPhysicsConfig:
    """Phase 1 model configuration.

    Args:
        encoder / graph: sub-configs.
        target_scope: which atoms the residue targets sum over. Also decides
            whether a hidden residual is identifiable.
        predict_hidden_force: emit the omitted-atom residual head. ``None`` means
            "decide from ``target_scope``": with a heavy-atom target the residual
            is only identifiable because mdCATH also stores hydrogens, so the
            all-atom target exists to supervise it. Setting this True while
            training on an all-atom target would be meaningless -- there would be
            nothing left out -- and is rejected.
        use_energy_branch: compute ``U`` and ``-grad U``.
        conservative_force_create_graph: keep the second-order graph so the
            conservative force can be trained against a label. Required during
            training with ``lambda_energy > 0``; wasteful at inference.
        validate_inputs: run the full batch contract check. Cheap next to the
            encoder, and it catches malformed adapters early.

    Ablation hooks (Checkpoint 8 requirement) are all config, never a different
    class: ``encoder.use_plm``, ``encoder.use_atom_branch``,
    ``encoder.use_backbone_branch``, ``predict_hidden_force``,
    ``use_energy_branch``.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    target_scope: TargetScope = "heavy_atom"
    predict_hidden_force: Optional[bool] = None
    use_energy_branch: bool = True
    conservative_force_create_graph: bool = True
    isotropic_uncertainty: bool = False
    validate_inputs: bool = True

    def resolved_predict_hidden_force(self) -> bool:
        if self.predict_hidden_force is None:
            return self.target_scope == "heavy_atom"
        if self.predict_hidden_force and self.target_scope == "all_atom":
            raise ValueError(
                "predict_hidden_force=True with target_scope='all_atom': an "
                "all-atom target leaves no atoms out, so the residual has "
                "nothing to explain and is unidentifiable"
            )
        return self.predict_hidden_force


class LocalPhysicsModel(nn.Module):
    """Hierarchical local-physics model, Phase 1.

    Shape:
        ``HierarchicalProteinBatch -> Phase1Output``.
    """

    def __init__(self, config: LocalPhysicsConfig = LocalPhysicsConfig()):
        super().__init__()
        self.config = config
        self.predict_hidden_force = config.resolved_predict_hidden_force()

        self.encoder = HierarchicalPhysicsEncoder(config.encoder)
        irreps = self.encoder.irreps
        self.atom_head = AtomicEffectiveForceHead(
            irreps, isotropic_uncertainty=config.isotropic_uncertainty
        )
        self.residue_head = ResiduePhysicsHead(
            irreps,
            predict_hidden_force=self.predict_hidden_force,
            isotropic_uncertainty=config.isotropic_uncertainty,
        )
        self.energy_head = (
            InvariantEnergyHead(irreps) if config.use_energy_branch else None
        )
        self.projector = ResidueSumProjector(config.target_scope)

    @property
    def irreps(self):
        return self.encoder.irreps

    def forward(
        self,
        batch: HierarchicalProteinBatch,
        graph: Optional[HierarchicalGraph] = None,
        *,
        compute_conservative_force: Optional[bool] = None,
    ) -> Phase1Output:
        """Run the full Phase 1 forward pass.

        Args:
            batch: input state. Its ``atoms.forces`` are **not** read.
            graph: prebuilt topology; rebuilt from ``batch`` when omitted.
            compute_conservative_force: override the config. Defaults to True
                whenever the energy branch exists.

        Returns:
            :class:`Phase1Output`.
        """
        if self.config.validate_inputs:
            batch.validate()

        want_conservative = (
            self.energy_head is not None
            if compute_conservative_force is None
            else compute_conservative_force and self.energy_head is not None
        )

        # -dU/dx needs autograd even at inference, where the caller will normally
        # be inside torch.no_grad(). Enable grad locally for the path that leads
        # to the energy, then detach on the way out so no graph escapes into a
        # no_grad caller. Without this the energy branch raises "does not require
        # grad" the moment the model is used the way inference actually uses it.
        outer_grad_enabled = torch.is_grad_enabled()
        if want_conservative:
            with torch.enable_grad():
                output = self._forward_inner(batch, graph, want_conservative=True)
            return output if outer_grad_enabled else _detach_output(output)
        return self._forward_inner(batch, graph, want_conservative=False)

    def _forward_inner(
        self,
        batch: HierarchicalProteinBatch,
        graph: Optional[HierarchicalGraph],
        *,
        want_conservative: bool,
    ) -> Phase1Output:
        # One tensor of record for geometry, so -dU/dx is complete.
        batch = link_backbone_to_atom_positions(batch)
        if want_conservative and not batch.atoms.positions.requires_grad:
            positions = batch.atoms.positions.detach().requires_grad_(True)
            batch = dataclasses.replace(
                batch, atoms=dataclasses.replace(batch.atoms, positions=positions)
            )
            batch = link_backbone_to_atom_positions(batch)

        if graph is None:
            graph = build_hierarchical_graph(batch, self.config.graph)
        frames: ResidueFrames = frames_from_batch(batch)

        encoded = self.encoder(batch, graph, frames)

        # ---- energy and its gradient ------------------------------------
        if self.energy_head is not None:
            energy, residue_energy = self.energy_head(
                encoded.residue_features,
                batch.residues.batch_index,
                batch.num_graphs,
                residue_mask=batch.residues.mask,
            )
        else:
            energy = encoded.residue_features.new_zeros(batch.num_graphs)
            residue_energy = encoded.residue_features.new_zeros(batch.num_residues)

        conservative = None
        if want_conservative:
            conservative = conservative_force(
                energy,
                batch.atoms.positions,
                create_graph=self.config.conservative_force_create_graph
                and self.training,
            )

        # ---- heads -------------------------------------------------------
        atom_out = self.atom_head(encoded.atom_features, conservative=conservative)
        residue_out = self.residue_head(
            encoded.residue_features, batch.backbone.ca_positions
        )

        # The predicted atom forces go through the *same* projector that built
        # the labels; this is what makes the consistency loss compare like with
        # like.
        aggregated = self.projector(
            batch, atom_out.mean, require_valid_forces=False
        )

        return Phase1Output(
            atom_force_mean=atom_out.mean,
            atom_force_residual=atom_out.residual,
            atom_force_logvar=atom_out.logvar,
            atom_force_conservative=atom_out.conservative,
            residue_explained_force=residue_out.explained_force,
            residue_hidden_force=residue_out.hidden_force,
            residue_force_mean=residue_out.force_mean,
            residue_torque_mean=residue_out.torque_mean,
            residue_torque_origin=residue_out.torque_origin,
            residue_force_logvar=residue_out.force_logvar,
            residue_torque_logvar=residue_out.torque_logvar,
            aggregated_atom_force=aggregated.force,
            aggregated_atom_torque=aggregated.torque,
            energy=energy,
            residue_energy=residue_energy,
            physics_latent=encoded.residue_features,
            physics_latent_irreps=str(self.irreps),
            target_scope=self.config.target_scope,
        )

    # -- Phase 2 handoff ---------------------------------------------------
    def latent_contract(self) -> dict[str, object]:
        """The exact promise Phase 2 may rely on."""
        return {
            "physics_latent_irreps": str(self.irreps),
            "physics_latent_dim": int(self.irreps.dim),
            "row_order": "aligned with batch.residues (residue node index)",
            "frame": "global",
            "target_scope": self.config.target_scope,
            "predicts_hidden_force": self.predict_hidden_force,
            "num_cycles": self.config.encoder.num_cycles,
            "lmax": self.config.encoder.irreps.lmax,
        }
