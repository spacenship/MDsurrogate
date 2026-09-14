"""Inference and frozen-upstream wrapper for the Stage 4 Cartesian flow."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from .context_encoder import HeavyFlowContextEncoder
from .flow_decoder import FlowDecoder
from .flow_types import FlowAtomTopology, FlowConditionBundle
from .force_head import ForceHead
from .physics_predictor import PhysicsPredictor
from .rectified_flow import sample_flow_base
from .solver import integrate_flow
from .types import HeavyFlowCondition

__all__ = ["HeavyAtomFlowModel", "sample_flow"]


class HeavyAtomFlowModel(nn.Module):
    """Compose frozen Stage 1--3 conditioning with a trainable flow decoder.

    ``sample`` accepts only an inference condition.  Future coordinates and
    force labels are intentionally absent from this class's conditioning path;
    they are used only by the separate flow training function.
    """

    def __init__(
        self,
        context_encoder: HeavyFlowContextEncoder,
        predictor: PhysicsPredictor,
        force_head: ForceHead,
        decoder: FlowDecoder,
        *,
        freeze_upstream: bool = True,
        sigma_scale: float = 0.1,
        remove_center_of_mass_noise: bool = True,
        upstream_config: Optional[dict[str, Any]] = None,
        upstream_provenance: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        if sigma_scale < 0:
            raise ValueError("sigma_scale must be non-negative")
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.force_head = force_head
        self.decoder = decoder
        self.freeze_upstream = bool(freeze_upstream)
        self.sigma_scale = float(sigma_scale)
        self.remove_center_of_mass_noise = bool(remove_center_of_mass_noise)
        # Metadata is intentionally not an nn.Module/buffer.  It is carried
        # by the training entry point into Stage 4 checkpoints so a resumed
        # decoder can be traced back to the exact Stage 3 handoff.
        self.upstream_config = upstream_config
        self.upstream_provenance = upstream_provenance
        if self.freeze_upstream:
            self._freeze(self.context_encoder)
            self._freeze(self.predictor)
            self._freeze(self.force_head)
        self.train()

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()

    def train(self, mode: bool = True) -> "HeavyAtomFlowModel":
        super().train(mode)
        if self.freeze_upstream:
            self.context_encoder.eval()
            self.predictor.eval()
            self.force_head.eval()
        self.decoder.train(mode)
        return self

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @property
    def upstream_is_frozen(self) -> bool:
        return all(not parameter.requires_grad for module in (self.context_encoder, self.predictor, self.force_head) for parameter in module.parameters())

    def encode_condition(self, condition: HeavyFlowCondition) -> FlowConditionBundle:
        """Build the inference bundle without accessing any target fields."""
        condition.validate()
        if self.freeze_upstream:
            with torch.no_grad():
                context = self.context_encoder(condition)
                state = self.predictor(context)
                force = self.force_head(state)
        else:
            context = self.context_encoder(condition)
            state = self.predictor(context)
            force = self.force_head(state)
        enriched = state.with_force(force.force_mean, force.force_logvar)
        topology = FlowAtomTopology.from_graph(
            context.geometry.graph,
            atom_mask=condition.atom_mask,
            atom_to_residue=condition.atom_to_residue,
            atom_type=condition.atom_type,
            atom_name=condition.atom_name,
            residue_mask=condition.residue_mask,
            chain_break=condition.chain_break,
            spatial_cutoff_angstrom=self.decoder.config.spatial_cutoff_angstrom,
            max_spatial_neighbors=self.decoder.config.max_spatial_neighbors,
        )
        bundle = FlowConditionBundle(
            atom_context=context.atom_context.atom_features,
            residue_context=context.residue_context.joint_scalar,
            physics_state=enriched,
            force_mean=force.force_mean,
            force_logvar=force.force_logvar,
            atom_topology=topology,
            temperature=condition.temperature,
            lag=condition.lag,
            current_positions=condition.current_positions,
        )
        bundle.validate()
        return bundle

    def velocity(self, bundle: FlowConditionBundle, x_s: Tensor, flow_time: Tensor, recompute_spatial: bool = True) -> Tensor:
        return self.decoder(
            x_s,
            flow_time,
            bundle.atom_context,
            bundle.residue_context,
            bundle.physics_state,
            bundle.atom_topology,
            bundle.temperature,
            bundle.lag,
            recompute_spatial=recompute_spatial,
        )

    def forward(self, condition: HeavyFlowCondition | FlowConditionBundle, x_s: Tensor, flow_time: Tensor) -> Tensor:
        bundle = condition if isinstance(condition, FlowConditionBundle) else self.encode_condition(condition)
        return self.velocity(bundle, x_s, flow_time)

    def sample(
        self,
        condition: HeavyFlowCondition | FlowConditionBundle,
        *,
        num_samples: int = 1,
        seed: Optional[int] = None,
        solver: str = "heun",
        steps: int = 20,
    ) -> Tensor:
        """Generate ``[K,B,N,3]`` future heavy-atom coordinates."""
        bundle = condition if isinstance(condition, FlowConditionBundle) else self.encode_condition(condition)
        return sample_flow(
            self.decoder,
            bundle,
            num_samples=num_samples,
            seed=seed,
            solver=solver,
            steps=steps,
            sigma_scale=self.sigma_scale,
            remove_center_of_mass_noise=self.remove_center_of_mass_noise,
        )


def _seeded_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


@torch.no_grad()
def sample_flow(
    decoder: FlowDecoder,
    condition: FlowConditionBundle,
    *,
    num_samples: int = 1,
    seed: Optional[int] = None,
    solver: str = "heun",
    steps: int = 20,
    sigma_scale: float = 0.1,
    remove_center_of_mass_noise: bool = True,
    spatial_recompute_every: Optional[int] = None,
) -> Tensor:
    """Integrate independent noise draws against one frozen condition bundle."""
    condition.validate()
    if num_samples < 1 or sigma_scale < 0:
        raise ValueError("num_samples must be positive and sigma_scale non-negative")
    if spatial_recompute_every is None:
        spatial_recompute_every = decoder.config.spatial_recompute_every
    generator = _seeded_generator(condition.current_positions.device, seed)
    outputs: list[Tensor] = []
    for _ in range(num_samples):
        x0, _, _ = sample_flow_base(
            condition.current_positions,
            atom_mask=condition.atom_mask,
            lag=condition.lag,
            sigma_scale=sigma_scale,
            remove_center_of_mass=remove_center_of_mass_noise,
            generator=generator,
        )
        decoder.reset_graph_cache()
        def velocity_fn(x: Tensor, time: Tensor, recompute: bool) -> Tensor:
            return decoder(
                x,
                time,
                condition.atom_context,
                condition.residue_context,
                condition.physics_state,
                condition.atom_topology,
                condition.temperature,
                condition.lag,
                recompute_spatial=recompute,
            )
        outputs.append(integrate_flow(
            x0,
            velocity_fn,
            solver=solver,
            steps=steps,
            spatial_recompute_every=spatial_recompute_every,
        ))
    return torch.stack(outputs, dim=0)
