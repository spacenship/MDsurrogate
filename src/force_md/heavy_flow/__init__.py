"""Clean-slate heavy-atom dynamics foundation through direct Stage 4 flow."""

from .atom_graph import (
    CHAIN_ADJACENT_EDGE,
    COVALENT_EDGE,
    SPATIAL_EDGE,
    HeavyFlowAtomGraph,
    HeavyFlowGraphConfig,
    HeavyFlowGraphStats,
    PackedAtomState,
    build_atom_graph,
    pack_atom_state,
)
from .data import bonds_from_topology, collate_heavy_flow, from_mdcath_example, make_history
from .esmc_encoder import ESMCConfig, ESMCEmbeddingCache, ESMCEncoding, ESMCEncoder, ESMCCacheMiss, ESMCUnavailable
from .geometry_encoder import HeavyFlowAtomEncoder, HeavyFlowConditionEncoder, HeavyFlowGeometryConfig, HeavyFlowGeometryEncoding
from .equivariant_pooling import EquivariantAtomToResiduePool, EquivariantResiduePoolingOutput, irrep_chunks
from .seq_geo_fusion import SeqGeoResidueContext, SequenceGeometryFusion, ResidueGlobalContext, equivariant_to_invariant, invariant_width
from .atom_context import AtomContextOutput, AtomContextRefiner
from .context_encoder import HeavyFlowContextConfig, HeavyFlowContextEncoding, HeavyFlowContextEncoder
from .auxiliary_heads import AuxiliaryHeadOutput, HeavyFlowAuxiliaryHeads
from .types import HeavyFlowCondition, HeavyFlowProvenance, HeavyFlowSample, HeavyFlowTargets
from .physics_types import PhysicsState
from .physics_predictor import PhysicsPredictorConfig, PhysicsPredictor, PhysicsModelOutput, HeavyFlowPhysicsModel
from .force_head import ForceDistribution, ForceHeadOutput, AtomForceDistribution, ForceHead, AtomForceHead
from .force_losses import ForceLossConfig, ForceNormalizer, heteroscedastic_force_nll, masked_heteroscedastic_gaussian_nll, force_nll, masked_force_nll, force_loss, loss_with_diagnostics, variance_inflation_diagnostic
from .physics_metrics import ForceTargetAudit, compute_force_metrics, force_metrics, residue_net_force_torque, zero_force_baseline, mean_force_baseline, unconditional_mean_variance_baseline, shuffled_force_labels, audit_force_targets, target_audit
from .history import HistoryConfig
from .physics_dataset import HeavyFlowTemporalPairDataset, PhysicsFrameKey, PhysicsSplitManifest, split_physics_frames, HeavyFlowPhysicsFrameDataset, build_physics_splits
from .chunk_rotation import DomainChunk, domain_order, build_domain_chunks, load_domain_chunks

# Stage 4 direct Cartesian conditional rectified flow.
from .flow_types import FlowAtomTopology, FlowConditionBundle, FlowCondition, HeavyAtomFlowCondition, build_flow_graph, coerce_flow_topology
from .rectified_flow import FlowPath, align_future_to_current, sample_flow_base, build_flow_path, endpoint_from_velocity, rectified_flow_loss, rf_loss
from .flow_decoder import FlowDecoderConfig, FlowDecoder, HeavyAtomFlowDecoder
from .geometry_losses import GeometryLossConfig, GeometryLossBreakdown, geometry_regularization, compute_geometry_regularization, geometry_gradient_norms, combined_flow_loss
from .solver import FlowSolverConfig, integrate_flow, euler_step, heun_step
from .sampler import HeavyAtomFlowModel, sample_flow

__all__ = [
    "HeavyFlowCondition", "HeavyFlowTargets", "HeavyFlowProvenance", "HeavyFlowSample",
    "HeavyFlowGraphConfig", "HeavyFlowGraphStats", "HeavyFlowAtomGraph", "PackedAtomState",
    "COVALENT_EDGE", "SPATIAL_EDGE", "CHAIN_ADJACENT_EDGE", "build_atom_graph", "pack_atom_state",
    "ESMCConfig", "ESMCEmbeddingCache", "ESMCEncoding", "ESMCEncoder", "ESMCCacheMiss", "ESMCUnavailable",
    "HeavyFlowGeometryConfig", "HeavyFlowGeometryEncoding", "HeavyFlowAtomEncoder", "HeavyFlowConditionEncoder",
    "EquivariantAtomToResiduePool", "EquivariantResiduePoolingOutput", "irrep_chunks",
    "HistoryConfig", "HeavyFlowTemporalPairDataset", "ResidueGlobalContext",
    "SeqGeoResidueContext", "SequenceGeometryFusion", "equivariant_to_invariant", "invariant_width",
    "AtomContextOutput", "AtomContextRefiner", "HeavyFlowContextConfig", "HeavyFlowContextEncoding", "HeavyFlowContextEncoder",
    "AuxiliaryHeadOutput", "HeavyFlowAuxiliaryHeads", "from_mdcath_example", "bonds_from_topology", "collate_heavy_flow", "make_history",
    "PhysicsState", "PhysicsPredictorConfig", "PhysicsPredictor", "PhysicsModelOutput", "HeavyFlowPhysicsModel",
    "ForceDistribution", "ForceHeadOutput", "AtomForceDistribution", "ForceHead", "AtomForceHead",
    "ForceLossConfig", "ForceNormalizer", "heteroscedastic_force_nll", "masked_heteroscedastic_gaussian_nll", "force_nll", "masked_force_nll", "force_loss", "loss_with_diagnostics", "variance_inflation_diagnostic",
    "ForceTargetAudit", "compute_force_metrics", "force_metrics", "residue_net_force_torque", "zero_force_baseline", "mean_force_baseline", "unconditional_mean_variance_baseline", "shuffled_force_labels", "audit_force_targets", "target_audit",
    "PhysicsFrameKey", "PhysicsSplitManifest", "split_physics_frames", "HeavyFlowPhysicsFrameDataset", "build_physics_splits",
    "DomainChunk", "domain_order", "build_domain_chunks", "load_domain_chunks",
    "FlowAtomTopology", "FlowConditionBundle", "FlowCondition", "HeavyAtomFlowCondition", "build_flow_graph", "coerce_flow_topology",
    "FlowPath", "align_future_to_current", "sample_flow_base", "build_flow_path", "endpoint_from_velocity", "rectified_flow_loss", "rf_loss",
    "FlowDecoderConfig", "FlowDecoder", "HeavyAtomFlowDecoder",
    "GeometryLossConfig", "GeometryLossBreakdown", "geometry_regularization", "compute_geometry_regularization", "geometry_gradient_norms", "combined_flow_loss",
    "FlowSolverConfig", "integrate_flow", "euler_step", "heun_step", "HeavyAtomFlowModel", "sample_flow",
]
