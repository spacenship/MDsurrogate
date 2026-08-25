"""Phase 1.5: the transition probe.

A deliberately small vertical slice that answers one question -- does Phase 1's
force-supervised representation help predict a 1-4 ns future structure, compared
with a structure-only baseline of the same size and budget? It is **not** a
stochastic flow model, and nothing here should be described as one.
"""

from .metrics import (
    METRIC_KEYS,
    MetricConfig,
    aggregate_metric_records,
    metric_records,
    per_graph_transition_metrics,
    transition_metrics,
)
from .conditioners import (
    CONDITIONER_ARMS,
    ConditionerConfig,
    ForcePatternShapeConditioner,
    OracleAtomicForceConditioner,
    PhysicsLatentConditioner,
    ResidueForceTorqueConditioner,
    TransitionConditioner,
    ZeroConditioner,
    build_conditioner,
    precision_weights,
)
from .local_frame import IrrepsLocalFrame
from .losses import TransitionLossWeights, transition_loss
from .moments import ForceMoments, ResidueShape, force_moments, residue_shape
from .phase1_features import (
    FeatureBundle,
    FrozenPhase1Extractor,
    OracleFeatureBundle,
    Phase1FeatureCache,
    assert_production_safe,
    checkpoint_fingerprint,
    merge_bundles,
    split_bundle,
)
from .probe import (
    DISPLACEMENT_FEATURE_DIM,
    HISTORY_FEATURE_DIM,
    LAG_FEATURE_DIM,
    TransitionProbe,
    TransitionProbeConfig,
    displacement_features,
    lag_features,
)
from .targets import (
    TransitionPrediction,
    TransitionTarget,
    apply_prediction,
    build_transition_target,
    identity_prediction,
    reconstruct_backbone,
    target_as_prediction,
)

__all__ = [
    "TransitionTarget",
    "TransitionPrediction",
    "build_transition_target",
    "identity_prediction",
    "target_as_prediction",
    "apply_prediction",
    "reconstruct_backbone",
    "MetricConfig",
    "METRIC_KEYS",
    "per_graph_transition_metrics",
    "transition_metrics",
    "metric_records",
    "aggregate_metric_records",
    "FeatureBundle",
    "OracleFeatureBundle",
    "FrozenPhase1Extractor",
    "Phase1FeatureCache",
    "assert_production_safe",
    "checkpoint_fingerprint",
    "split_bundle",
    "merge_bundles",
    "ConditionerConfig",
    "TransitionConditioner",
    "ZeroConditioner",
    "ResidueForceTorqueConditioner",
    "PhysicsLatentConditioner",
    "ForcePatternShapeConditioner",
    "OracleAtomicForceConditioner",
    "CONDITIONER_ARMS",
    "build_conditioner",
    "precision_weights",
    "IrrepsLocalFrame",
    "ForceMoments",
    "ResidueShape",
    "force_moments",
    "residue_shape",
    "TransitionProbe",
    "TransitionProbeConfig",
    "lag_features",
    "LAG_FEATURE_DIM",
    "displacement_features",
    "DISPLACEMENT_FEATURE_DIM",
    "HISTORY_FEATURE_DIM",
    "TransitionLossWeights",
    "transition_loss",
]
