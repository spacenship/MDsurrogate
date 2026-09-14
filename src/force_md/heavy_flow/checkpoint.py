"""Versioned architecture/normalizer handoff. Legacy reuse is explicit warm start."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping
import torch
from .force_losses import ForceNormalizer
from .history import HistoryConfig

ARCHITECTURE_VERSION = "heavy_flow_v2_geometry_context_pair_messages"
NORMALIZER_DEFINITION = "direct_heavy_atom_component_rms"


def architecture_config(config):
    result = deepcopy(dict(config))
    version = result.setdefault("architecture_version", ARCHITECTURE_VERSION)
    if version != ARCHITECTURE_VERSION:
        raise ValueError(f"incompatible architecture version: {version}")
    if "sequence_modality_dropout" in result.get("context", {}):
        raise ValueError("Stage 2 sequence fusion/dropout was removed; update the config")
    result["history"] = HistoryConfig.from_config(result).as_dict()
    return result


def validate_normalizer(values):
    normalizer = ForceNormalizer.from_dict(values)
    if normalizer.force_unit != "kcal/mol/angstrom" or normalizer.fit_source != "train":
        raise ValueError("normalizer must be a train-split direct heavy-atom RMS in kcal/mol/angstrom")
    return normalizer


def load_normalizer_artifact(path):
    """Reuse a scalar, retaining its original fit provenance; never scan data."""
    path = Path(path)
    payload = json.loads(path.read_text())
    definition = payload.get("definition", NORMALIZER_DEFINITION)
    if definition != NORMALIZER_DEFINITION:
        raise ValueError("incompatible force normalizer definition")
    spec = payload.get("fit_spec", {})
    scope = spec.get("represented_scope", spec.get("config", {}).get("data", {}).get("represented_scope", "heavy_atom"))
    if scope != "heavy_atom":
        raise ValueError("normalizer was not fit on direct heavy atoms")
    normalizer = validate_normalizer(payload.get("normalizer", payload))
    if normalizer.count <= 0:
        raise ValueError("reused normalizer must have fitted observations")
    provenance = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "definition": definition, "original_artifact": payload}
    return normalizer, provenance


def validate_checkpoint(payload, *, allow_untrained_fixture=False, require_trained=False):
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("incompatible architecture checkpoint; use explicit warm start and retrain Stage 3")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("checkpoint config is missing its architecture version")
    architecture_config(config)
    history = payload.get("history")
    if history != HistoryConfig.from_config(config).as_dict() or "history" not in config:
        raise ValueError("checkpoint history definition disagrees with config")
    validate_normalizer(payload["normalizer"])
    if payload.get("normalizer_definition") != NORMALIZER_DEFINITION:
        raise ValueError("checkpoint normalizer definition is missing or incompatible")
    training = payload.get("upstream_training", {})
    if require_trained:
        trained = training.get("status") == "force_optimized" and int(training.get("optimizer_steps", 0)) > 0
        fixture = training.get("status") == "fixture" and allow_untrained_fixture
        if not (trained or fixture):
            raise ValueError("Stage 4 requires force-trained v2 upstream; untrained fixture needs explicit test-only opt-in")
    return config


def warm_start_stage3(model, path):
    """Shape-compatible weights only; no optimizer reuse, no freezing."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = payload.get("upstream", payload.get("model_state"))
    if not isinstance(source, Mapping):
        raise ValueError("warm-start checkpoint has no upstream/model_state")
    target = model.state_dict()
    compatible = {key: value for key, value in source.items()
                  if key in target and value.shape == target[key].shape}
    missing = sorted(set(target) - set(compatible))
    unused = sorted(set(source) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    report = {"source": str(path), "source_architecture": payload.get("architecture_version", "legacy"),
              "loaded": sorted(compatible), "not_loaded": missing, "unused_source": unused,
              "requires_stage3_force_training": True}
    model.training_provenance = {"warm_start": report, "source_provenance": payload.get("provenance", {}),
                                 "source_step": payload.get("step")}
    return report


def finite_optimizer_gradients(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    if any(not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("non-finite gradient: optimizer step was not applied")
