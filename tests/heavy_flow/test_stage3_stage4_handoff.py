from __future__ import annotations

import pytest
import torch

from experiments.heavy_flow.train_flow import (
    build_stage4_model,
    load_stage4_checkpoint,
    save_stage4_checkpoint,
)
from experiments.heavy_flow.train_physics import build_stage3_model
from force_md.heavy_flow.checkpoint import ARCHITECTURE_VERSION, NORMALIZER_DEFINITION
from force_md.heavy_flow.history import HistoryConfig
from force_md.heavy_flow.force_losses import ForceNormalizer


def fixture_metadata():
    return {"architecture_version": ARCHITECTURE_VERSION, "history": HistoryConfig().as_dict(),
            "normalizer": ForceNormalizer().as_dict(), "normalizer_definition": NORMALIZER_DEFINITION,
            "upstream_training": {"status": "fixture", "optimizer_steps": 0}}


def _stage3_config() -> dict:
    return {
        "stage": "stage3_atom_physics",
        "architecture_version": ARCHITECTURE_VERSION,
        "history": HistoryConfig().as_dict(),
        "sequence_encoder": {
            "backend": "stub",
            "stub_dim": 12,
            "projected_dim": 8,
            "stub_seed": 17,
        },
        "geometry": {
            "num_blocks": 1,
            "hidden_irreps": "8x0e + 2x1o + 1x1e + 1x2e",
            "radial_basis": 6,
        },
        "graph": {"max_spatial_neighbors": 3},
        "context": {
            "joint_scalar_dim": 32,
            "fusion_blocks": 2,
            "attention_heads": 4,
            "dropout": 0.0,
            "atom_refine_blocks": 2,
            "attention_hidden_dim": 16,
            "identity_embedding_dim": 4,
        },
        "physics": {
            "physics_blocks": 1,
            "scalar_dim": 12,
            "vector_channels": 2,
            "axial_channels": 1,
            "pair_dim": 5,
            "radial_basis": 4,
            "edge_embedding_dim": 3,
        },
        "force_head": {"logvar_min": -10.0, "logvar_max": 10.0},
    }


def _stage4_config(stage3: dict) -> dict:
    return {
        **stage3,
        "stage": "stage4_heavy_atom_flow",
        "flow": {
            "flow_blocks": 2,
            "hidden_irreps": "8x0e + 2x1o + 1x1e + 1x2e",
            "conditioning_layers": [0, 1],
            "radial_basis": 4,
            "edge_embedding_dim": 3,
            "time_embedding_dim": 8,
            "atom_identity_dim": 4,
            "max_spatial_neighbors": 3,
            "freeze_upstream": True,
        },
    }


def test_stage4_strictly_loads_complete_stage3_upstream(tmp_path):
    torch.manual_seed(91)
    stage3_config = _stage3_config()
    stage3 = build_stage3_model(stage3_config, device="cpu")
    stage3_state = stage3.state_dict()
    stage3_path = tmp_path / "stage3.pt"
    torch.save(
        {
            **fixture_metadata(),
            "stage": "stage3_atom_physics",
            "config": stage3_config,
            "upstream": stage3_state,
        },
        stage3_path,
    )

    model = build_stage4_model(
        _stage4_config(stage3_config),
        stage3_checkpoint=stage3_path,
        device="cpu",
        allow_untrained_fixture=True,
    )

    assert model.upstream_is_frozen
    assert model.training
    assert not model.context_encoder.training
    assert not model.predictor.training
    assert not model.force_head.training
    model.train()
    assert not model.context_encoder.training
    assert not model.predictor.training
    assert not model.force_head.training
    assert model.upstream_provenance["sha256"]
    for key, value in stage3_state.items():
        assert torch.equal(model.state_dict()[key], value), key
    assert model.context_encoder.geometry_encoder.sequence_encoder.projection.weight.requires_grad is False


def test_stage4_checkpoint_resume_reuses_embedded_stage3_upstream(tmp_path):
    torch.manual_seed(92)
    stage3_config = _stage3_config()
    stage3 = build_stage3_model(stage3_config, device="cpu")
    stage3_path = tmp_path / "stage3.pt"
    torch.save(
        {**fixture_metadata(), "stage": "stage3_atom_physics", "config": stage3_config, "model_state": stage3.state_dict()},
        stage3_path,
    )
    stage4_config = _stage4_config(stage3_config)
    model = build_stage4_model(stage4_config, stage3_checkpoint=stage3_path, device="cpu", allow_untrained_fixture=True)
    stage4_path = tmp_path / "stage4.pt"
    save_stage4_checkpoint(stage4_path, model, config=stage4_config, step=3)

    raw = torch.load(stage4_path, map_location="cpu", weights_only=False)
    assert "upstream" in raw
    assert raw["upstream_config"] == stage3_config
    assert raw["upstream_provenance"]["stage"] == "stage3_atom_physics"

    resumed, checkpoint = load_stage4_checkpoint(stage4_path, device="cpu", allow_untrained_fixture=True)
    assert checkpoint["step"] == 3
    assert resumed.upstream_is_frozen
    for key, value in model.state_dict().items():
        assert torch.equal(resumed.state_dict()[key], value), key


def test_stage4_requires_stage3_checkpoint():
    with pytest.raises(ValueError, match="requires a Stage 3 checkpoint"):
        build_stage4_model(_stage4_config(_stage3_config()), device="cpu")
