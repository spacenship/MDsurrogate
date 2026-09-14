from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from force_md.heavy_flow import (
    ForceHead,
    ForceNormalizer,
    PhysicsPredictor,
    PhysicsPredictorConfig,
    PhysicsState,
    compute_force_metrics,
    heteroscedastic_force_nll,
    split_physics_frames,
)

from tests.heavy_flow.test_stage2_seq_geo import condition, context_model


def small_predictor(encoding):
    return PhysicsPredictor(
        encoding.atom_context.irreps,
        config=PhysicsPredictorConfig(
            physics_blocks=1, scalar_dim=12, vector_channels=2, axial_channels=1,
            pair_dim=5, radial_basis=4, edge_embedding_dim=3,
        ),
    )


def test_predictor_and_head_boundaries_have_no_ground_truth_force_argument():
    assert "force" not in inspect.signature(PhysicsPredictor.forward).parameters
    assert list(inspect.signature(ForceHead.forward).parameters) == ["self", "state"]


def test_physics_state_and_force_head_shapes():
    torch.manual_seed(20)
    encoding = context_model().eval()(condition())
    state = small_predictor(encoding).eval()(encoding)
    output = ForceHead(12, vector_channels=2).eval()(state)
    assert state.atom_scalar.shape == (6, 12)
    assert state.atom_vector.shape == (6, 2, 3)
    assert state.atom_axial.shape == (6, 1, 3)
    assert output.force_mean.shape == (1, 6, 3)
    assert output.force_logvar.shape == (1, 6, 1)
    dense = PhysicsState(torch.randn(2, 4, 12), torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.empty(0, 5))
    assert ForceHead(12)(dense).force_mean.shape == (2, 4, 3)


def test_physics_force_rotation_equivariance_and_invariant_variance():
    torch.manual_seed(21)
    context = context_model().eval()
    predictor = small_predictor(context(condition())).eval()
    head = ForceHead(12, vector_channels=2).eval()
    base = condition()
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(base, x_history=torch.einsum("bkni,ji->bknj", base.x_history, rotation))
    first = head(predictor(context(base)))
    second = head(predictor(context(rotated)))
    assert torch.allclose(second.force_mean, first.force_mean @ rotation.T, atol=3e-4, rtol=3e-4)
    assert torch.allclose(second.force_logvar, first.force_logvar, atol=3e-4, rtol=3e-4)


def test_atom_permutation_equivariance():
    torch.manual_seed(22)
    context = context_model().eval()
    base = condition()
    encoding = context(base)
    predictor = small_predictor(encoding).eval()
    head = ForceHead(12, vector_channels=2).eval()
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    bond = base.bond_index.clone()
    valid = bond >= 0
    bond[valid] = inverse[bond[valid]]
    permuted = replace(
        base, atom_type=base.atom_type[:, permutation], atom_name=base.atom_name[:, permutation],
        atom_to_residue=base.atom_to_residue[:, permutation], atom_mask=base.atom_mask[:, permutation],
        x_history=base.x_history[:, :, permutation], bond_index=bond,
        is_backbone=base.is_backbone[:, permutation], is_sidechain=base.is_sidechain[:, permutation],
    )
    first = head(predictor(encoding))
    second = head(predictor(context(permuted)))
    assert torch.allclose(second.force_mean, first.force_mean[:, permutation], atol=4e-4, rtol=4e-4)
    assert torch.allclose(second.force_logvar, first.force_logvar[:, permutation], atol=4e-4, rtol=4e-4)


def test_force_nll_is_finite_with_all_mask_and_has_finite_gradients():
    mean = torch.randn(2, 4, 3, requires_grad=True)
    logvar = torch.randn(2, 4, 1, requires_grad=True)
    target = torch.randn(2, 4, 3)
    objective = heteroscedastic_force_nll(mean, logvar, target, torch.zeros(2, 4, dtype=torch.bool))
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(mean.grad).all() and torch.isfinite(logvar.grad).all()


def test_normalizer_roundtrip_metrics_and_temporal_split():
    force = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    normalizer = ForceNormalizer.fit(force, torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(normalizer.inverse_transform(normalizer.transform(force)), force)
    metrics = compute_force_metrics(force, force, force_logvar=torch.zeros(1, 2, 1), normalizer=normalizer)
    assert metrics["rmse"] == 0.0 and metrics["coverage_95"] == 1.0
    index = [("a", "320", "0", frame) for frame in range(32)] + [("b", "320", "0", frame) for frame in range(32)]
    manifest = split_physics_frames(index, same_domain_fraction=0.5, unseen_domain_fraction=0.5, temporal_block_size=8, seed=5)
    train, same, unseen = set(manifest.train), set(manifest.same_domain_validation), set(manifest.unseen_domain_validation)
    assert not train & same and not train & unseen and not same & unseen
    assert {x.domain for x in unseen} == {"b"}
    blocks = {tuple(block) for block in manifest.metadata["same_domain_validation_blocks"]}
    assert all((x.domain, x.temperature, x.replica, x.frame // 8) not in blocks for x in manifest.train)


def test_force_array_normalizer_uses_only_valid_represented_rows():
    samples = [
        (
            torch.tensor([[3.0, 0.0, 0.0], [100.0, 100.0, 100.0]]),
            torch.tensor([True, False]),
        ),
        (
            torch.tensor([[0.0, 4.0, 0.0]]),
            torch.tensor([True]),
        ),
    ]
    normalizer = ForceNormalizer.fit_from_force_arrays(samples)
    assert normalizer.count == 2
    assert normalizer.scale == pytest.approx((25.0 / 6.0) ** 0.5)
