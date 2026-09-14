from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from force_md.heavy_flow import (
    FlowAtomTopology,
    FlowConditionBundle,
    FlowDecoder,
    FlowDecoderConfig,
    ForceHead,
    HeavyAtomFlowModel,
    HeavyFlowSample,
    HeavyFlowTargets,
    PhysicsPredictor,
    PhysicsPredictorConfig,
    build_flow_path,
    compute_geometry_regularization,
    integrate_flow,
    rectified_flow_loss,
    sample_flow_base,
)

from tests.heavy_flow.test_stage2_seq_geo import condition, context_model
from experiments.heavy_flow.train_flow import train_flow_steps


@pytest.fixture(scope="module")
def stage4_fixture():
    torch.manual_seed(41)
    c = condition()
    encoder = context_model().eval()
    context = encoder(c)
    predictor = PhysicsPredictor(
        context.atom_context.irreps,
        config=PhysicsPredictorConfig(
            physics_blocks=1, scalar_dim=12, vector_channels=2, axial_channels=1,
            pair_dim=5, radial_basis=4, edge_embedding_dim=3,
        ),
    ).eval()
    state = predictor(context)
    head = ForceHead(12, vector_channels=2).eval()
    distribution = head(state)
    state = state.with_force(distribution.force_mean, distribution.force_logvar)
    topology = FlowAtomTopology.from_graph(
        context.geometry.graph,
        atom_mask=c.atom_mask,
        atom_to_residue=c.atom_to_residue,
        atom_type=c.atom_type,
        atom_name=c.atom_name,
        residue_mask=c.residue_mask,
        chain_break=c.chain_break,
        max_spatial_neighbors=3,
    )
    bundle = FlowConditionBundle(
        context.atom_context.atom_features,
        context.residue_context.joint_scalar,
        state,
        distribution.force_mean,
        distribution.force_logvar,
        topology,
        c.temperature,
        c.lag,
        c.current_positions,
    )
    decoder = FlowDecoder(
        context.atom_context.irreps,
        context.residue_context.joint_scalar.shape[-1],
        12,
        physics_vector_channels=2,
        physics_axial_channels=1,
        physics_pair_dim=5,
        config=FlowDecoderConfig(
            flow_blocks=2,
            hidden_irreps="8x0e + 2x1o + 1x1e + 1x2e",
            conditioning_layers=(0, 1),
            radial_basis=4,
            edge_embedding_dim=3,
            time_embedding_dim=8,
            atom_identity_dim=4,
            max_spatial_neighbors=3,
        ),
    ).eval()
    model = HeavyAtomFlowModel(encoder, predictor, head, decoder)
    return c, bundle, model


def test_decoder_contract_has_no_future_or_ground_truth_force(stage4_fixture):
    parameters = list(inspect.signature(FlowDecoder.forward).parameters)
    assert parameters[:9] == [
        "self", "x_s", "flow_time", "atom_context", "residue_context",
        "physics_state", "atom_topology", "temperature", "lag",
    ]
    assert not {"x_future", "future", "target", "force_target", "gt_force"}.intersection(parameters)


def test_flow_path_boundaries_alignment_and_mask(stage4_fixture):
    c, _, _ = stage4_fixture
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    future = torch.einsum("bni,ji->bnj", c.current_positions, rotation) + torch.tensor([10.0, -4.0, 2.0])
    path = build_flow_path(
        c.current_positions,
        future,
        atom_mask=c.atom_mask,
        lag=c.lag,
        noise=torch.zeros_like(future),
        flow_time=torch.tensor([0.0]),
    )
    assert torch.allclose(path.x_s, path.x0)
    assert torch.allclose(path.x1, c.current_positions, atol=2e-5)
    assert torch.allclose(path.target_velocity, c.current_positions - path.x0, atol=1e-5)
    masked = c.atom_mask.clone()
    masked[0, -1] = False
    path_masked = build_flow_path(c.current_positions, future, atom_mask=masked, future_mask=masked, lag=c.lag, noise=torch.zeros_like(future))
    assert not path_masked.mask[0, -1]
    assert torch.equal(path_masked.x1[0, -1], c.current_positions[0, -1])


def test_decoder_rotation_equivariance_and_dynamic_graph(stage4_fixture):
    c, bundle, model = stage4_fixture
    decoder = model.decoder
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotate = lambda value: torch.einsum("...i,ji->...j", value, rotation)
    state = replace(
        bundle.physics_state,
        atom_vector=rotate(bundle.physics_state.atom_vector),
        atom_axial=rotate(bundle.physics_state.atom_axial),
        force_mean=rotate(bundle.physics_state.force_mean),
    )
    rotated = replace(
        bundle,
        atom_context=bundle.atom_context @ decoder.atom_context_irreps.D_from_matrix(rotation).T,
        physics_state=state,
        force_mean=rotate(bundle.force_mean),
        current_positions=rotate(bundle.current_positions),
    )
    x = c.current_positions
    first = decoder(x, torch.tensor([0.37]), bundle.atom_context, bundle.residue_context, bundle.physics_state, bundle.atom_topology, bundle.temperature, bundle.lag)
    decoder.reset_graph_cache()
    second = decoder(rotate(x), torch.tensor([0.37]), rotated.atom_context, rotated.residue_context, rotated.physics_state, rotated.atom_topology, rotated.temperature, rotated.lag)
    assert torch.allclose(second, rotate(first), atol=4e-5, rtol=4e-5)
    decoder.reset_graph_cache()
    decoder(x, torch.tensor([0.1]), bundle.atom_context, bundle.residue_context, bundle.physics_state, bundle.atom_topology, bundle.temperature, bundle.lag, recompute_spatial=True)
    decoder(rotate(x) + 20.0, torch.tensor([0.2]), bundle.atom_context, bundle.residue_context, bundle.physics_state, bundle.atom_topology, bundle.temperature, bundle.lag, recompute_spatial=False)
    assert decoder.spatial_graph_recomputations == 1


def test_decoder_is_consistent_under_atom_permutation(stage4_fixture):
    c, bundle, model = stage4_fixture
    decoder = model.decoder
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    topology = bundle.atom_topology
    permuted_topology = replace(
        topology,
        fixed_edge_index=inverse[topology.fixed_edge_index],
        atom_local=torch.arange(permutation.numel()),
        atom_to_residue=topology.atom_to_residue[:, permutation],
        atom_type=topology.atom_type[:, permutation],
        atom_name=topology.atom_name[:, permutation],
    )
    state = replace(
        bundle.physics_state,
        atom_scalar=bundle.physics_state.atom_scalar[permutation],
        atom_vector=bundle.physics_state.atom_vector[permutation],
        atom_axial=bundle.physics_state.atom_axial[permutation],
        force_mean=bundle.physics_state.force_mean[:, permutation],
        force_logvar=bundle.physics_state.force_logvar[:, permutation],
        edge_index=inverse[bundle.physics_state.edge_index],
    )
    decoder.reset_graph_cache()
    first = decoder(
        c.current_positions, c.lag * 0 + 0.41, bundle.atom_context,
        bundle.residue_context, bundle.physics_state, topology,
        bundle.temperature, bundle.lag,
    )
    decoder.reset_graph_cache()
    second = decoder(
        c.current_positions[:, permutation], c.lag * 0 + 0.41,
        bundle.atom_context[permutation], bundle.residue_context, state,
        permuted_topology, bundle.temperature, bundle.lag,
    )
    assert torch.allclose(second, first[:, permutation], atol=4e-5, rtol=4e-5)


def test_zeroed_physics_conditioning_changes_velocity(stage4_fixture):
    c, bundle, model = stage4_fixture
    zero_state = replace(
        bundle.physics_state,
        atom_scalar=torch.zeros_like(bundle.physics_state.atom_scalar),
        atom_vector=torch.zeros_like(bundle.physics_state.atom_vector),
        atom_axial=torch.zeros_like(bundle.physics_state.atom_axial),
        force_mean=torch.zeros_like(bundle.physics_state.force_mean),
        force_logvar=torch.zeros_like(bundle.physics_state.force_logvar),
    )
    zero_bundle = replace(
        bundle,
        physics_state=zero_state,
        force_mean=torch.zeros_like(bundle.force_mean),
        force_logvar=torch.zeros_like(bundle.force_logvar),
    )
    model.decoder.reset_graph_cache()
    conditioned = model.velocity(bundle, c.current_positions, torch.tensor([0.4]))
    model.decoder.reset_graph_cache()
    zeroed = model.velocity(zero_bundle, c.current_positions, torch.tensor([0.4]))
    assert torch.linalg.vector_norm(conditioned - zeroed) > 1e-7


def test_masked_rf_and_geometry_losses_are_finite_and_have_gradients(stage4_fixture):
    c, bundle, _ = stage4_fixture
    prediction = c.current_positions.clone().requires_grad_()
    target = c.current_positions + 0.1
    mask = c.atom_mask.clone()
    mask[0, -1] = False
    rf = rectified_flow_loss(prediction, target, mask)
    topology = replace(bundle.atom_topology, atom_mask=mask)
    geometry = compute_geometry_regularization(prediction, target, topology)
    objective = rf + geometry.total
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(prediction.grad).all()
    assert rectified_flow_loss(prediction.detach(), target, torch.zeros_like(mask)) == 0


def test_sampling_seed_reproducibility_and_solver_boundaries(stage4_fixture):
    _, bundle, model = stage4_fixture
    one = model.sample(bundle, num_samples=2, seed=77, solver="euler", steps=1)
    two = model.sample(bundle, num_samples=2, seed=77, solver="euler", steps=1)
    three = model.sample(bundle, num_samples=2, seed=78, solver="euler", steps=1)
    assert torch.equal(one, two)
    assert not torch.equal(one, three)
    assert one.shape == (2, 1, 6, 3)
    x0 = torch.zeros(1, 2, 3)
    result = integrate_flow(x0, lambda x, t, refresh: torch.ones_like(x), solver="heun", steps=1)
    multi = integrate_flow(x0, lambda x, t, refresh: torch.ones_like(x), solver="heun", steps=4)
    assert torch.allclose(result, torch.ones_like(result))
    assert torch.allclose(multi, torch.ones_like(multi))


def test_physics_conditioning_and_decoder_adapters_have_gradients(stage4_fixture):
    c, bundle, model = stage4_fixture
    model.train()
    velocity = model.velocity(bundle, c.current_positions, torch.tensor([0.4]))
    velocity.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.decoder.context_gate.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.residue_context_adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.physics_tensor_product.parameters())
    assert model.upstream_is_frozen
    assert all(parameter.grad is None for parameter in model.context_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to exercise CPU-to-GPU sample migration")
def test_train_flow_steps_moves_cpu_samples_to_model_device(stage4_fixture):
    c, _, model = stage4_fixture
    device = torch.device("cuda:0")
    model.to(device)
    sample = HeavyFlowSample(
        c,
        HeavyFlowTargets(
            force_current=torch.zeros_like(c.current_positions),
            x_future=c.current_positions + 0.05,
            force_mask=c.atom_mask.clone(),
            future_mask=c.atom_mask.clone(),
        ),
    )
    assert sample.condition.current_positions.device.type == "cpu"
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=1e-4)
    history = train_flow_steps(
        model,
        [sample],
        optimizer,
        steps=1,
        report_gradient_norms=False,
        seed=13,
    )
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["loss"]))
    assert next(model.decoder.parameters()).device.type == "cuda"
