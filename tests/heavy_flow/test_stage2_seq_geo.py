from __future__ import annotations

from dataclasses import replace

import torch
from e3nn import o3

from force_md.heavy_flow import (
    ESMCConfig,
    HeavyFlowAtomEncoder,
    HeavyFlowAuxiliaryHeads,
    HeavyFlowCondition,
    HeavyFlowContextConfig,
    HeavyFlowContextEncoder,
    HeavyFlowGeometryConfig,
    HeavyFlowGraphConfig,
    EquivariantAtomToResiduePool,
    irrep_chunks,
)


def condition(*, empty_middle: bool = False, offset: float = 0.0) -> HeavyFlowCondition:
    sequence = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    residue_mask = torch.tensor([[True, False, True]], dtype=torch.bool) if empty_middle else torch.ones((1, 3), dtype=torch.bool)
    atom_to_residue = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.int64)
    atom_type = torch.tensor([[6, 7, 8, 6, 7, 8]], dtype=torch.int64)
    atom_name = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.int64)
    if empty_middle:
        atom_to_residue = torch.tensor([[0, 0, 2, 2, 2, 2]], dtype=torch.int64)
        atom_type = torch.tensor([[6, 7, 6, 7, 8, 6]], dtype=torch.int64)
        atom_name = torch.tensor([[1, 2, 1, 2, 3, 1]], dtype=torch.int64)
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [3.0, 0.0, 0.0],
         [4.3, 0.0, 0.0], [6.0, 1.0, 0.0], [7.0, 1.0, 0.0]], dtype=torch.float32
    ) + offset
    frames = torch.stack([base + (i - 2) * 0.05 for i in range(3)], dim=0).unsqueeze(0)
    bond_index = torch.tensor([[[0, 0, 3], [1, 2, 4]]], dtype=torch.int64)
    return HeavyFlowCondition(
        sequence, residue_mask, atom_type, atom_name, atom_to_residue,
        torch.ones((1, 6), dtype=torch.bool), frames, bond_index,
        torch.ones((1, 3), dtype=torch.int64), torch.tensor([320.0]), torch.tensor([1.0]),
        is_backbone=torch.tensor([[True, True, True, False, False, False]], dtype=torch.bool),
        is_sidechain=torch.tensor([[False, False, False, True, True, True]], dtype=torch.bool),
        terminal_residue=torch.tensor([[True, False, True]], dtype=torch.bool),
        chain_break=torch.tensor([[False, False]], dtype=torch.bool),
    )


def context_model() -> HeavyFlowContextEncoder:
    geometry = HeavyFlowAtomEncoder(
        geometry_config=HeavyFlowGeometryConfig(
            num_blocks=1, hidden_irreps="8x0e + 2x1o + 1x1e + 1x2e", radial_basis=6
        ),
        graph_config=HeavyFlowGraphConfig(max_spatial_neighbors=3),
        esmc_config=ESMCConfig(backend="stub", stub_dim=12, projected_dim=8),
    )
    return HeavyFlowContextEncoder(
        geometry_encoder=geometry,
        context_config=HeavyFlowContextConfig(
            joint_scalar_dim=32, fusion_blocks=2, attention_heads=4, dropout=0.0,
            atom_refine_blocks=2, attention_hidden_dim=16, identity_embedding_dim=4,
        ),
    )


def test_atom_to_residue_pool_is_permutation_invariant_and_handles_empty_residue():
    torch.manual_seed(3)
    irreps = o3.Irreps("3x0e + 2x1o + 1x1e + 1x2e")
    pool = EquivariantAtomToResiduePool(irreps, attention_input_dim=5, attention_hidden_dim=12)
    features = torch.randn(6, irreps.dim)
    attention = torch.randn(6, 5)
    batch = torch.zeros(6, dtype=torch.long)
    residue = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = torch.ones((1, 3), dtype=torch.bool)
    first = pool(features, atom_batch=batch, atom_to_residue=residue, residue_mask=mask, attention_features=attention)
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    second = pool(features[permutation], atom_batch=batch, atom_to_residue=residue[permutation], residue_mask=mask, attention_features=attention[permutation])
    assert torch.allclose(first.padded_features, second.padded_features)

    empty = pool(
        features[[0, 1, 4, 5]], atom_batch=batch[:4], atom_to_residue=torch.tensor([0, 0, 2, 2]),
        residue_mask=torch.tensor([[True, False, True]]), attention_features=attention[[0, 1, 4, 5]],
    )
    assert torch.equal(empty.padded_features[:, 1], torch.zeros_like(empty.padded_features[:, 1]))
    expected_weights = torch.stack([first.attention_weights[0], first.attention_weights[1], first.attention_weights[4], first.attention_weights[5]]).detach()
    assert torch.allclose(empty.attention_weights.detach(), expected_weights)


def test_atom_to_residue_pool_preserves_rotation_equivariance():
    torch.manual_seed(4)
    irreps = o3.Irreps("2x0e + 1x1o + 1x1e + 1x2e")
    pool = EquivariantAtomToResiduePool(irreps, attention_input_dim=4, attention_hidden_dim=10)
    features = torch.randn(5, irreps.dim)
    attention = torch.randn(5, 4)
    batch = torch.zeros(5, dtype=torch.long)
    residue = torch.tensor([0, 0, 1, 1, 1])
    mask = torch.ones((1, 2), dtype=torch.bool)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first = pool(features, atom_batch=batch, atom_to_residue=residue, residue_mask=mask, attention_features=attention)
    rotated = pool(features @ irreps.D_from_matrix(rotation).T, atom_batch=batch, atom_to_residue=residue, residue_mask=mask, attention_features=attention)
    assert torch.allclose(rotated.residue_features, first.residue_features @ irreps.D_from_matrix(rotation).T, atol=2e-5, rtol=2e-5)


def test_context_broadcast_keeps_atom_order_and_atom_identity():
    torch.manual_seed(5)
    model = context_model().eval()
    out = model(condition())
    assert torch.equal(out.padded_atom_features[0, 0], out.atom_features[0])
    assert torch.equal(out.padded_atom_features[0, 5], out.atom_features[5])
    assert not torch.allclose(out.atom_features[0], out.atom_features[1])
    assert not torch.allclose(out.atom_features[2], out.atom_features[3])


def test_scalar_transformer_boundary_frozen_esm_and_rotation_invariant_joint():
    torch.manual_seed(6)
    model = context_model().eval()
    c = condition()
    out = model(c)
    assert all(irrep.l == 0 for _, irrep, _ in irrep_chunks(model.fusion.geometry_input_irreps))
    frozen_before = out.geometry.sequence.frozen_embedding.clone()
    assert not hasattr(out.residue_context, "esm_frozen")
    import inspect
    assert "sequence" not in inspect.signature(model.fusion.forward).parameters
    assert "condition" not in inspect.signature(model.fusion.forward).parameters
    assert not hasattr(model.fusion, "esm_projection")
    theta = torch.tensor(0.47)
    rotation = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0], [torch.sin(theta), torch.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(c, x_history=torch.einsum("bkni,ji->bknj", c.x_history, rotation))
    rotated_out = model(rotated)
    assert torch.allclose(out.residue_context.joint_scalar, rotated_out.residue_context.joint_scalar, atol=2e-4, rtol=2e-4)
    assert torch.equal(out.geometry.sequence.frozen_embedding, frozen_before)


def test_empty_residue_and_modality_dropouts_change_outputs():
    torch.manual_seed(7)
    model = context_model().eval()
    c = condition(empty_middle=True)
    out = model(c)
    assert torch.equal(out.residue_context.joint_scalar[:, 1], torch.zeros_like(out.residue_context.joint_scalar[:, 1]))
    normal = model(condition())
    changed = condition()
    changed.sequence_tokens = (changed.sequence_tokens + 1) % 20
    no_sequence = model(changed)
    no_geometry = model(condition(), geometry_dropout=True)
    assert not torch.allclose(normal.residue_context.joint_scalar, no_sequence.residue_context.joint_scalar)
    assert not torch.allclose(normal.residue_context.joint_scalar, no_geometry.residue_context.joint_scalar)


def test_joint_and_atom_refinement_gradients_and_two_auxiliary_losses_decrease():
    torch.manual_seed(8)
    model = context_model()
    c = condition()
    model.eval()
    encoding = model(c)
    heads = HeavyFlowAuxiliaryHeads(
        atom_irreps=encoding.atom_context.irreps,
        residue_joint_dim=encoding.residue_context.joint_scalar.shape[-1],
        geometry_invariant_dim=encoding.residue_context.geometry_invariant.shape[-1],
        hidden_dim=32,
    )
    loss = heads.loss(encoding, c)["total"] + encoding.atom_context.atom_features.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.fusion.parameters() if parameter.requires_grad)
    assert any(parameter.grad is not None for parameter in model.refiner.parameters() if parameter.requires_grad)
    assert torch.isfinite(loss)

    with torch.no_grad():
        fixed_encoding = model(c)
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.02)
    initial = float(heads.loss(fixed_encoding, c)["total"].detach())
    for _ in range(12):
        optimizer.zero_grad()
        objective = heads.loss(fixed_encoding, c)["total"]
        assert torch.isfinite(objective)
        objective.backward()
        optimizer.step()
    final = float(heads.loss(fixed_encoding, c)["total"].detach())
    assert final < initial
