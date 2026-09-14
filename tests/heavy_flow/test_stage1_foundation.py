from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
from e3nn import o3

from force_md.heavy_flow import (
    COVALENT_EDGE,
    ESMCCacheMiss,
    ESMCConfig,
    ESMCEncoder,
    HeavyFlowAtomEncoder,
    HeavyFlowCondition,
    HeavyFlowGeometryConfig,
    HeavyFlowGraphConfig,
    HeavyFlowSample,
    HeavyFlowTargets,
    build_atom_graph,
    collate_heavy_flow,
    equivariant_to_invariant,
)
from force_md.heavy_flow.geometry_encoder import _EquivariantNormActivation


def condition(*, atoms: int = 6, history: int = 3, offset: float = 0.0) -> HeavyFlowCondition:
    seq = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    residue_mask = torch.ones((1, 3), dtype=torch.bool)
    atom_to_residue = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.int64)[:, :atoms]
    atom_type = torch.tensor([[6, 7, 8, 6, 7, 8]], dtype=torch.int64)[:, :atoms]
    atom_name = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.int64)[:, :atoms]
    atom_mask = torch.ones((1, atoms), dtype=torch.bool)
    base = torch.tensor([
        [0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [20.0, 0.0, 0.0],
        [21.3, 0.0, 0.0], [22.0, 1.0, 0.0], [23.0, 1.0, 0.0],
    ], dtype=torch.float32)[:atoms] + offset
    frames = torch.stack([base + (i - history + 1) * 0.05 for i in range(history)], dim=0)
    edge_count = 2 if atoms < 5 else 3
    bond_index = torch.tensor([[[0, 0, 3], [1, 2, 4]]], dtype=torch.int64)[:, :, :edge_count]
    bond_type = torch.ones((1, edge_count), dtype=torch.int64)
    return HeavyFlowCondition(
        seq, residue_mask, atom_type, atom_name, atom_to_residue, atom_mask,
        frames.unsqueeze(0), bond_index, bond_type,
        torch.tensor([320.0]), torch.tensor([1.0]),
        is_backbone=torch.tensor([[True, True, True, False, False, False]], dtype=torch.bool)[:, :atoms],
        is_sidechain=torch.tensor([[False, False, False, True, True, True]], dtype=torch.bool)[:, :atoms],
        terminal_residue=torch.tensor([[True, False, True]], dtype=torch.bool),
        chain_break=torch.tensor([[False, False]], dtype=torch.bool),
    )


def model(*, projected_dim: int = 12) -> HeavyFlowAtomEncoder:
    return HeavyFlowAtomEncoder(
        geometry_config=HeavyFlowGeometryConfig(
            num_blocks=1,
            hidden_irreps="16x0e + 4x1o + 2x1e + 2x2e",
            radial_basis=8,
        ),
        graph_config=HeavyFlowGraphConfig(spatial_cutoff_angstrom=6.0, max_spatial_neighbors=3),
        esmc_config=ESMCConfig(backend="stub", stub_dim=24, projected_dim=projected_dim),
    )


def test_condition_validation_and_target_ordering():
    c = condition()
    c.validate()
    future = c.current_positions + 1.0
    target = HeavyFlowTargets(torch.randn(1, 6, 3), future)
    target.validate(c)
    assert torch.equal(c.current_positions, c.x_history[:, -1])
    assert target.force_current.shape == future.shape
    bad = replace(c, atom_mask=torch.tensor([[True, True, True, True, False, False]]))
    with pytest.raises(ValueError, match="bonds"):
        bad.validate()


def test_covalent_edges_survive_radius_and_no_wrong_chain_bond():
    c = condition()
    graph = build_atom_graph(c, HeavyFlowGraphConfig(spatial_cutoff_angstrom=6.0, max_spatial_neighbors=32))
    covalent = graph.edge_kind == COVALENT_EDGE
    pairs = {tuple(sorted(x)) for x in graph.edge_index[:, covalent].t().tolist()}
    assert (0, 2) in pairs  # 20 A apart but explicitly covalent
    assert (2, 3) not in pairs  # no supplied peptide bond is invented


def test_atom_permutation_equivariance_and_translation_invariance():
    torch.manual_seed(4)
    c = condition()
    m = model()
    original = m(c).atom_features.detach()
    perm = torch.tensor([2, 0, 5, 3, 1, 4])
    inverse = torch.argsort(perm)
    remapped_bonds = inverse[c.bond_index]
    permuted = replace(c, atom_type=c.atom_type[:, perm], atom_name=c.atom_name[:, perm],
                       atom_to_residue=c.atom_to_residue[:, perm], atom_mask=c.atom_mask[:, perm],
                       x_history=c.x_history[:, :, perm], bond_index=remapped_bonds,
                       is_backbone=c.is_backbone[:, perm], is_sidechain=c.is_sidechain[:, perm])
    permuted_out = m(permuted).atom_features.detach()
    assert torch.allclose(permuted_out, original[perm], atol=3e-5, rtol=3e-5)
    translated = replace(c, x_history=c.x_history + torch.tensor([11.0, -4.0, 7.0]))
    translated_out = m(translated).atom_features.detach()
    assert torch.allclose(translated_out, original, atol=3e-5, rtol=3e-5)


def test_rotation_equivariance_of_vector_and_tensor_irreps():
    torch.manual_seed(8)
    c = condition()
    m = model()
    out = m(c)
    theta = torch.tensor(0.63)
    rotation = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = replace(c, x_history=torch.einsum("bkni,ji->bknj", c.x_history, rotation))
    rotated_out = m(rotated).atom_features.detach()
    representation = out.irreps.D_from_matrix(rotation)
    expected = out.atom_features.detach() @ representation.T
    assert torch.allclose(rotated_out, expected, atol=3e-4, rtol=3e-4)


def test_esmc_cache_is_sequence_only_and_strict(tmp_path):
    c = condition()
    config = ESMCConfig(backend="stub", stub_dim=20, projected_dim=10, cache_dir=str(tmp_path))
    encoder = ESMCEncoder(config)
    first = encoder(c)
    changed = replace(c, x_history=c.x_history + 2.0, temperature=torch.tensor([450.0]))
    second = encoder(changed)
    assert torch.equal(first.frozen_embedding, second.frozen_embedding)
    assert torch.equal(first.projected_embedding, second.projected_embedding)
    strict = ESMCEncoder(ESMCConfig(backend="stub", stub_dim=20, projected_dim=10,
                                    cache_dir=str(tmp_path), cache_only=True))
    loaded = strict(c)
    assert torch.equal(loaded.frozen_embedding, first.frozen_embedding)
    missing = ESMCEncoder(ESMCConfig(backend="stub", stub_dim=20, projected_dim=10,
                                     cache_dir=str(tmp_path / "missing"), cache_only=True))
    with pytest.raises(ESMCCacheMiss):
        missing(c)


def test_esmc_frozen_detached_and_projection_trainable(tmp_path):
    c = condition()
    encoder = ESMCEncoder(ESMCConfig(backend="stub", stub_dim=20, projected_dim=10, cache_dir=str(tmp_path)))
    first = encoder(c)
    assert first.frozen_embedding.dtype == torch.float32
    assert not first.frozen_embedding.requires_grad
    assert all(not p.requires_grad for p in encoder.backend.parameters())
    first.projected_embedding.square().mean().backward()
    assert encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in encoder.backend.parameters())


def test_condition_encoder_cannot_receive_ground_truth():
    assert list(inspect.signature(HeavyFlowAtomEncoder.forward).parameters) == ["self", "condition"]
    with pytest.raises(TypeError):
        model()(condition(), force_current=torch.zeros(1, 6, 3))


def test_padded_collate_matches_individual_packed_outputs():
    torch.manual_seed(10)
    first = condition(atoms=4)
    second = condition(atoms=6, offset=0.4)
    first = replace(first, bond_index=first.bond_index[:, :, :1], bond_type=first.bond_type[:, :1])
    target1 = HeavyFlowTargets(torch.zeros(1, 4, 3), first.current_positions.clone())
    target2 = HeavyFlowTargets(torch.zeros(1, 6, 3), second.current_positions.clone())
    batch = collate_heavy_flow([HeavyFlowSample(first, target1), HeavyFlowSample(second, target2)])
    m = model()
    packed = m(batch.condition).padded_features
    out1 = m(first).padded_features[0, :4]
    out2 = m(second).padded_features[0, :6]
    assert torch.allclose(packed[0, :4], out1, atol=3e-5, rtol=3e-5)
    assert torch.allclose(packed[1, :6], out2, atol=3e-5, rtol=3e-5)
    assert torch.equal(packed[0, 4:], torch.zeros_like(packed[0, 4:]))


def test_synthetic_forward_backward_smoke():
    m = model()
    c = condition()
    out = m(c)
    out.atom_features.square().mean().backward()
    assert out.atom_features.shape[0] == int(c.atom_mask.sum())
    assert m.trainable_parameter_count > 0


def test_zero_irrep_norm_backward_is_finite():
    """Zero disallowed irrep channels must not poison the first optimizer step."""
    irreps = o3.Irreps("2x0e + 2x1e")
    norm = _EquivariantNormActivation(irreps)
    values = torch.zeros((3, norm.irreps.dim), dtype=torch.float32, requires_grad=True)
    output = norm(values)
    assert torch.equal(output, torch.zeros_like(output))
    output.square().sum().backward()
    assert torch.isfinite(values.grad).all()

    equivariant = torch.zeros((3, norm.irreps.dim), dtype=torch.float32, requires_grad=True)
    invariant = equivariant_to_invariant(equivariant, norm.irreps)
    assert torch.equal(invariant, torch.zeros_like(invariant))
    invariant.sum().backward()
    assert torch.isfinite(equivariant.grad).all()
