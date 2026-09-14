from __future__ import annotations

from dataclasses import replace

import torch

from test_stage2_seq_geo import condition, context_model


def test_c_atom_is_equivariant_under_global_rotation():
    torch.manual_seed(9)
    model = context_model().eval()
    original_condition = condition()
    original = model(original_condition)
    theta = torch.tensor(0.51)
    rotation = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta), 0.0],
         [torch.sin(theta), torch.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rotated_condition = replace(
        original_condition,
        x_history=torch.einsum("bkni,ji->bknj", original_condition.x_history, rotation),
    )
    rotated = model(rotated_condition)
    representation = original.atom_context.irreps.D_from_matrix(rotation)
    expected = original.atom_context.atom_features @ representation.T
    assert torch.allclose(rotated.atom_context.atom_features, expected, atol=3e-4, rtol=3e-4)
