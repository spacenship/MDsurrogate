from __future__ import annotations

import torch

from force_md.data.residue_constants import ATOM_NAME_TO_ID
from force_md.heavy_flow import (
    CHAIN_ADJACENT_EDGE,
    HeavyFlowCondition,
    HeavyFlowGraphConfig,
    build_atom_graph,
)


def _two_residue_condition(*, chain_break: bool) -> HeavyFlowCondition:
    """Make an isolated C(r)--N(r+1) fixture with no PSF bonds."""
    condition = HeavyFlowCondition(
        sequence_tokens=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        residue_mask=torch.ones((1, 3), dtype=torch.bool),
        atom_type=torch.tensor([[6, 6, 7, 6]], dtype=torch.int64),
        atom_name=torch.tensor([[
            ATOM_NAME_TO_ID["C"], ATOM_NAME_TO_ID["CA"],
            ATOM_NAME_TO_ID["N"], ATOM_NAME_TO_ID["CA"],
        ]], dtype=torch.int64),
        atom_to_residue=torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
        atom_mask=torch.ones((1, 4), dtype=torch.bool),
        x_history=torch.tensor([[[
            [0.0, 0.0, 0.0], [1.3, 0.0, 0.0],
            [20.0, 0.0, 0.0], [21.3, 0.0, 0.0],
        ]]], dtype=torch.float32),
        bond_index=torch.empty((1, 2, 0), dtype=torch.int64),
        bond_type=torch.empty((1, 0), dtype=torch.int64),
        temperature=torch.tensor([320.0]),
        lag=torch.tensor([1.0]),
        is_backbone=torch.tensor([[True, True, True, False]], dtype=torch.bool),
        is_sidechain=torch.tensor([[False, False, False, True]], dtype=torch.bool),
        terminal_residue=torch.tensor([[True, True, False]], dtype=torch.bool),
        chain_break=torch.tensor([[chain_break, False]], dtype=torch.bool),
    )
    condition.validate()
    return condition


def test_chain_break_suppresses_only_explicit_chain_adjacent_edge():
    config = HeavyFlowGraphConfig(
        spatial_cutoff_angstrom=0.5,
        max_spatial_neighbors=1,
        include_covalent_edges=False,
        include_chain_adjacent_edges=True,
    )
    connected = build_atom_graph(_two_residue_condition(chain_break=False), config)
    broken = build_atom_graph(_two_residue_condition(chain_break=True), config)

    assert connected.stats.num_chain_adjacent_edges == 2
    assert torch.all(connected.edge_kind == CHAIN_ADJACENT_EDGE)
    assert broken.stats.num_chain_adjacent_edges == 0
    assert broken.edge_index.shape == (2, 0)
