"""Attach typed, directed Z_t edges to geometry recomputed at each X_s."""
from dataclasses import replace
import torch
from .atom_graph import COVALENT_EDGE, SPATIAL_EDGE, CHAIN_ADJACENT_EDGE


def attach_physics_edges(graph, state, pair_dim):
    """Union dynamic neighbors and valid physics edges, keyed by atom IDs/type.

    New spatial neighbors receive zero Z and a false availability indicator.
    Original physics edges remain usable even after leaving the spatial cutoff.
    No coordinates or harmonics from t are cached in the returned graph.
    """
    state.validate()
    if state.atom_scalar.ndim == 2:
        if state.atom_batch is None or not torch.equal(state.atom_batch, graph.state.batch) or not torch.equal(state.atom_local, graph.state.local_atom):
            raise ValueError("physics atom mapping differs from decoder graph")
    if state.edge_scalar.shape[1] != pair_dim:
        raise ValueError("physics pair width differs from decoder")
    count = state.edge_scalar.shape[0]
    if not count:
        return graph, state.edge_scalar.new_zeros((graph.num_edges, pair_dim + 1))
    if state.edge_index is None or state.edge_kind is None or state.bond_type is None or state.edge_mask is None:
        raise ValueError("pair latents require edge index, kind, bond type, and mask")
    active = state.edge_mask
    old_rows = torch.cat((state.edge_index[:, active].T, state.edge_kind[active, None], state.bond_type[active, None]), dim=-1)
    if torch.unique(old_rows, dim=0).shape[0] != old_rows.shape[0]:
        raise ValueError("duplicate typed physics edges")
    dynamic_rows = torch.cat((graph.edge_index.T, graph.edge_kind[:, None], graph.bond_type[:, None]), dim=-1)
    rows, inverse = torch.unique(torch.cat((dynamic_rows, old_rows)), dim=0, return_inverse=True)
    pair = state.edge_scalar.new_zeros((rows.shape[0], pair_dim + 1))
    values = torch.cat((state.edge_scalar[active], state.edge_scalar.new_ones((int(active.sum()), 1))), dim=-1)
    pair = pair.index_copy(0, inverse[graph.num_edges:], values)
    kinds = rows[:, 2]
    degree = torch.bincount(rows[kinds == SPATIAL_EDGE, 1], minlength=graph.num_nodes)
    stats = replace(graph.stats, num_edges=rows.shape[0],
                    num_covalent_edges=int((kinds == COVALENT_EDGE).sum()),
                    num_spatial_edges=int((kinds == SPATIAL_EDGE).sum()),
                    num_chain_adjacent_edges=int((kinds == CHAIN_ADJACENT_EDGE).sum()),
                    maximum_spatial_degree=int(degree.max()) if degree.numel() else 0)
    return replace(graph, edge_index=rows[:, :2].T, edge_kind=kinds, bond_type=rows[:, 3],
                   edge_batch=graph.state.batch[rows[:, 0]], stats=stats), pair
