"""Invariant energy head and its conservative force.

What this branch is, and what it is not:

* ``U`` is an invariant scalar built from ``l=0`` features, so it is unchanged by
  any rotation or translation of the structure.
* ``-dU/dx`` is a genuine conservative force field **of this learned potential**.
  It is *not* the full effective force on a coarse-grained atom, and the code
  never sets the two equal. mdCATH's forces come from an all-atom simulation with
  solvent; once solvent and momentum are integrated out the remaining effective
  force on the protein subsystem is not the gradient of any potential of the
  protein coordinates alone.
* mdCATH ships **no energy label**. The energy branch is therefore trained only
  through its gradient (as an auxiliary force term) and through a gauge
  regulariser -- never against a target energy that does not exist.
* The gradient is taken on a **fixed graph**. Coordinates enter through edge
  vectors and distances, and those are differentiable; the neighbour list is a
  discrete function of the coordinates and is not. Rebuilding the graph between
  the forward pass and the gradient would silently change the function being
  differentiated.
"""

from __future__ import annotations

from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..nn.blocks import extract_scalars, scalar_dim
from ..nn.irreps import scatter_sum

__all__ = ["InvariantEnergyHead", "conservative_force"]


class InvariantEnergyHead(nn.Module):
    """Per-residue invariant energy, summed to a per-graph total.

    Args:
        irreps_node: irreps of the residue features.
        per_residue: also return the per-residue decomposition. It is a latent
            decomposition of the learned potential, not a measurable per-residue
            energy, and is labelled as such.

    Shape:
        ``[N_res, D] -> (graph_energy [B], residue_energy [N_res])``.
    """

    def __init__(self, irreps_node: o3.Irreps, hidden: int = 64):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        # Reading only the l=0 block guarantees invariance by construction
        # rather than by hoping the linear layer learns it.
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim(self.irreps_node), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, residue_features: Tensor, batch_index: Tensor, num_graphs: int,
        residue_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        scalars = extract_scalars(residue_features, self.irreps_node)
        residue_energy = self.mlp(scalars).squeeze(-1)
        if residue_mask is not None:
            residue_energy = residue_energy * residue_mask.to(residue_energy.dtype)
        graph_energy = scatter_sum(
            residue_energy.unsqueeze(-1), batch_index, num_graphs
        ).squeeze(-1)
        return graph_energy, residue_energy


def conservative_force(
    energy: Tensor,
    positions: Tensor,
    *,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Tensor:
    """``-dU/dx`` on the current (fixed) graph.

    Args:
        energy: scalar or ``[B]`` energy that depends on ``positions``.
        positions: ``[N_atom, 3]`` tensor that must have ``requires_grad=True``
            and be the same tensor the forward pass consumed.
        create_graph: keep the graph so this force can be backpropagated through
            (needed when training against a force label). Costs a second-order
            graph; leave False at inference.

    Returns:
        ``[N_atom, 3]`` conservative force in the global frame.

    Raises:
        ValueError: if ``positions`` does not require grad, which would silently
            yield a zero force.
    """
    if not positions.requires_grad:
        raise ValueError(
            "positions must require grad to differentiate the energy; a "
            "detached input would return a zero conservative force"
        )
    (grad,) = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=allow_unused,
    )
    if grad is None:
        raise ValueError("energy does not depend on positions")
    return -grad
