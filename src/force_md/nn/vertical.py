"""Vertical operations between the three levels.

All four modules work on **global-frame** irreps, so equivariance needs no frame
bookkeeping: a scatter of equivariant features is equivariant, and a gather is
too. Frames appear only when building invariant input scalars and when
expressing output uncertainty.

Scalars are combined with non-scalar irreps by *concatenating irreps sets* and
letting :class:`e3nn.o3.Linear` mix them -- ``o3.Linear`` only connects paths of
equal degree, so a ``0e`` input can never leak into an ``l>0`` output channel.
That is what makes ``irreps_a + irreps_b + "Nx0e"`` a valid combination rather
than the invalid "concatenate a scalar onto a vector" the design forbids.
"""

from __future__ import annotations

from e3nn import o3
from torch import Tensor, nn

from .irreps import GatedLinear, scatter_mean, scatter_sum

__all__ = [
    "AtomToResiduePool",
    "ResidueToBackboneInjection",
    "BackboneToResidue",
    "ResidueToAtomBroadcast",
]


class AtomToResiduePool(nn.Module):
    """Equivariant pooling of atom irreps onto their parent residue.

    Args:
        reduction: ``"mean"`` (default) or ``"sum"``.

    ``mean`` is the default because this is a *representation* pool: glycine has
    4 heavy atoms and tryptophan 14, and with ``sum`` the pooled feature of a
    large residue would be systematically ~3x larger for reasons that have
    nothing to do with its physics. The physically extensive aggregation -- the
    one that must be a plain sum because forces add -- is a separate, explicit
    projector (:mod:`force_md.physics.projection`), deliberately not this module.

    Shape:
        ``[N_atom, D] -> [N_res, D]``.
    """

    def __init__(self, irreps_node: o3.Irreps, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.reduction = reduction
        self.irreps_node = o3.Irreps(irreps_node)
        self.linear = o3.Linear(self.irreps_node, self.irreps_node)

    def forward(
        self, atom_features: Tensor, atom_to_residue: Tensor, num_residues: int
    ) -> Tensor:
        projected = self.linear(atom_features)
        reduce = scatter_mean if self.reduction == "mean" else scatter_sum
        return reduce(projected, atom_to_residue, num_residues)


class ResidueToBackboneInjection(nn.Module):
    """Fuse pooled atom irreps and invariant residue scalars into backbone nodes.

    The residue semantic node contributes only ``l=0`` information (PLM,
    identity, temperature); the pooled atom branch contributes the full irreps
    set. Both are concatenated as irreps and mixed by one ``o3.Linear``, then
    gated -- so residue scalars modulate the equivariant channels through the
    gate rather than being pasted onto them.

    Shape:
        ``([N_res, D], [N_res, D], [N_res, S]) -> [N_res, D]`` (residual).
    """

    def __init__(self, irreps_node: o3.Irreps, residue_scalar_dim: int):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        irreps_in = (
            self.irreps_node + self.irreps_node + o3.Irreps(f"{residue_scalar_dim}x0e")
        )
        self.mix = GatedLinear(irreps_in, self.irreps_node)

    def forward(
        self,
        backbone_features: Tensor,
        pooled_atom_features: Tensor,
        residue_scalars: Tensor,
    ) -> Tensor:
        import torch

        combined = torch.cat(
            [backbone_features, pooled_atom_features, residue_scalars], dim=-1
        )
        return backbone_features + self.mix(combined)


class BackboneToResidue(nn.Module):
    """Send backbone-level global context back to the residue nodes.

    The mapping is 1:1, so this is a gather by ``residue_to_backbone`` followed
    by a gated mix with the residue's own features -- not a plain copy, so the
    residue level can weigh how much global context it admits.

    Shape:
        ``([N_res, D], [N_res, D]) -> [N_res, D]`` (residual).
    """

    def __init__(self, irreps_node: o3.Irreps):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.mix = GatedLinear(self.irreps_node + self.irreps_node, self.irreps_node)

    def forward(
        self,
        residue_features: Tensor,
        backbone_features: Tensor,
        residue_to_backbone: Tensor,
    ) -> Tensor:
        import torch

        gathered = backbone_features[residue_to_backbone]
        return residue_features + self.mix(
            torch.cat([residue_features, gathered], dim=-1)
        )


class ResidueToAtomBroadcast(nn.Module):
    """Broadcast residue context down to child atoms through a learned gate.

    Explicitly *not* a raw copy of the residue vector onto each atom: the parent
    features are mixed with the atom's own features and gated, so each atom
    receives a modulation rather than a duplicated 1280-dim PLM row.

    Shape:
        ``([N_atom, D], [N_res, D], [N_atom]) -> [N_atom, D]`` (residual).
    """

    def __init__(self, irreps_node: o3.Irreps):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.mix = GatedLinear(self.irreps_node + self.irreps_node, self.irreps_node)

    def forward(
        self,
        atom_features: Tensor,
        residue_features: Tensor,
        atom_to_residue: Tensor,
    ) -> Tensor:
        import torch

        gathered = residue_features[atom_to_residue]
        return atom_features + self.mix(torch.cat([atom_features, gathered], dim=-1))
