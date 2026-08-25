"""Expressing global-frame irreps in each residue's own frame.

Phase 1's ``physics_latent`` lives in the **global frame** and carries irreps
``64x0e + 16x1o + 8x2e``. A conditioner that flattened those 152 numbers into an
MLP would be reading a quantity that changes when the protein is rotated, and the
transition probe would spend capacity learning to undo a rotation that carries no
information. That is the failure the Phase 1.5 plan calls out by name.

The fix is a change of frame, not a change of representation::

    z_local_i = D(R_i^T) z_i

where ``D`` is the Wigner-D of the feature's irreps and ``R_i`` is residue ``i``'s
frame. Under a global rotation ``Q`` the frame becomes ``Q R_i`` and the feature
becomes ``D(Q) z``, so

    D((Q R_i)^T) D(Q) z = D(R_i^T) D(Q^T) D(Q) z = D(R_i^T) z

is unchanged: every component of ``z_local`` is an SE(3) **invariant** and may be
fed to an ordinary MLP. Nothing is discarded -- the l=1 and l=2 channels keep all
their information, now expressed relative to the residue's own axes.

**Why this is not one call to ``irreps.D_from_matrix``.** Two measured reasons:

1. *It cannot run on the training device.* e3nn caches its Wigner generators on
   CPU and a GPU rotation raises ``Expected all tensors to be on the same
   device``. The same note already appears in ``graph.edges``. Building the full
   ``[N_res, 152, 152]`` block-diagonal on CPU and transferring it costs ~47 MB
   per batch; the per-degree blocks are 3x3 and 5x5 and cost ~70 kB.
2. *The l=1 block is exactly the rotation matrix.* e3nn reaches it through
   ``matrix -> Euler angles -> D``, which loses precision: measured
   ``|D_1(R) - R| = 4e-7`` in **float64**. Using ``R`` directly is exact and free.

So l=0 passes through, l=1 uses ``R`` itself, and only l>=2 goes through e3nn --
on CPU, as small per-degree matrices. ``test_l1_wigner_d_is_the_rotation_matrix``
pins assumption 2, so a future e3nn that changes the l=1 basis ordering fails
loudly instead of silently transposing every vector channel.
"""

from __future__ import annotations

from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn

__all__ = ["IrrepsLocalFrame"]


class IrrepsLocalFrame(nn.Module):
    """Rotate irreps features into their residue's local frame.

    Args:
        irreps: the feature's irreps, e.g. ``"64x0e+16x1o+8x2e"``.

    Shape:
        ``([N, irreps.dim], [N, 3, 3]) -> [N, irreps.dim]``, invariant under a
        global rigid motion of the structure that produced both arguments.

    Has no parameters: this is a change of basis, not a learned map.
    """

    def __init__(self, irreps: o3.Irreps | str):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        blocks = []
        offset = 0
        for multiplicity, irrep in self.irreps:
            width = multiplicity * irrep.dim
            blocks.append((offset, multiplicity, irrep, width))
            offset += width
        self._blocks = blocks
        self.dim = int(self.irreps.dim)

    @property
    def scalar_dim(self) -> int:
        return sum(m * ir.dim for m, ir in self.irreps if ir.l == 0)

    def forward(self, features: Tensor, rotation: Tensor) -> Tensor:
        """Args:
            features: ``[N, dim]`` in the global frame.
            rotation: ``[N, 3, 3]`` residue frames, local axes as columns.
        """
        if features.shape[-1] != self.dim:
            raise ValueError(
                f"features have width {features.shape[-1]}, irreps {self.irreps} "
                f"need {self.dim}"
            )
        if rotation.shape[0] != features.shape[0]:
            raise ValueError(
                f"{rotation.shape[0]} rotations for {features.shape[0]} feature rows"
            )

        # R^T maps global -> local. For l=1 that *is* the Wigner-D.
        inverse = rotation.transpose(-1, -2)
        out = []
        for offset, multiplicity, irrep, width in self._blocks:
            block = features[:, offset : offset + width]
            if irrep.l == 0 or multiplicity == 0:
                out.append(block)
                continue
            wigner = (
                inverse
                if irrep.l == 1
                else self._wigner(irrep, inverse, features.dtype, features.device)
            )
            reshaped = block.reshape(block.shape[0], multiplicity, irrep.dim)
            rotated = torch.einsum("nij,ncj->nci", wigner.to(block.dtype), reshaped)
            out.append(rotated.reshape(block.shape[0], width))
        return torch.cat(out, dim=-1)

    @staticmethod
    def _wigner(
        irrep: o3.Irrep, rotation: Tensor, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """``D_l(rotation)`` for ``l >= 2``, built on CPU because e3nn's generators are.

        Only a ``[N, 2l+1, 2l+1]`` matrix crosses the bus -- 51 kB for 512 residues
        at ``l = 2``, against 47 MB for the full block-diagonal.
        """
        matrix = irrep.D_from_matrix(rotation.detach().to("cpu", torch.float64))
        return matrix.to(device=device, dtype=dtype)
