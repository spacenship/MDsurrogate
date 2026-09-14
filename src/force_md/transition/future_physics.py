"""``P4``: predicting Phase 1's physics latent at ``t + lag``, as an auxiliary loss.

The idea being tested is that a transition representation is better if it knows
*what the protein's local physics will be like* when it arrives, not only where
the residues end up. So the probe grows one extra head that reads its own hidden
state and predicts the future physics latent, supervised against frozen Phase 1
applied to the ground-truth future structure.

**This is a loss target and only a loss target.** Three separate mechanisms keep
it there:

1. The target is computed in the trainer, never inside the probe's forward, and
   is ``detach()``-ed at the point of creation (:func:`future_physics_target`).
2. It is delivered through :class:`~force_md.transition.targets.TransitionPrediction`
   -- the *output* dataclass -- so there is no input path it could take.
3. The future state is built with ``forces=None`` (:func:`future_state_batch`), so
   even a future *force* label does not exist to be read, let alone a future
   force feature.

**Why a latent consistency loss and not a future-force MSE.** Phase 1.5 measured
that instantaneous force at these lags is worth ~0.2 degrees of frame rotation,
and the oracle showed that is a ceiling. Regressing the exact future force would
spend the auxiliary budget on the quantity already known to carry almost nothing.
The latent is the representation, is bounded, and a Huber on it with a cosine
diagnostic says "will the neighbourhood look like this" rather than "reproduce
this vector".

**Invariance.** Both sides are expressed in the **current** residue frame before
they are compared. The target is ``D(R_cur^T) z_future`` and the prediction comes
from ``D(R_cur^T) h_node``, so the loss is invariant under a global rigid motion
of the whole trajectory -- which it must be, or the auxiliary term would teach the
probe to track the laboratory frame.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import Tensor, nn

from ..data.contracts import FrameGeometry, HierarchicalProteinBatch
from .local_frame import IrrepsLocalFrame

__all__ = [
    "future_state_batch",
    "FuturePhysicsHead",
    "future_physics_target",
    "future_physics_loss",
]


def future_state_batch(
    current: HierarchicalProteinBatch, future: FrameGeometry
) -> HierarchicalProteinBatch:
    """The same protein at ``t + lag``: current topology, future coordinates.

    ``FrameGeometry`` already carries every represented atom's position for that
    frame, row-for-row with ``current.atoms`` (``FrameGeometry.matches`` checks
    exactly this), so no extra file read is needed -- the frame was loaded to
    build the transition target.

    Chemistry, sequence, PLM embedding, temperature and the residue mask are
    properties of the molecule and are carried over unchanged. **Forces are
    dropped**, not carried and not recomputed: there is no legitimate consumer of
    a future force in Phase 1.6, so the field is made absent rather than left
    lying around as a temptation with a guard next to it.

    Returns:
        A batch suitable for the frozen Phase 1 extractor. It is *not* suitable
        for a conditioner and must never be handed to one.
    """
    future.matches(current)
    atoms = replace(
        current.atoms, positions=future.positions, forces=None, force_valid=None
    )
    backbone = replace(
        current.backbone,
        n_positions=future.n_positions,
        ca_positions=future.ca_positions,
        c_positions=future.c_positions,
        frame_valid=future.frame_valid,
    )
    return replace(
        current, atoms=atoms, backbone=backbone, frame_index=future.frame_index
    )


class FuturePhysicsHead(nn.Module):
    """Probe hidden state -> predicted future physics latent, in the local frame.

    Args:
        node_irreps: the probe's node irreps (global frame).
        latent_irreps: Phase 1's ``physics_latent_irreps``, so the head's width is
            read from the checkpoint contract rather than assumed to be 152.

    Shape:
        ``([N_res, node_irreps.dim], [N_res, 3, 3]) -> [N_res, latent_dim]``,
        invariant under a global rigid motion.

    Zero-initialised on the output layer, for the same reason the transition heads
    are: the auxiliary term starts at a definite, interpretable place instead of
    injecting a random gradient into the primary objective on step 1.
    """

    def __init__(self, node_irreps: str, latent_irreps: str):
        super().__init__()
        self.projection = IrrepsLocalFrame(node_irreps)
        self.target_projection = IrrepsLocalFrame(latent_irreps)
        self.latent_dim = self.target_projection.dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.projection.dim),
            nn.Linear(self.projection.dim, self.projection.dim),
            nn.SiLU(),
            nn.Linear(self.projection.dim, self.latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, node: Tensor, rotation: Tensor) -> Tensor:
        return self.net(self.projection(node, rotation))


def future_physics_target(
    future_latent: Tensor, rotation: Tensor, head: FuturePhysicsHead
) -> Tensor:
    """``stopgrad(D(R_cur^T) z_{t+lag})`` -- the auxiliary target.

    Args:
        future_latent: ``[N_res, D]`` frozen Phase 1's latent for the **future**
            structure, global frame.
        rotation: ``[N_res, 3, 3]`` the **current** residue frames. Using the
            current frame, not the future one, is what makes the target something
            the model can be asked for: it is "how will the physics look, seen
            from where I am now".

    Detached here rather than at the call site, so a caller cannot forget.
    """
    return head.target_projection(future_latent, rotation).detach()


def future_physics_loss(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    delta: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Huber consistency plus a cosine diagnostic, over valid residues only.

    Returns:
        ``(loss, diagnostics)``. The cosine is reported, never optimised: it says
        whether the *direction* of the predicted latent is right independently of
        its scale, which is what distinguishes "learned the representation" from
        "learned the mean norm".
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"future physics prediction {tuple(predicted.shape)} does not match "
            f"target {tuple(target.shape)}"
        )
    mask = valid.to(predicted.dtype).unsqueeze(-1)
    error = predicted - target
    absolute = error.abs()
    huber = torch.where(
        absolute <= delta, 0.5 * absolute.pow(2), delta * (absolute - 0.5 * delta)
    )
    denominator = mask.sum().clamp(min=1.0) * predicted.shape[-1]
    loss = (huber * mask).sum() / denominator

    with torch.no_grad():
        cosine = torch.nn.functional.cosine_similarity(
            predicted, target, dim=-1, eps=1e-8
        )
        rows = valid.to(cosine.dtype)
        diagnostics = {
            "future_physics_huber": float(loss),
            "future_physics_cosine": float(
                (cosine * rows).sum() / rows.sum().clamp(min=1.0)
            ),
            "future_physics_target_norm": float(
                (target.norm(dim=-1) * rows).sum() / rows.sum().clamp(min=1.0)
            ),
        }
    return loss, diagnostics
