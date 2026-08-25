"""Collating single-protein states into one ragged batch.

Concatenation, not padding: node arrays are joined along their single axis and
the index tensors are shifted by the running offsets. Getting an offset wrong
produces edges and parent links that point into a neighbouring protein, which is
invisible in the loss and corrupts every prediction in the batch -- so the
offsets are applied in one place and checked by ``validate()``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from .contracts import (
    BackboneFrameBatch,
    FrameGeometry,
    HierarchicalProteinBatch,
    ProteinAtomBatch,
    ResidueSemanticBatch,
)

__all__ = ["collate_batches", "collate_frame_geometries"]


def _cat(tensors: Sequence[Optional[Tensor]]) -> Optional[Tensor]:
    if any(t is None for t in tensors):
        return None
    return torch.cat(list(tensors), dim=0)


def collate_batches(
    batches: Sequence[HierarchicalProteinBatch],
    *,
    validate: bool = True,
) -> HierarchicalProteinBatch:
    """Join per-protein states into a single batch.

    Args:
        batches: one or more states, each typically holding a single protein.
        validate: run the contract check on the result.

    Returns:
        One :class:`HierarchicalProteinBatch` whose ``batch_index`` tensors
        number the inputs 0..len(batches)-1.

    Raises:
        ValueError: on an empty list or mismatched units / PLM widths.
    """
    if not batches:
        raise ValueError("cannot collate an empty list of batches")
    units = batches[0].units
    if any(b.units != units for b in batches):
        raise ValueError("cannot collate batches with different units")
    plm_dim = batches[0].residues.plm_dim
    if any(b.residues.plm_dim != plm_dim for b in batches):
        raise ValueError(
            "cannot collate batches with different PLM widths "
            f"({sorted({b.residues.plm_dim for b in batches})})"
        )

    atom_offset = residue_offset = graph_offset = 0
    a_pos, a_res, a_batch, a_z, a_name, a_bb, a_cap, a_f, a_fv = ([] for _ in range(9))
    r_type, r_plm, r_resid, r_chain, r_batch, r_mask = ([] for _ in range(6))
    b_n, b_ca, b_c, b_r2b, b_valid, b_batch = ([] for _ in range(6))
    temperature, replica, frame = [], [], []
    domain_ids: list[str] = []

    for b in batches:
        n_res = b.num_residues
        a_pos.append(b.atoms.positions)
        a_res.append(b.atoms.atom_to_residue + residue_offset)
        a_batch.append(b.atoms.batch_index + graph_offset)
        a_z.append(b.atoms.atomic_number)
        a_name.append(b.atoms.atom_name_id)
        a_bb.append(b.atoms.is_backbone)
        a_cap.append(b.atoms.is_cap)
        a_f.append(b.atoms.forces)
        a_fv.append(b.atoms.force_valid)

        r_type.append(b.residues.residue_type)
        r_plm.append(b.residues.plm_embedding)
        r_resid.append(b.residues.resid_original)
        r_chain.append(b.residues.chain_index)
        r_batch.append(b.residues.batch_index + graph_offset)
        r_mask.append(b.residues.mask)

        b_n.append(b.backbone.n_positions)
        b_ca.append(b.backbone.ca_positions)
        b_c.append(b.backbone.c_positions)
        b_r2b.append(b.backbone.residue_to_backbone + residue_offset)
        b_valid.append(b.backbone.frame_valid)
        b_batch.append(b.backbone.batch_index + graph_offset)

        temperature.append(b.temperature)
        replica.append(b.replica_index)
        frame.append(b.frame_index)
        domain_ids.extend(b.domain_id)

        atom_offset += b.num_atoms
        residue_offset += n_res
        graph_offset += b.num_graphs

    merged = HierarchicalProteinBatch(
        atoms=ProteinAtomBatch(
            positions=torch.cat(a_pos), atom_to_residue=torch.cat(a_res),
            batch_index=torch.cat(a_batch), atomic_number=torch.cat(a_z),
            atom_name_id=torch.cat(a_name), is_backbone=torch.cat(a_bb),
            is_cap=torch.cat(a_cap), forces=_cat(a_f), force_valid=_cat(a_fv),
        ),
        residues=ResidueSemanticBatch(
            residue_type=torch.cat(r_type), plm_embedding=torch.cat(r_plm),
            resid_original=torch.cat(r_resid), chain_index=torch.cat(r_chain),
            batch_index=torch.cat(r_batch), mask=torch.cat(r_mask),
        ),
        backbone=BackboneFrameBatch(
            n_positions=torch.cat(b_n), ca_positions=torch.cat(b_ca),
            c_positions=torch.cat(b_c), residue_to_backbone=torch.cat(b_r2b),
            frame_valid=torch.cat(b_valid), batch_index=torch.cat(b_batch),
        ),
        units=units,
        temperature=torch.cat(temperature),
        domain_id=tuple(domain_ids),
        replica_index=torch.cat(replica),
        frame_index=torch.cat(frame),
    )
    if validate:
        merged.validate()
    return merged


def collate_frame_geometries(
    frames: Sequence[FrameGeometry], *, validate: bool = False
) -> FrameGeometry:
    """Join per-protein :class:`FrameGeometry` into one ragged batch.

    The offsets applied here are only graph indices: unlike the full state there
    are no cross-level index tensors to shift, because a frame carries geometry
    and nothing that points at another node.
    """
    if not frames:
        raise ValueError("cannot collate an empty list of frames")
    graph_offset = 0
    pos, npos, capos, cpos, valid, a_batch, r_batch, idx = ([] for _ in range(8))
    for f in frames:
        pos.append(f.positions)
        npos.append(f.n_positions)
        capos.append(f.ca_positions)
        cpos.append(f.c_positions)
        valid.append(f.frame_valid)
        a_batch.append(f.atom_batch_index + graph_offset)
        r_batch.append(f.residue_batch_index + graph_offset)
        idx.append(f.frame_index)
        graph_offset += f.num_graphs
    merged = FrameGeometry(
        positions=torch.cat(pos),
        n_positions=torch.cat(npos),
        ca_positions=torch.cat(capos),
        c_positions=torch.cat(cpos),
        frame_valid=torch.cat(valid),
        atom_batch_index=torch.cat(a_batch),
        residue_batch_index=torch.cat(r_batch),
        frame_index=torch.cat(idx),
    )
    if validate:
        merged.validate(num_graphs=graph_offset)
    return merged
