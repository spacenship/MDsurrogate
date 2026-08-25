"""Deterministic synthetic peptides satisfying the hierarchical contract.

These fixtures exist so that every module below them -- frames, topology,
encoder blocks, heads, losses -- can be developed and equivariance-tested with
no mdCATH file, no ESM-2 checkpoint and no network access. They are *fixtures*,
not data: they are never used to report model quality.

What is faithful to real data:
    ideal backbone bond lengths/angles placed by NeRF, right-handed chirality,
    per-residue heavy-atom counts from the real CHARMM templates (0-10 side-chain
    atoms), ragged residue/atom counts, chain breaks, CHARMM caps and the
    all-atom/heavy-atom split.

What is not:
    side-chain atoms are placed on a deterministic spiral rather than at real
    rotamer positions, and the force labels come from the smooth pair potential
    in :func:`synthetic_forces`, which is exactly conservative. Real mdCATH
    forces are not: they are an open subsystem coupled to solvent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from . import residue_constants as rc
from .contracts import (
    BackboneFrameBatch,
    HierarchicalProteinBatch,
    ProteinAtomBatch,
    ResidueSemanticBatch,
)
from .units import MDCATH_UNITS, UnitMetadata

# Ideal backbone internal coordinates (Engh & Huber).
_BOND_N_CA = 1.458
_BOND_CA_C = 1.525
_BOND_C_N = 1.329
_BOND_C_O = 1.231
_BOND_CA_CB = 1.521
_ANGLE_N_CA_C = math.radians(111.2)
_ANGLE_CA_C_N = math.radians(116.2)
_ANGLE_C_N_CA = math.radians(121.7)
_ANGLE_CA_C_O = math.radians(120.8)
_OMEGA = math.radians(180.0)
#: alpha-helical backbone torsions, which give well-conditioned residue frames.
_PHI_HELIX = math.radians(-57.0)
_PSI_HELIX = math.radians(-47.0)


def _place_atom(
    a: Tensor, b: Tensor, c: Tensor, bond: float, angle: float, torsion: float
) -> Tensor:
    """NeRF: place atom ``d`` from the internal coordinates relative to a,b,c.

    Args:
        a, b, c: ``[3]`` preceding atom positions.
        bond: ``|c-d|``; angle: ``b-c-d``; torsion: dihedral ``a-b-c-d`` (rad).

    Returns:
        ``[3]`` position of ``d``. Right-handed by construction, so a structure
        built from these calls has protein chirality rather than its mirror.
    """
    bc = c - b
    bc = bc / torch.linalg.norm(bc)
    n = torch.linalg.cross(b - a, bc)
    n = n / torch.linalg.norm(n)
    m = torch.stack([bc, torch.linalg.cross(n, bc), n], dim=1)  # columns
    d2 = torch.tensor(
        [
            -bond * math.cos(angle),
            bond * math.sin(angle) * math.cos(torsion),
            bond * math.sin(angle) * math.sin(torsion),
        ],
        dtype=a.dtype,
        device=a.device,
    )
    return c + m @ d2


def _build_backbone(
    num_residues: int, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build an ideal alpha-helical backbone.

    Returns:
        ``(N, CA, C, O)``, each ``[num_residues, 3]``, in global coordinates.
    """
    n_list: list[Tensor] = []
    ca_list: list[Tensor] = []
    c_list: list[Tensor] = []

    kw = {"dtype": dtype, "device": device}
    n0 = torch.tensor([0.0, 0.0, 0.0], **kw)
    ca0 = torch.tensor([_BOND_N_CA, 0.0, 0.0], **kw)
    c0 = _place_atom(
        torch.tensor([0.0, 1.0, 0.0], **kw), n0, ca0, _BOND_CA_C, _ANGLE_N_CA_C, _PHI_HELIX
    )
    n_list.append(n0)
    ca_list.append(ca0)
    c_list.append(c0)

    for i in range(1, num_residues):
        n_i = _place_atom(n_list[i - 1], ca_list[i - 1], c_list[i - 1],
                          _BOND_C_N, _ANGLE_CA_C_N, _PSI_HELIX)
        ca_i = _place_atom(ca_list[i - 1], c_list[i - 1], n_i,
                           _BOND_N_CA, _ANGLE_C_N_CA, _OMEGA)
        c_i = _place_atom(c_list[i - 1], n_i, ca_i,
                          _BOND_CA_C, _ANGLE_N_CA_C, _PHI_HELIX)
        n_list.append(n_i)
        ca_list.append(ca_i)
        c_list.append(c_i)

    n = torch.stack(n_list)
    ca = torch.stack(ca_list)
    c = torch.stack(c_list)
    # Carbonyl O in the peptide plane, anti to the next N.
    o = torch.stack(
        [
            _place_atom(n[i], ca[i], c[i], _BOND_C_O, _ANGLE_CA_C_O, _PSI_HELIX + math.pi)
            for i in range(num_residues)
        ]
    )
    return n, ca, c, o


def _sidechain_positions(
    n: Tensor, ca: Tensor, c: Tensor, names: Sequence[str], dtype: torch.dtype
) -> Tensor:
    """Deterministic side-chain heavy-atom placement for one residue.

    ``CB`` gets the correct tetrahedral position; the remaining atoms walk
    outward on a spiral about the CA->CB axis with ~1.5 A spacing. Geometrically
    valid and collision-free, but not a real rotamer.
    """
    if not names:
        return torch.zeros((0, 3), dtype=dtype, device=ca.device)
    # Standard CB construction from the backbone (right-handed).
    b = ca - n
    cc = c - ca
    a = torch.linalg.cross(b, cc)
    cb_dir = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc
    cb_dir = cb_dir / torch.linalg.norm(cb_dir)
    cb = ca + _BOND_CA_CB * cb_dir

    out = [cb]
    # Orthonormal frame about the CA->CB axis for the spiral.
    u = cb_dir
    tmp = b / torch.linalg.norm(b)
    v = tmp - (tmp @ u) * u
    v = v / torch.linalg.norm(v)
    w = torch.linalg.cross(u, v)
    for k in range(1, len(names)):
        ang = 1.9 * k
        radius = 1.0 + 0.25 * (k % 3)
        pos = (
            cb
            + (1.35 * k) * u
            + radius * (math.cos(ang) * v + math.sin(ang) * w)
        )
        out.append(pos)
    return torch.stack(out)


def synthetic_forces(positions: Tensor, atom_to_residue: Tensor) -> Tensor:
    """Deterministic, learnable force labels from a smooth pair potential.

    ``U`` is a soft inverse-square repulsion plus a weak harmonic attraction to
    the residue centroid, and the label is ``-dU/dx``. Being an exact gradient,
    it is a valid target for the conservative-force branch; real mdCATH forces
    are not exactly conservative in the protein-only subsystem, so tests must
    not treat this property as physical.

    Args:
        positions: ``[N_atom, 3]``.
        atom_to_residue: ``[N_atom]``.

    Returns:
        ``[N_atom, 3]`` forces, same dtype/device as ``positions``.
    """
    x = positions.detach().clone().requires_grad_(True)
    d = torch.cdist(x, x)
    eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    d = d.masked_fill(eye, float("inf"))
    repulsion = (1.0 / (d.clamp(min=0.5) ** 2)).sum() * 0.5

    n_res = int(atom_to_residue.max()) + 1 if atom_to_residue.numel() else 0
    centroid = torch.zeros((n_res, 3), dtype=x.dtype, device=x.device)
    counts = torch.zeros((n_res,), dtype=x.dtype, device=x.device)
    centroid = centroid.index_add(0, atom_to_residue, x)
    counts = counts.index_add(0, atom_to_residue, torch.ones_like(x[:, 0]))
    centroid = centroid / counts.clamp(min=1.0).unsqueeze(-1)
    harmonic = 0.05 * ((x - centroid[atom_to_residue]) ** 2).sum()

    u = repulsion + harmonic
    (grad,) = torch.autograd.grad(u, x)
    return (-grad).detach()


def fake_plm_embedding(
    residue_type: Tensor,
    dim: int = 1280,
    seed: int = 0,
    position_index: Optional[Tensor] = None,
) -> Tensor:
    """Deterministic stand-in for a frozen ESM-2 residue embedding.

    Depends only on residue type and position, so it is reproducible across
    processes and devices and requires no checkpoint download. Offline unit
    tests use this; the real pipeline caches genuine ESM-2 output instead.

    Args:
        position_index: ``[N_res]`` position of each residue **within its own
            protein**. Defaults to a batch-global ``arange``, which would make a
            protein's embedding depend on what else is in the batch -- exactly
            the coupling a real per-protein PLM cache does not have.
    """
    n = int(residue_type.shape[0])
    idx = (
        torch.arange(n, device=residue_type.device, dtype=torch.float32)
        if position_index is None
        else position_index.to(device=residue_type.device, dtype=torch.float32)
    )
    k = torch.arange(dim, device=residue_type.device, dtype=torch.float32)
    phase = (
        residue_type.to(torch.float32).unsqueeze(1) * 0.31
        + idx.unsqueeze(1) * 0.017
        + k.unsqueeze(0) * 0.011
        + float(seed)
    )
    return torch.sin(phase) * 0.1


@dataclass(frozen=True)
class SyntheticSpec:
    """Description of one synthetic protein.

    Args:
        num_residues: residue count.
        num_chains: chains, split as evenly as possible. Chain boundaries are
            real breaks: sequence edges must not cross them.
        nonstandard_at: residue indices given an unknown residue type, to
            exercise the UNK path and residue masking.
        drop_atom_at: residue indices whose last side-chain atom is removed, to
            exercise missing-atom handling.
        drop_frame_atom_at: residue indices whose N/CA/C are left collinear, to
            exercise degenerate-frame masking.
    """

    num_residues: int
    num_chains: int = 1
    nonstandard_at: tuple[int, ...] = ()
    drop_atom_at: tuple[int, ...] = ()
    drop_frame_atom_at: tuple[int, ...] = ()


def _residue_types_for(spec: SyntheticSpec, generator: torch.Generator) -> list[str]:
    pool = [r for r in rc.RESIDUE_TYPES if r != "UNK"]
    idx = torch.randint(
        0, len(pool), (spec.num_residues,), generator=generator, dtype=torch.int64
    )
    names = [pool[int(i)] for i in idx]
    for i in spec.nonstandard_at:
        names[i] = "LIG"  # not in the alphabet -> canonicalises to UNK
    return names


def synthetic_batch(
    specs: Sequence[SyntheticSpec] | Sequence[int],
    *,
    seed: int = 0,
    include_hydrogens: bool = False,
    plm_dim: int = 32,
    with_forces: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    units: UnitMetadata = MDCATH_UNITS,
) -> HierarchicalProteinBatch:
    """Build a deterministic batch of synthetic proteins.

    Args:
        specs: per-protein :class:`SyntheticSpec`, or plain residue counts.
        seed: fixes residue types, jitter and the fake PLM embedding.
        include_hydrogens: add ``HN``/``HA`` to exercise the all-atom path and
            the heavy-atom split. mdCATH does store hydrogens, so both modes are
            real deployment options rather than one being hypothetical.
        plm_dim: width of the fake PLM embedding (real ESM-2 650M is 1280).
        with_forces: attach :func:`synthetic_forces` labels.

    Returns:
        A validated :class:`HierarchicalProteinBatch` on ``device``.
    """
    device = torch.device(device)
    specs = [SyntheticSpec(s) if isinstance(s, int) else s for s in specs]

    pos_all: list[Tensor] = []
    a2r_all: list[Tensor] = []
    atom_batch: list[Tensor] = []
    z_all: list[Tensor] = []
    name_all: list[Tensor] = []
    is_bb_all: list[Tensor] = []
    is_cap_all: list[Tensor] = []

    res_type_all: list[int] = []
    resid_all: list[int] = []
    chain_all: list[int] = []
    res_batch: list[int] = []
    res_mask: list[bool] = []

    n_pos_all: list[Tensor] = []
    ca_pos_all: list[Tensor] = []
    c_pos_all: list[Tensor] = []
    frame_valid: list[bool] = []

    position_in_protein: list[int] = []
    protein_atom_counts: list[int] = []
    residue_offset = 0
    for g, spec in enumerate(specs):
        # One generator per protein, seeded from (seed, index): protein g must be
        # byte-identical whether it is alone in the batch or batched with others,
        # otherwise batch-invariance is untestable.
        generator = torch.Generator().manual_seed(seed * 1_000_003 + g)
        names = _residue_types_for(spec, generator)
        n_bb, ca_bb, c_bb, o_bb = _build_backbone(spec.num_residues, dtype, device)
        # Separate chains so they do not overlap in space.
        per_chain = max(1, math.ceil(spec.num_residues / spec.num_chains))
        shift = torch.tensor([0.0, 0.0, 25.0], dtype=dtype, device=device)
        chain_of = [min(i // per_chain, spec.num_chains - 1) for i in range(spec.num_residues)]
        for i in range(spec.num_residues):
            off = shift * float(chain_of[i])
            n_bb[i] = n_bb[i] + off
            ca_bb[i] = ca_bb[i] + off
            c_bb[i] = c_bb[i] + off
            o_bb[i] = o_bb[i] + off

        for i in range(spec.num_residues):
            raw = names[i]
            canon = rc.canonical_resname(raw)
            res_type_all.append(rc.residue_type_id(raw))
            resid_all.append(i + 8)  # mdCATH keeps original PDB numbering
            chain_all.append(chain_of[i])
            res_batch.append(g)
            position_in_protein.append(i)
            res_mask.append(canon != "UNK")

            n_i, ca_i, c_i = n_bb[i], ca_bb[i], c_bb[i]
            if i in spec.drop_frame_atom_at:
                # Force a collinear (degenerate) frame: N, CA, C on one line.
                n_i = ca_i - (c_i - ca_i)
            n_pos_all.append(n_i)
            ca_pos_all.append(ca_i)
            c_pos_all.append(c_i)
            frame_valid.append(i not in spec.drop_frame_atom_at)

            sc_names = list(rc.SIDECHAIN_HEAVY_ATOMS[canon])
            if i in spec.drop_atom_at and sc_names:
                sc_names = sc_names[:-1]
            sc = _sidechain_positions(n_bb[i], ca_bb[i], c_bb[i], sc_names, dtype)

            atom_names = ["N", "CA", "C", "O"] + sc_names
            atom_pos = [n_i, ca_i, c_i, o_bb[i]] + list(sc)
            if include_hydrogens:
                hn = n_i + (n_i - ca_i) / torch.linalg.norm(n_i - ca_i) * 1.01
                ha = ca_i + (ca_i - c_i) / torch.linalg.norm(ca_i - c_i) * 1.09
                atom_names += ["HN", "HA"]
                atom_pos += [hn, ha]
            # CHARMM caps live on the terminal residues of each chain.
            first_of_chain = i == 0 or chain_of[i] != chain_of[i - 1]
            last_of_chain = i == spec.num_residues - 1 or chain_of[i] != chain_of[i + 1]
            if first_of_chain:
                atom_names += ["CAY", "CY", "OY"]
                atom_pos += [n_i + torch.tensor([-1.5, 0.3, 0.0], dtype=dtype, device=device),
                             n_i + torch.tensor([-2.4, 1.0, 0.0], dtype=dtype, device=device),
                             n_i + torch.tensor([-3.5, 0.7, 0.0], dtype=dtype, device=device)]
            if last_of_chain:
                atom_names += ["NT", "CAT"]
                atom_pos += [c_i + torch.tensor([1.3, 0.4, 0.0], dtype=dtype, device=device),
                             c_i + torch.tensor([2.4, 1.1, 0.0], dtype=dtype, device=device)]

            k = len(atom_names)
            pos_all.append(torch.stack(atom_pos))
            a2r_all.append(torch.full((k,), residue_offset + i, dtype=torch.int64, device=device))
            atom_batch.append(torch.full((k,), g, dtype=torch.int64, device=device))
            z_all.append(torch.tensor(
                [rc.ELEMENT_TO_Z[_element_of(nm)] for nm in atom_names],
                dtype=torch.int64, device=device))
            name_all.append(torch.tensor(
                [rc.atom_name_id(nm) for nm in atom_names], dtype=torch.int64, device=device))
            is_bb_all.append(torch.tensor(
                [rc.is_backbone_atom(nm) for nm in atom_names], dtype=torch.bool, device=device))
            is_cap_all.append(torch.tensor(
                [rc.is_cap_atom(nm) for nm in atom_names], dtype=torch.bool, device=device))

        residue_offset += spec.num_residues
        protein_atom_counts.append(sum(int(p.shape[0]) for p in pos_all) - sum(protein_atom_counts))

    positions = torch.cat(pos_all).to(device=device, dtype=dtype)
    # Small deterministic jitter so no two atoms coincide exactly. Drawn per
    # protein from its own generator, so a protein's coordinates do not depend
    # on what else shares the batch.
    jitter_parts = []
    for g, count in enumerate(protein_atom_counts):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + g + 7)
        jitter_parts.append(torch.empty((count, 3), dtype=torch.float32).normal_(generator=gen))
    positions = positions + torch.cat(jitter_parts).to(device=device, dtype=dtype) * 0.02

    atom_to_residue = torch.cat(a2r_all)
    n_res = residue_offset
    residue_type = torch.tensor(res_type_all, dtype=torch.int64, device=device)

    atoms = ProteinAtomBatch(
        positions=positions,
        atom_to_residue=atom_to_residue,
        batch_index=torch.cat(atom_batch),
        atomic_number=torch.cat(z_all),
        atom_name_id=torch.cat(name_all),
        is_backbone=torch.cat(is_bb_all),
        is_cap=torch.cat(is_cap_all),
    )
    if with_forces:
        atoms.forces = synthetic_forces(positions, atom_to_residue)
        atoms.force_valid = torch.ones(atoms.num_atoms, dtype=torch.bool, device=device)

    residues = ResidueSemanticBatch(
        residue_type=residue_type,
        plm_embedding=fake_plm_embedding(
            residue_type, plm_dim, seed,
            position_index=torch.tensor(position_in_protein, device=device),
        ).to(dtype),
        resid_original=torch.tensor(resid_all, dtype=torch.int64, device=device),
        chain_index=torch.tensor(chain_all, dtype=torch.int64, device=device),
        batch_index=torch.tensor(res_batch, dtype=torch.int64, device=device),
        mask=torch.tensor(res_mask, dtype=torch.bool, device=device),
    )
    backbone = BackboneFrameBatch(
        n_positions=torch.stack(n_pos_all).to(device=device, dtype=dtype),
        ca_positions=torch.stack(ca_pos_all).to(device=device, dtype=dtype),
        c_positions=torch.stack(c_pos_all).to(device=device, dtype=dtype),
        residue_to_backbone=torch.arange(n_res, dtype=torch.int64, device=device),
        frame_valid=torch.tensor(frame_valid, dtype=torch.bool, device=device),
        batch_index=residues.batch_index.clone(),
    )

    b = len(specs)
    temperatures = torch.tensor(
        [float([320, 348, 379, 413, 450][g % 5]) for g in range(b)],
        dtype=dtype, device=device,
    )
    batch = HierarchicalProteinBatch(
        atoms=atoms,
        residues=residues,
        backbone=backbone,
        units=units,
        temperature=temperatures,
        domain_id=tuple(f"synth{g:03d}A00" for g in range(b)),
        replica_index=torch.zeros(b, dtype=torch.int64, device=device),
        frame_index=torch.arange(b, dtype=torch.int64, device=device),
    )
    batch.validate()
    return batch


def _element_of(atom_name: str) -> str:
    """Element of a synthetic atom name. Real data reads ``element``/``z``."""
    name = atom_name.upper()
    if name.startswith("H"):
        return "H"
    if name.startswith("N"):
        return "N"
    if name.startswith("O"):
        return "O"
    if name.startswith("S"):
        return "S"
    return "C"
