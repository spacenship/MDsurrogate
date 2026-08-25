"""The synthetic fixture must be deterministic, ragged and geometrically sane."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import ca_pseudo_torsions, dihedral
from force_md.data import SyntheticSpec, synthetic_batch, synthetic_forces
from force_md.data import residue_constants as rc


def test_determinism_same_seed():
    a = synthetic_batch([SyntheticSpec(5)], seed=7)
    b = synthetic_batch([SyntheticSpec(5)], seed=7)
    assert torch.equal(a.atoms.positions, b.atoms.positions)
    assert torch.equal(a.residues.residue_type, b.residues.residue_type)
    assert torch.equal(a.atoms.forces, b.atoms.forces)


def test_different_seed_changes_sequence():
    a = synthetic_batch([SyntheticSpec(12)], seed=0)
    b = synthetic_batch([SyntheticSpec(12)], seed=1)
    assert not torch.equal(a.residues.residue_type, b.residues.residue_type)


def test_variable_sizes_and_ragged_boundaries():
    """Different residue counts per graph must land in the right segments."""
    batch = synthetic_batch([SyntheticSpec(3), SyntheticSpec(9), SyntheticSpec(5)], seed=0)
    assert batch.num_graphs == 3
    assert batch.num_residues == 17
    counts = torch.bincount(batch.residues.batch_index, minlength=3)
    assert counts.tolist() == [3, 9, 5]
    # atoms partition across graphs with no gaps and no overlap
    atom_counts = torch.bincount(batch.atoms.batch_index, minlength=3)
    assert int(atom_counts.sum()) == batch.num_atoms
    assert all(c > 0 for c in atom_counts.tolist())
    # residue ids are global and contiguous per graph
    for g in range(3):
        r = batch.atoms.atom_to_residue[batch.atoms.batch_index == g]
        assert int(r.max()) - int(r.min()) == counts[g] - 1


def test_atom_counts_follow_charmm_templates():
    batch = synthetic_batch([SyntheticSpec(30)], seed=3)
    types = batch.residues.residue_type
    for i in range(batch.num_residues):
        name = rc.RESIDUE_TYPES[int(types[i])]
        sel = batch.atoms.atom_to_residue == i
        names = {rc.ATOM_NAMES[int(j)] for j in batch.atoms.atom_name_id[sel]}
        expected_sc = set(rc.SIDECHAIN_HEAVY_ATOMS[name])
        assert expected_sc <= names, f"{name} missing side-chain atoms"
        assert {"N", "CA", "C", "O"} <= names
        if name == "GLY":
            assert "CB" not in names


def test_nonstandard_residue_maps_to_unk_and_is_masked():
    batch = synthetic_batch([SyntheticSpec(6, nonstandard_at=(2,))], seed=0)
    assert int(batch.residues.residue_type[2]) == rc.UNK_RESIDUE_ID
    assert not bool(batch.residues.mask[2])
    assert bool(batch.residues.mask[0])


def test_missing_atom_shrinks_that_residue_only():
    full = synthetic_batch([SyntheticSpec(6)], seed=11)
    holed = synthetic_batch([SyntheticSpec(6, drop_atom_at=(3,))], seed=11)
    n_full = torch.bincount(full.atoms.atom_to_residue, minlength=6)
    n_hole = torch.bincount(holed.atoms.atom_to_residue, minlength=6)
    diff = (n_full - n_hole).tolist()
    # residue 3 loses at most one atom (GLY has no side chain to drop)
    assert diff[3] in (0, 1)
    assert all(d == 0 for i, d in enumerate(diff) if i != 3)


def test_degenerate_frame_is_flagged():
    batch = synthetic_batch([SyntheticSpec(6, drop_frame_atom_at=(4,))], seed=0)
    assert not bool(batch.backbone.frame_valid[4])
    assert bool(batch.backbone.frame_valid[0])
    n, ca, c = (batch.backbone.n_positions[4], batch.backbone.ca_positions[4],
                batch.backbone.c_positions[4])
    cross = torch.linalg.cross(n - ca, c - ca)
    assert float(torch.linalg.norm(cross)) < 1e-4  # collinear


def test_chain_break_separates_chains():
    batch = synthetic_batch([SyntheticSpec(10, num_chains=2)], seed=0)
    chains = batch.residues.chain_index
    assert set(chains.tolist()) == {0, 1}
    ca = batch.backbone.ca_positions
    first_of_second = int((chains == 1).nonzero()[0])
    gap = torch.linalg.norm(ca[first_of_second] - ca[first_of_second - 1])
    assert float(gap) > 10.0, "chains must be spatially separated, not a fake bond"


def test_hydrogens_and_heavy_atom_split():
    heavy = synthetic_batch([SyntheticSpec(5)], seed=0)
    assert bool(heavy.atoms.is_heavy.all()), "heavy-atom mode must contain no H"

    allatom = synthetic_batch([SyntheticSpec(5)], seed=0, include_hydrogens=True)
    assert int((~allatom.atoms.is_heavy).sum()) > 0
    assert int(allatom.atoms.is_heavy.sum()) == heavy.num_atoms
    assert (allatom.atoms.atomic_number[~allatom.atoms.is_heavy] == 1).all()


def test_cap_atoms_are_flagged_and_not_backbone():
    """CAY is a cap carbon, not the alpha carbon. Confusing them breaks frames."""
    batch = synthetic_batch([SyntheticSpec(4)], seed=0)
    caps = batch.atoms.is_cap
    assert int(caps.sum()) > 0
    assert not bool((caps & batch.atoms.is_backbone).any())
    cap_names = {rc.ATOM_NAMES[int(i)] for i in batch.atoms.atom_name_id[caps]}
    assert cap_names <= rc.CAP_ATOM_NAMES
    assert "CAY" in cap_names


def test_backbone_bond_geometry_is_ideal():
    batch = synthetic_batch([SyntheticSpec(12)], seed=0)
    bb = batch.backbone
    d_n_ca = torch.linalg.norm(bb.ca_positions - bb.n_positions, dim=-1)
    d_ca_c = torch.linalg.norm(bb.c_positions - bb.ca_positions, dim=-1)
    assert torch.allclose(d_n_ca, torch.full_like(d_n_ca, 1.458), atol=1e-3)
    assert torch.allclose(d_ca_c, torch.full_like(d_ca_c, 1.525), atol=1e-3)
    # consecutive CA-CA is the known virtual bond length
    d_ca_ca = torch.linalg.norm(bb.ca_positions[1:] - bb.ca_positions[:-1], dim=-1)
    assert torch.allclose(d_ca_ca, torch.full_like(d_ca_ca, 3.8), atol=0.1)


def test_helix_is_right_handed():
    """Chirality is a modelling requirement, not a symmetry to be averaged out.

    A right-handed alpha helix has a positive CA pseudo-torsion near +50 deg.
    Mirroring the structure must flip the sign, so this also proves the test
    would actually catch a left-handed build.
    """
    batch = synthetic_batch([SyntheticSpec(10)], seed=0, dtype=torch.float64)
    ca = batch.backbone.ca_positions
    torsions = ca_pseudo_torsions(ca)
    mean = sum(torsions) / len(torsions)
    assert mean > 0, "synthetic helix came out left-handed"
    assert 30.0 < mean < 70.0

    mirrored = ca * torch.tensor([-1.0, 1.0, 1.0], dtype=ca.dtype)
    m_torsions = ca_pseudo_torsions(mirrored)
    assert sum(m_torsions) / len(m_torsions) < 0


def test_backbone_torsions_are_alpha_helical():
    """phi/psi/omega must come out as requested, not 180 deg off."""
    batch = synthetic_batch([SyntheticSpec(6)], seed=0, dtype=torch.float64)
    n, ca, c = (batch.backbone.n_positions, batch.backbone.ca_positions,
                batch.backbone.c_positions)
    for i in (1, 2, 3):
        assert abs(dihedral(c[i - 1], n[i], ca[i], c[i]) - (-57.0)) < 1.0
        assert abs(dihedral(n[i], ca[i], c[i], n[i + 1]) - (-47.0)) < 1.0
        omega = abs(dihedral(ca[i], c[i], n[i + 1], ca[i + 1]))
        assert abs(omega - 180.0) < 1.0, "peptide bond must be trans"


def test_forces_shape_finiteness_and_newton_third_law():
    batch = synthetic_batch([SyntheticSpec(8)], seed=0)
    f = batch.atoms.forces
    assert f.shape == batch.atoms.positions.shape
    assert bool(torch.isfinite(f).all())
    # the fixture potential is an exact pair potential plus a per-residue
    # centroid restraint, so the total force must vanish
    assert float(torch.linalg.norm(f.sum(0))) < 1e-3


def test_forces_can_be_omitted():
    batch = synthetic_batch([SyntheticSpec(4)], seed=0, with_forces=False)
    assert batch.atoms.forces is None
    assert batch.atoms.force_valid is None
    batch.validate()


def test_forces_are_deterministic_given_positions():
    batch = synthetic_batch([SyntheticSpec(6)], seed=0)
    again = synthetic_forces(batch.atoms.positions, batch.atoms.atom_to_residue)
    assert torch.allclose(batch.atoms.forces, again)


def test_temperature_and_identity_metadata():
    batch = synthetic_batch([SyntheticSpec(3)] * 6, seed=0)
    assert batch.temperature.shape == (6,)
    assert set(batch.temperature.tolist()) <= {320.0, 348.0, 379.0, 413.0, 450.0}
    assert len(batch.domain_id) == 6
    assert batch.units.length == "angstrom"
    assert batch.units.force == "kcal/mol/angstrom"
    # frame_index counts frames, not nanoseconds: mdCATH stores no timestamp
    assert batch.frame_index.dtype == torch.int64


def test_resid_original_is_not_the_node_index():
    """mdCATH keeps original PDB numbering; using it as a 0-based index is a bug."""
    batch = synthetic_batch([SyntheticSpec(5)], seed=0)
    assert int(batch.residues.resid_original[0]) == 8
    assert not torch.equal(
        batch.residues.resid_original, torch.arange(batch.num_residues)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_respected(dtype):
    batch = synthetic_batch([SyntheticSpec(4)], seed=0, dtype=dtype)
    assert batch.atoms.positions.dtype == dtype
    assert batch.backbone.ca_positions.dtype == dtype
    batch.validate()
