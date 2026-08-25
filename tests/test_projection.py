"""Residue force/torque projection: hand-computed values, symmetry laws, masks."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import (
    apply_rigid_transform,
    frames_from_batch,
    random_rotation_matrix,
    to_local_vectors,
)
from force_md.physics import (
    ResidueSumProjector,
    omitted_atom_residual,
    shift_torque_origin,
)


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(6), SyntheticSpec(4)], seed=0,
                           include_hydrogens=True, dtype=torch.float64)


# --------------------------------------------------------------------------
# hand-computed baseline
# --------------------------------------------------------------------------


def test_hand_computed_two_atom_residue():
    """Two atoms, known positions and forces, arithmetic done by hand.

    residue 0: atoms at (1,0,0) and (-1,0,0) about origin (0,0,0)
      f = (0,1,0) and (0,-1,0)
      F   = (0,0,0)                                 -- a pure couple
      tau = (1,0,0)x(0,1,0) + (-1,0,0)x(0,-1,0)
          = (0,0,1) + (0,0,1) = (0,0,2)
    """
    b = synthetic_batch([SyntheticSpec(1)], seed=0, dtype=torch.float64)
    n_atom = b.num_atoms
    positions = torch.zeros(n_atom, 3, dtype=torch.float64)
    positions[0] = torch.tensor([1.0, 0.0, 0.0])
    positions[1] = torch.tensor([-1.0, 0.0, 0.0])
    forces = torch.zeros(n_atom, 3, dtype=torch.float64)
    forces[0] = torch.tensor([0.0, 1.0, 0.0])
    forces[1] = torch.tensor([0.0, -1.0, 0.0])

    b = dataclasses.replace(
        b,
        atoms=dataclasses.replace(b.atoms, positions=positions, forces=forces,
                                  
                                  force_valid=torch.ones(n_atom, dtype=torch.bool)),
        backbone=dataclasses.replace(
            b.backbone, ca_positions=torch.zeros(1, 3, dtype=torch.float64)
        ),
    )
    out = ResidueSumProjector("all_atom")(b)
    assert torch.allclose(out.force, torch.zeros(1, 3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(out.torque, torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
                          atol=1e-12)


def test_projection_equals_an_explicit_python_loop(batch):
    """Cross-check the vectorised scatter against the literal definition."""
    proj = ResidueSumProjector("all_atom")
    out = proj(batch)
    a2r = batch.atoms.atom_to_residue
    origin = batch.backbone.ca_positions
    for i in (0, 3, 7):
        sel = (a2r == i).nonzero(as_tuple=True)[0]
        f = batch.atoms.forces[sel].sum(0)
        tau = torch.linalg.cross(
            batch.atoms.positions[sel] - origin[i], batch.atoms.forces[sel]
        ).sum(0)
        assert torch.allclose(out.force[i], f, atol=1e-12)
        assert torch.allclose(out.torque[i], tau, atol=1e-12)


# --------------------------------------------------------------------------
# scopes
# --------------------------------------------------------------------------


def test_heavy_scope_excludes_hydrogen(batch):
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    n_h = int((~batch.atoms.is_heavy).sum())
    assert n_h > 0
    assert int(heavy.num_atoms.sum()) == int(batch.atoms.is_heavy.sum())
    assert int(allat.num_atoms.sum()) == batch.num_atoms
    assert not torch.allclose(heavy.force, allat.force)


def test_heavy_and_all_atom_agree_when_there_is_no_hydrogen():
    b = synthetic_batch([SyntheticSpec(5)], seed=0, dtype=torch.float64)
    assert bool(b.atoms.is_heavy.all())
    heavy = ResidueSumProjector("heavy_atom")(b)
    allat = ResidueSumProjector("all_atom")(b)
    assert torch.allclose(heavy.force, allat.force, atol=1e-12)
    assert torch.allclose(heavy.torque, allat.torque, atol=1e-12)


def test_omitted_atom_residual_is_the_hydrogen_sum(batch):
    """The residual must equal the summed force on the excluded atoms exactly."""
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    df, dtau = omitted_atom_residual(allat, heavy)

    h = ~batch.atoms.is_heavy
    from force_md.nn.irreps import scatter_sum
    expected = scatter_sum(batch.atoms.forces * h.unsqueeze(-1).double(),
                           batch.atoms.atom_to_residue, batch.num_residues)
    assert torch.allclose(df, expected, atol=1e-12)
    assert df.shape == (batch.num_residues, 3)
    assert dtau.shape == (batch.num_residues, 3)


def test_residual_rejects_swapped_scopes(batch):
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    with pytest.raises(ValueError, match="expected"):
        omitted_atom_residual(heavy, allat)


def test_residual_rejects_mismatched_torque_origins(batch):
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    moved = shift_torque_origin(allat, allat.origin + 1.0)
    with pytest.raises(ValueError, match="origins differ"):
        omitted_atom_residual(moved, heavy)


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be"):
        ResidueSumProjector("backbone_only")


# --------------------------------------------------------------------------
# symmetry laws
# --------------------------------------------------------------------------


def test_force_and_torque_are_translation_invariant(batch):
    proj = ResidueSumProjector("all_atom")
    a = proj(batch)
    eye = torch.eye(3, dtype=torch.float64)
    t = torch.tensor([12.0, -3.0, 40.0], dtype=torch.float64)
    b = proj(apply_rigid_transform(batch, eye, t))
    assert torch.allclose(a.force, b.force, atol=1e-10)
    assert torch.allclose(a.torque, b.torque, atol=1e-10), (
        "torque must be translation invariant because its origin translates too"
    )


def test_force_and_torque_are_rotation_equivariant(batch):
    proj = ResidueSumProjector("all_atom")
    q = random_rotation_matrix(torch.Generator().manual_seed(0), dtype=torch.float64)
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a = proj(batch)
    b = proj(apply_rigid_transform(batch, q, t))
    assert torch.allclose(b.force, a.force @ q.T, atol=1e-10)
    assert torch.allclose(b.torque, a.torque @ q.T, atol=1e-10)


def test_torque_origin_shift_law(batch):
    """tau(o') = tau(o) + (o - o') x F, checked against a direct recomputation."""
    proj = ResidueSumProjector("all_atom")
    base = proj(batch)
    new_origin = batch.backbone.n_positions  # use N instead of CA
    shifted = shift_torque_origin(base, new_origin)
    direct = proj(batch, origin=new_origin)
    assert torch.allclose(shifted.torque, direct.torque, atol=1e-10)
    assert torch.allclose(shifted.force, direct.force, atol=1e-12)
    assert torch.allclose(shifted.origin, new_origin)


def test_torque_origin_is_irrelevant_when_net_force_vanishes():
    """A pure couple has the same torque about every point."""
    b = synthetic_batch([SyntheticSpec(1)], seed=0, dtype=torch.float64)
    n = b.num_atoms
    f = torch.randn(n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    f = f - f.mean(0)  # net zero
    b = dataclasses.replace(b, atoms=dataclasses.replace(
        b.atoms, forces=f, force_valid=torch.ones(n, dtype=torch.bool)))
    proj = ResidueSumProjector("all_atom")
    a = proj(b)
    c = proj(b, origin=b.backbone.ca_positions + 10.0)
    assert float(a.force.norm()) < 1e-10
    assert torch.allclose(a.torque, c.torque, atol=1e-9)


def test_local_frame_representation_is_rotation_invariant(batch):
    """Expressed in the residue frame, the target is invariant -- which is what
    makes a local-frame uncertainty well defined."""
    proj = ResidueSumProjector("all_atom")
    q = random_rotation_matrix(torch.Generator().manual_seed(2), dtype=torch.float64)
    t = torch.tensor([-4.0, 0.5, 2.0], dtype=torch.float64)
    idx = torch.arange(batch.num_residues)

    a = proj(batch)
    la = to_local_vectors(a.force, frames_from_batch(batch), idx)
    moved = apply_rigid_transform(batch, q, t)
    bb = proj(moved)
    lb = to_local_vectors(bb.force, frames_from_batch(moved), idx)
    assert torch.allclose(la, lb, atol=1e-9)


# --------------------------------------------------------------------------
# masks and validity
# --------------------------------------------------------------------------


def test_invalid_force_labels_invalidate_their_residue(batch):
    """mdCATH ships trajectories whose forces are a copy of coords; a silent sum
    over them would become a training target."""
    fv = batch.atoms.force_valid.clone()
    victim = int(batch.atoms.atom_to_residue[3])
    fv[3] = False
    b = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, force_valid=fv))
    out = ResidueSumProjector("all_atom")(b)
    assert not bool(out.valid[victim])
    assert bool(out.valid[(victim + 1) % batch.num_residues])


def test_validity_can_be_switched_off(batch):
    fv = batch.atoms.force_valid.clone()
    fv[3] = False
    b = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, force_valid=fv))
    out = ResidueSumProjector("all_atom")(b, require_valid_forces=False)
    assert bool(out.valid.all())


def test_heavy_scope_ignores_an_invalid_hydrogen(batch):
    """A bad hydrogen label must not invalidate a heavy-atom-only target."""
    h_idx = (~batch.atoms.is_heavy).nonzero(as_tuple=True)[0][0]
    fv = batch.atoms.force_valid.clone()
    fv[h_idx] = False
    b = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, force_valid=fv))
    heavy = ResidueSumProjector("heavy_atom")(b)
    allat = ResidueSumProjector("all_atom")(b)
    victim = int(batch.atoms.atom_to_residue[h_idx])
    assert bool(heavy.valid[victim])
    assert not bool(allat.valid[victim])


def test_residue_with_no_contributing_atoms_is_invalid():
    """A GLY has no side chain; a hypothetical hydrogen-only scope would be empty."""
    b = synthetic_batch([SyntheticSpec(4)], seed=0, dtype=torch.float64)
    assert bool(b.atoms.is_heavy.all())

    class HydrogenOnly(ResidueSumProjector):
        def atom_selection(self, batch):
            return ~batch.atoms.is_heavy

    out = HydrogenOnly("all_atom")(b)
    assert not bool(out.valid.any())
    assert torch.allclose(out.force, torch.zeros_like(out.force))


def test_missing_forces_raise_a_clear_error():
    b = synthetic_batch([SyntheticSpec(3)], seed=0, with_forces=False, dtype=torch.float64)
    with pytest.raises(ValueError, match="no atom forces"):
        ResidueSumProjector()(b)


# --------------------------------------------------------------------------
# the operator is reused for predictions
# --------------------------------------------------------------------------


def test_the_same_projector_aggregates_predicted_forces(batch):
    """Aggregation consistency in Checkpoint 7 compares like with like only if
    the prediction goes through the identical operator."""
    proj = ResidueSumProjector("heavy_atom")
    predicted = torch.randn_like(batch.atoms.forces)
    agg = proj(batch, predicted)
    target = proj(batch)
    assert agg.force.shape == target.force.shape
    assert agg.scope == target.scope
    assert torch.allclose(agg.origin, target.origin)
    # feeding the labels back through reproduces the target exactly
    same = proj(batch, batch.atoms.forces)
    assert torch.allclose(same.force, target.force, atol=1e-12)
    assert torch.allclose(same.torque, target.torque, atol=1e-12)


def test_projection_is_differentiable(batch):
    proj = ResidueSumProjector("all_atom")
    predicted = torch.randn_like(batch.atoms.forces).requires_grad_(True)
    out = proj(batch, predicted)
    (out.force.pow(2).sum() + out.torque.pow(2).sum()).backward()
    assert bool(torch.isfinite(predicted.grad).all())
    assert float(predicted.grad.abs().sum()) > 0


def test_projector_has_no_parameters():
    assert list(ResidueSumProjector().parameters()) == []
