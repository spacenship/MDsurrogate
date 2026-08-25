"""Residue frames: orthonormality, SE(3) behaviour, chirality, degeneracy."""

from __future__ import annotations

import pytest
import torch

from force_md.data import SyntheticSpec, synthetic_batch
from force_md.geometry import (
    apply_rigid_transform,
    atom_local_coordinates,
    build_residue_frames,
    frames_from_batch,
    random_rotation_matrix,
    to_global_points,
    to_global_vectors,
    to_local_points,
    to_local_vectors,
)


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(7), SyntheticSpec(4)], seed=0, dtype=torch.float64)


@pytest.fixture
def frames(batch):
    return frames_from_batch(batch)


# --------------------------------------------------------------------------
# the rotation itself
# --------------------------------------------------------------------------


def test_rotation_is_orthonormal_and_right_handed(frames):
    r = frames.rotation
    eye = torch.eye(3, dtype=r.dtype).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-10)
    det = torch.linalg.det(r)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-10)
    assert bool((det > 0).all()), "a det=-1 frame would mirror the residue"


def test_rotation_columns_are_the_local_axes(batch, frames):
    """R's columns are e1,e2,e3 in global coordinates. A transpose here would
    silently flip every equivariant feature downstream."""
    bb = batch.backbone
    e1 = bb.c_positions - bb.ca_positions
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True)
    assert torch.allclose(frames.rotation[:, :, 0], e1, atol=1e-10)

    u = bb.n_positions - bb.ca_positions
    v2 = u - (u * e1).sum(-1, keepdim=True) * e1
    e2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True)
    assert torch.allclose(frames.rotation[:, :, 1], e2, atol=1e-10)
    assert torch.allclose(
        frames.rotation[:, :, 2], torch.linalg.cross(e1, e2, dim=-1), atol=1e-10
    )


def test_origin_is_ca(batch, frames):
    assert torch.allclose(frames.origin, batch.backbone.ca_positions, atol=1e-12)


def test_ca_is_the_local_origin(batch, frames):
    idx = torch.arange(batch.num_residues)
    y = to_local_points(batch.backbone.ca_positions, frames, idx)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-10)


def test_c_lies_on_the_local_x_axis(batch, frames):
    """By construction e1 points at C, so C's local coords are (|C-CA|, 0, 0)."""
    idx = torch.arange(batch.num_residues)
    y = to_local_points(batch.backbone.c_positions, frames, idx)
    assert torch.allclose(y[:, 1:], torch.zeros_like(y[:, 1:]), atol=1e-10)
    assert bool((y[:, 0] > 0).all())


def test_n_lies_in_the_local_xy_plane(batch, frames):
    idx = torch.arange(batch.num_residues)
    y = to_local_points(batch.backbone.n_positions, frames, idx)
    assert torch.allclose(y[:, 2], torch.zeros_like(y[:, 2]), atol=1e-10)
    assert bool((y[:, 1] > 0).all()), "e2 must point toward N"


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


def test_point_round_trip(batch, frames):
    x = batch.atoms.positions
    idx = batch.atoms.atom_to_residue
    assert torch.allclose(to_global_points(to_local_points(x, frames, idx), frames, idx),
                          x, atol=1e-10)


def test_vector_round_trip(batch, frames):
    v = batch.atoms.forces
    idx = batch.atoms.atom_to_residue
    assert torch.allclose(to_global_vectors(to_local_vectors(v, frames, idx), frames, idx),
                          v, atol=1e-10)


def test_points_and_vectors_differ_by_the_origin(batch, frames):
    """A vector must not pick up the translation; a point must."""
    x = batch.atoms.positions
    idx = batch.atoms.atom_to_residue
    y_point = to_local_points(x, frames, idx)
    y_vector = to_local_vectors(x, frames, idx)
    assert not torch.allclose(y_point, y_vector)
    shifted = to_local_vectors(x - frames.origin[idx], frames, idx)
    assert torch.allclose(y_point, shifted, atol=1e-10)


# --------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------


def test_frame_is_equivariant_under_rigid_motion(batch):
    g = torch.Generator().manual_seed(0)
    q = random_rotation_matrix(g, dtype=torch.float64)
    t = torch.tensor([3.0, -7.5, 0.25], dtype=torch.float64)

    f0 = frames_from_batch(batch)
    f1 = frames_from_batch(apply_rigid_transform(batch, q, t))

    assert torch.allclose(f1.rotation, q @ f0.rotation, atol=1e-10)
    assert torch.allclose(f1.origin, f0.origin @ q.T + t, atol=1e-10)


def test_local_coordinates_are_rigid_invariant(batch):
    g = torch.Generator().manual_seed(1)
    q = random_rotation_matrix(g, dtype=torch.float64)
    t = torch.tensor([-11.0, 2.0, 5.0], dtype=torch.float64)

    y0, _ = atom_local_coordinates(batch)
    y1, _ = atom_local_coordinates(apply_rigid_transform(batch, q, t))
    assert torch.allclose(y0, y1, atol=1e-9)


def test_local_forces_are_rotation_invariant(batch):
    """Uncertainty is predicted in the local frame, so this invariance is what
    makes a local-frame covariance meaningful."""
    g = torch.Generator().manual_seed(2)
    q = random_rotation_matrix(g, dtype=torch.float64)
    t = torch.zeros(3, dtype=torch.float64)

    f0 = frames_from_batch(batch)
    idx = batch.atoms.atom_to_residue
    local0 = to_local_vectors(batch.atoms.forces, f0, idx)

    moved = apply_rigid_transform(batch, q, t)
    local1 = to_local_vectors(moved.atoms.forces, frames_from_batch(moved), idx)
    assert torch.allclose(local0, local1, atol=1e-9)


def test_translation_alone_leaves_rotation_unchanged(batch):
    t = torch.tensor([100.0, -50.0, 7.0], dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    f0 = frames_from_batch(batch)
    f1 = frames_from_batch(apply_rigid_transform(batch, eye, t))
    assert torch.allclose(f0.rotation, f1.rotation, atol=1e-12)
    assert torch.allclose(f1.origin - f0.origin, t.expand_as(f0.origin), atol=1e-10)


# --------------------------------------------------------------------------
# chirality: reflection is NOT a symmetry
# --------------------------------------------------------------------------


def test_reflection_is_not_a_symmetry_of_the_frame(batch):
    """Mirroring must NOT produce the mirrored frame.

    e3 is a cross product, so under an improper transform M the frame becomes
    [Me1, Me2, -Me3] != M R. If this ever passed, the model would be free to
    treat a D-amino-acid protein as identical to its L form.
    """
    m = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    f0 = frames_from_batch(batch)
    f1 = frames_from_batch(apply_rigid_transform(batch, m, torch.zeros(3, dtype=torch.float64)))

    assert not torch.allclose(f1.rotation, m @ f0.rotation, atol=1e-6)
    # the third column is exactly the one that flips
    assert torch.allclose(f1.rotation[:, :, 0], (m @ f0.rotation)[:, :, 0], atol=1e-10)
    assert torch.allclose(f1.rotation[:, :, 2], -(m @ f0.rotation)[:, :, 2], atol=1e-10)
    # frames stay right-handed even for the mirrored input
    assert torch.allclose(torch.linalg.det(f1.rotation),
                          torch.ones(f1.num_residues, dtype=torch.float64), atol=1e-10)


def test_reflection_changes_local_coordinates(batch):
    """Local coordinates are SE(3)-invariant but chirality-sensitive."""
    m = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    y0, _ = atom_local_coordinates(batch)
    y1, _ = atom_local_coordinates(
        apply_rigid_transform(batch, m, torch.zeros(3, dtype=torch.float64))
    )
    assert not torch.allclose(y0, y1, atol=1e-6)


# --------------------------------------------------------------------------
# degeneracy
# --------------------------------------------------------------------------


def test_collinear_backbone_is_flagged_not_silently_used():
    batch = synthetic_batch([SyntheticSpec(6, drop_frame_atom_at=(2,))], seed=0,
                            dtype=torch.float64)
    frames = frames_from_batch(batch)
    assert not bool(frames.valid[2])
    assert bool(frames.valid[0])
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(frames.rotation[2], eye, atol=1e-12)
    assert bool(torch.isfinite(frames.rotation).all())


def test_coincident_atoms_do_not_produce_nan():
    ca = torch.zeros(3, 3, dtype=torch.float64)
    c = ca.clone()  # C == CA  -> e1 undefined
    n = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float64)
    frames = build_residue_frames(n, ca, c)
    assert not bool(frames.valid.any())
    assert bool(torch.isfinite(frames.rotation).all())


def test_prior_valid_is_respected():
    batch = synthetic_batch([SyntheticSpec(4)], seed=0, dtype=torch.float64)
    bb = batch.backbone
    prior = torch.tensor([True, False, True, True])
    frames = build_residue_frames(bb.n_positions, bb.ca_positions, bb.c_positions,
                                  prior_valid=prior)
    assert frames.valid.tolist() == [True, False, True, True]


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------


def test_gradients_are_finite_for_valid_frames(batch):
    bb = batch.backbone
    n = bb.n_positions.clone().requires_grad_(True)
    ca = bb.ca_positions.clone().requires_grad_(True)
    c = bb.c_positions.clone().requires_grad_(True)
    frames = build_residue_frames(n, ca, c)
    frames.rotation.sum().backward()
    for t in (n, ca, c):
        assert bool(torch.isfinite(t.grad).all())


def test_gradients_are_finite_at_a_degenerate_frame():
    """A collinear residue must not poison the backward pass of the whole batch."""
    n = torch.tensor([[-1.0, 0, 0], [1.0, 0.5, 0]], dtype=torch.float64)
    ca = torch.zeros(2, 3, dtype=torch.float64)
    c = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)  # row 0 collinear
    n = n.requires_grad_(True)
    ca = ca.requires_grad_(True)
    c = c.requires_grad_(True)
    frames = build_residue_frames(n, ca, c)
    assert frames.valid.tolist() == [False, True]
    frames.rotation.sum().backward()
    for t in (n, ca, c):
        assert bool(torch.isfinite(t.grad).all()), "degenerate row produced non-finite grad"


def test_local_coordinates_are_differentiable_wrt_positions(batch):
    pos = batch.atoms.positions.clone().requires_grad_(True)
    import dataclasses
    b2 = dataclasses.replace(batch, atoms=dataclasses.replace(batch.atoms, positions=pos))
    y, _ = atom_local_coordinates(b2)
    y.pow(2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())
    assert float(pos.grad.abs().sum()) > 0


def test_near_degenerate_frame_is_numerically_stable():
    """Almost-collinear backbones must stay finite and orthonormal."""
    for angle in (1e-2, 1e-4, 1e-6):
        n = torch.tensor([[torch.cos(torch.tensor(angle)).item(),
                           torch.sin(torch.tensor(angle)).item(), 0.0]], dtype=torch.float64)
        ca = torch.zeros(1, 3, dtype=torch.float64)
        c = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        frames = build_residue_frames(n, ca, c)
        assert bool(torch.isfinite(frames.rotation).all())
        if bool(frames.valid[0]):
            r = frames.rotation[0]
            assert torch.allclose(r.T @ r, torch.eye(3, dtype=torch.float64), atol=1e-8)
            assert abs(float(torch.linalg.det(r)) - 1.0) < 1e-8


# --------------------------------------------------------------------------
# indexing / batching
# --------------------------------------------------------------------------


def test_atoms_use_their_own_parent_frame(batch):
    """Cross-checks the gather: atom a of residue i must use frame i, not i+1."""
    y, frames = atom_local_coordinates(batch)
    idx = batch.atoms.atom_to_residue
    for a in (0, 5, 17, batch.num_atoms - 1):
        i = int(idx[a])
        expected = frames.rotation[i].T @ (batch.atoms.positions[a] - frames.origin[i])
        assert torch.allclose(y[a], expected, atol=1e-12)


def test_second_graph_is_unaffected_by_the_first(batch):
    """No cross-graph leakage: frames are per-residue and index-local."""
    single = synthetic_batch([SyntheticSpec(4)], seed=0, dtype=torch.float64)
    pair = synthetic_batch([SyntheticSpec(4), SyntheticSpec(9)], seed=0, dtype=torch.float64)
    f_single = frames_from_batch(single)
    f_pair = frames_from_batch(pair)
    assert torch.allclose(f_single.rotation, f_pair.rotation[:4], atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    batch = synthetic_batch([SyntheticSpec(4)], seed=0, dtype=dtype)
    frames = frames_from_batch(batch)
    assert frames.rotation.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    eye = torch.eye(3, dtype=dtype).expand_as(frames.rotation)
    assert torch.allclose(frames.rotation.transpose(-1, -2) @ frames.rotation, eye, atol=tol)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_frames_on_cuda_match_cpu(batch):
    cpu = frames_from_batch(batch)
    gpu = frames_from_batch(batch.to("cuda:0"))
    assert torch.allclose(cpu.rotation, gpu.rotation.cpu(), atol=1e-10)
    assert bool(torch.equal(cpu.valid, gpu.valid.cpu()))


def test_random_rotation_matrix_is_proper():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        q = random_rotation_matrix(g, dtype=torch.float64)
        assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-10)
        assert abs(float(torch.linalg.det(q)) - 1.0) < 1e-10
