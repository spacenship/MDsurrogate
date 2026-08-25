"""Contract validation: what must be rejected, and when checking happens."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from force_md.data import (
    HierarchicalProteinBatch,
    ProteinAtomBatch,
    SyntheticSpec,
    synthetic_batch,
)


@pytest.fixture
def batch() -> HierarchicalProteinBatch:
    return synthetic_batch([SyntheticSpec(6), SyntheticSpec(4)], seed=0)


def test_valid_batch_passes(batch):
    batch.validate()  # must not raise
    assert batch.num_graphs == 2
    assert batch.num_residues == 10
    assert batch.num_atoms == batch.atoms.positions.shape[0]


def test_validation_is_not_run_in_constructor():
    """Constructing must stay cheap; validation is an explicit, separate call.

    Building a batch is on the training hot path, so a malformed container has
    to be constructible -- and then rejected by validate().
    """
    bad = ProteinAtomBatch(
        positions=torch.zeros(3, 3),
        atom_to_residue=torch.arange(3, dtype=torch.int64),
        batch_index=torch.zeros(3, dtype=torch.int64),
        atomic_number=torch.zeros(3, dtype=torch.int64),
        atom_name_id=torch.zeros(3, dtype=torch.int64),
        is_backbone=torch.zeros(3, dtype=torch.bool),
        is_cap=torch.zeros(3, dtype=torch.bool),
    )  # no exception here: construction stays on the hot path
    with pytest.raises(ValueError, match="only 1 target node"):
        bad.validate(num_residues=1, num_graphs=1)
    bad.validate(num_residues=3, num_graphs=1)  # consistent sizes: accepted


def test_atom_to_residue_out_of_range(batch):
    atoms = dataclasses.replace(
        batch.atoms, atom_to_residue=batch.atoms.atom_to_residue + batch.num_residues
    )
    bad = dataclasses.replace(batch, atoms=atoms)
    with pytest.raises(ValueError, match="indexes"):
        bad.validate()


def test_atom_to_residue_must_be_non_decreasing(batch):
    shuffled = batch.atoms.atom_to_residue.clone()
    shuffled[0], shuffled[-1] = shuffled[-1].clone(), shuffled[0].clone()
    atoms = dataclasses.replace(batch.atoms, atom_to_residue=shuffled)
    with pytest.raises(ValueError, match="non-decreasing"):
        dataclasses.replace(batch, atoms=atoms).validate()


def test_wrong_dtype_is_rejected(batch):
    atoms = dataclasses.replace(
        batch.atoms, atom_to_residue=batch.atoms.atom_to_residue.to(torch.int32)
    )
    with pytest.raises(ValueError, match="int64"):
        dataclasses.replace(batch, atoms=atoms).validate()


def test_atom_graph_must_match_parent_residue_graph(batch):
    """An atom in a different graph than its residue is a batch-leak bug.

    The boundary is moved by one atom, which keeps ``batch_index``
    non-decreasing so this exercises the cross-level check rather than the
    monotonicity check.
    """
    bi = batch.atoms.batch_index.clone()
    first_of_graph1 = int((bi == 1).nonzero()[0])
    bi[first_of_graph1] = 0
    atoms = dataclasses.replace(batch.atoms, batch_index=bi)
    with pytest.raises(ValueError, match="different graph than its parent residue"):
        dataclasses.replace(batch, atoms=atoms).validate()


def test_residue_with_no_atoms_is_rejected():
    """An empty pooling segment yields a zero feature indistinguishable from data."""
    b = synthetic_batch([SyntheticSpec(3)], seed=0)
    keep = b.atoms.atom_to_residue != 1
    atoms = ProteinAtomBatch(
        positions=b.atoms.positions[keep],
        atom_to_residue=b.atoms.atom_to_residue[keep],
        batch_index=b.atoms.batch_index[keep],
        atomic_number=b.atoms.atomic_number[keep],
        atom_name_id=b.atoms.atom_name_id[keep],
        is_backbone=b.atoms.is_backbone[keep],
        is_cap=b.atoms.is_cap[keep],
    )
    with pytest.raises(ValueError, match="own no atoms"):
        dataclasses.replace(b, atoms=atoms).validate()


def test_residue_to_backbone_must_be_a_permutation(batch):
    r2b = batch.backbone.residue_to_backbone.clone()
    r2b[0] = r2b[1]
    bb = dataclasses.replace(batch.backbone, residue_to_backbone=r2b)
    with pytest.raises(ValueError, match="permutation"):
        dataclasses.replace(batch, backbone=bb).validate()


def test_force_valid_without_forces_is_rejected(batch):
    atoms = dataclasses.replace(batch.atoms, forces=None)
    with pytest.raises(ValueError, match="without forces"):
        dataclasses.replace(batch, atoms=atoms).validate()


def test_domain_id_length_must_match_batch(batch):
    with pytest.raises(ValueError, match="domain_id"):
        dataclasses.replace(batch, domain_id=("only-one",)).validate()


def test_to_device_cpu_roundtrip(batch):
    moved = batch.to("cpu")
    moved.validate()
    assert moved.device == torch.device("cpu")
    assert torch.equal(moved.atoms.positions, batch.atoms.positions)
    # original is untouched
    assert batch.atoms.positions.device == torch.device("cpu")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_device_cuda(batch):
    moved = batch.to("cuda:0")
    moved.validate()
    assert moved.device.type == "cuda"
    assert moved.atoms.forces.device.type == "cuda"
    assert moved.residues.plm_embedding.device.type == "cuda"
    assert moved.backbone.ca_positions.device.type == "cuda"
    back = moved.to("cpu")
    assert torch.allclose(back.atoms.positions, batch.atoms.positions)


def test_mixed_device_is_rejected(batch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    atoms = dataclasses.replace(batch.atoms, positions=batch.atoms.positions.to("cuda:0"))
    with pytest.raises(ValueError, match="multiple devices"):
        dataclasses.replace(batch, atoms=atoms).validate()
