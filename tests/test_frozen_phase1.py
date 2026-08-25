"""Frozen Phase 1 feature extraction (Phase 1.5, Checkpoint 3).

Two classes of failure are guarded here, and only one of them is visible without
a test.

**Contract drift.** A checkpoint that loads but produces features of a different
width, row order or frame is worse than one that fails to load: everything
downstream keeps running and every number is wrong. The recorded contract, the
rebuilt model's contract and any runtime expectation are all cross-checked.

**Label leakage.** If ground-truth forces reach a production arm, Phase 1.5
measures nothing. The separation is structural -- a different class from a
different method -- so these tests assert on types and on the *absence* of
fields, not on values.

Most tests build a small Phase 1 checkpoint in ``tmp_path`` so the file stays
runnable without the 14 MB real checkpoint; the ``mdcath`` ones then check the
real one.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.transition import (  # noqa: E402
    FeatureBundle,
    FrozenPhase1Extractor,
    OracleFeatureBundle,
    Phase1FeatureCache,
    assert_production_safe,
    checkpoint_fingerprint,
    merge_bundles,
    split_bundle,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
REAL_CHECKPOINT = os.path.join(ROOT, "runs", "phase1_full", "last.pt")
PLM_DIM = 32


def small_config() -> LocalPhysicsConfig:
    """A Phase 1 model of the same family, narrow enough to run in a test."""
    return LocalPhysicsConfig(
        encoder=EncoderConfig(
            plm_dim=PLM_DIM,
            num_cycles=1,
            irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
        ),
        use_energy_branch=False,
    )


def write_checkpoint(path, *, config=None, contract=None, step: int = 1234) -> str:
    """Write a checkpoint with the same payload shape ``Phase1Trainer`` writes."""
    config = config or small_config()
    model = LocalPhysicsModel(config)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": config,
        "step": step,
        "latent_contract": contract if contract is not None else model.latent_contract(),
    }
    torch.save(payload, str(path))
    return str(path)


def batch(sizes=(6, 5), seed: int = 0):
    return synthetic_batch(
        [SyntheticSpec(n) for n in sizes], seed=seed, plm_dim=PLM_DIM,
        include_hydrogens=True,
    )


@pytest.fixture
def extractor(tmp_path):
    return FrozenPhase1Extractor.from_checkpoint(
        write_checkpoint(tmp_path / "phase1.pt")
    )


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


def test_extractor_reports_the_recorded_contract_and_provenance(tmp_path):
    path = write_checkpoint(tmp_path / "phase1.pt", step=7)
    extractor = FrozenPhase1Extractor.from_checkpoint(path)
    assert extractor.contract["physics_latent_dim"] == extractor.phase1.irreps.dim
    assert extractor.metadata["step"] == 7
    assert extractor.metadata["checkpoint_sha256"] == checkpoint_fingerprint(path)
    assert extractor.metadata["checkpoint_path"] == os.path.abspath(path)


def test_bundle_matches_the_latent_contract(extractor):
    bundle = extractor(batch())
    contract = extractor.contract
    assert bundle.latent_dim == contract["physics_latent_dim"]
    assert bundle.physics_latent_irreps == contract["physics_latent_irreps"]
    assert bundle.latent_frame == contract["frame"]
    assert bundle.latent_row_order == contract["row_order"]


def test_residue_rows_align_with_the_batch(extractor):
    """Row ``i`` of the latent is residue ``i`` of ``batch.residues``, in order."""
    sizes = (6, 5)
    state = batch(sizes)
    joint = extractor(state)
    assert joint.physics_latent.shape[0] == state.num_residues
    assert joint.atom_force_mean.shape[0] == state.num_atoms
    assert torch.equal(joint.residue_batch_index, state.residues.batch_index)
    assert torch.equal(joint.atom_to_residue, state.atoms.atom_to_residue)

    offset = 0
    for graph, size in enumerate(sizes):
        rows = (joint.residue_batch_index == graph).nonzero(as_tuple=True)[0]
        assert rows.tolist() == list(range(offset, offset + size))
        offset += size


def test_a_graphs_features_do_not_depend_on_its_batch_neighbours(extractor):
    """No edge crosses a graph, so a protein's rows must be batch-independent."""
    alone = extractor(batch(sizes=(6,), seed=3))
    joint = extractor(batch(sizes=(6, 5), seed=3))
    first = split_bundle(joint, 0)
    assert torch.allclose(first.physics_latent, alone.physics_latent, atol=1e-5)
    assert torch.allclose(first.atom_force_mean, alone.atom_force_mean, atol=1e-5)


def test_atom_rows_point_at_their_own_residues(extractor):
    bundle = extractor(batch())
    graph_of_atom = bundle.residue_batch_index[bundle.atom_to_residue]
    assert torch.equal(graph_of_atom, bundle.atom_batch_index)


def test_a_runtime_expectation_that_the_checkpoint_violates_is_refused(tmp_path):
    path = write_checkpoint(tmp_path / "phase1.pt")
    with pytest.raises(ValueError, match="latent contract mismatch"):
        FrozenPhase1Extractor.from_checkpoint(
            path, expect={"physics_latent_dim": 152}
        )


def test_a_runtime_expectation_that_matches_is_accepted(tmp_path):
    path = write_checkpoint(tmp_path / "phase1.pt")
    model = LocalPhysicsModel(small_config())
    extractor = FrozenPhase1Extractor.from_checkpoint(
        path, expect={"physics_latent_dim": model.irreps.dim, "lmax": 2}
    )
    assert extractor.contract["lmax"] == 2


def test_a_tampered_recorded_contract_is_detected(tmp_path):
    """The recorded promise and the rebuilt model must agree."""
    model = LocalPhysicsModel(small_config())
    lying = dict(model.latent_contract())
    lying["physics_latent_dim"] = 999
    path = write_checkpoint(tmp_path / "phase1.pt", contract=lying)
    with pytest.raises(ValueError, match="physics_latent_dim"):
        FrozenPhase1Extractor.from_checkpoint(path)


def test_a_checkpoint_without_a_model_config_is_refused(tmp_path):
    path = str(tmp_path / "bad.pt")
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="model_config"):
        FrozenPhase1Extractor.from_checkpoint(path)


def test_loading_does_not_modify_the_checkpoint(tmp_path):
    """Phase 1 is a finished artefact; Phase 1.5 must not write to it."""
    path = write_checkpoint(tmp_path / "phase1.pt")
    before = checkpoint_fingerprint(path)
    mtime = os.path.getmtime(path)
    extractor = FrozenPhase1Extractor.from_checkpoint(path)
    extractor(batch())
    assert checkpoint_fingerprint(path) == before
    assert os.path.getmtime(path) == mtime


# --------------------------------------------------------------------------
# frozen
# --------------------------------------------------------------------------


def test_frozen_model_is_in_eval_mode_with_no_trainable_parameters(extractor):
    assert not extractor.phase1.training
    assert all(not p.requires_grad for p in extractor.phase1.parameters())


def test_frozen_bundle_carries_no_gradient(extractor):
    state = extractor(batch()).requires_grad_state()
    assert not any(state.values()), [k for k, v in state.items() if v]


def test_a_downstream_backward_produces_no_phase1_gradient(extractor):
    """The check that matters: not just the flag, but an actual backward pass."""
    bundle = extractor(batch())
    head = torch.nn.Linear(bundle.latent_dim, 3)
    head(bundle.physics_latent).pow(2).mean().backward()

    assert head.weight.grad is not None
    assert all(p.grad is None for p in extractor.phase1.parameters())


def test_unfreezing_is_a_configuration_change_not_a_different_class(tmp_path):
    path = write_checkpoint(tmp_path / "phase1.pt")
    extractor = FrozenPhase1Extractor.from_checkpoint(path, freeze=False)
    assert extractor.phase1.training
    assert all(p.requires_grad for p in extractor.phase1.parameters())
    bundle = extractor(batch())
    assert bundle.physics_latent.requires_grad


# --------------------------------------------------------------------------
# the production / oracle separation
# --------------------------------------------------------------------------


def test_the_production_bundle_exposes_no_ground_truth(extractor):
    bundle = extractor(batch())
    assert isinstance(bundle, FeatureBundle)
    for forbidden in ("forces", "atom_force", "atoms", "batch", "future"):
        assert not hasattr(bundle, forbidden), forbidden


def test_assert_production_safe_rejects_an_oracle_bundle(extractor):
    oracle = extractor.oracle_bundle(batch())
    with pytest.raises(TypeError, match="OracleFeatureBundle reached a production"):
        assert_production_safe(oracle)
    # ... and the way through is explicit
    assert assert_production_safe(oracle.production) is oracle.production


def test_assert_production_safe_rejects_anything_else(extractor):
    with pytest.raises(TypeError, match="expected a FeatureBundle"):
        assert_production_safe({"physics_latent": torch.zeros(3, 4)})


def test_the_oracle_bundle_carries_labels_and_the_production_features(extractor):
    state = batch()
    oracle = extractor.oracle_bundle(state)
    assert isinstance(oracle, OracleFeatureBundle)
    assert torch.equal(oracle.atom_force, state.atoms.forces)
    assert isinstance(oracle.production, FeatureBundle)
    assert oracle.production.num_atoms == state.num_atoms


def test_the_oracle_arm_refuses_a_batch_without_labels(extractor):
    unlabelled = dataclasses.replace(
        batch(), atoms=dataclasses.replace(batch().atoms, forces=None, force_valid=None)
    )
    with pytest.raises(ValueError, match="oracle arm needs"):
        extractor.oracle_bundle(unlabelled)


def test_predicted_forces_are_not_the_labels(extractor):
    """Sanity: the production bundle's forces are predictions, not a copy."""
    state = batch()
    bundle = extractor(state)
    assert not torch.allclose(bundle.atom_force_mean, state.atoms.forces)


# --------------------------------------------------------------------------
# split / merge
# --------------------------------------------------------------------------


def test_split_then_merge_reproduces_the_batch(extractor):
    bundle = extractor(batch(sizes=(6, 5, 4)))
    merged = merge_bundles([split_bundle(bundle, g) for g in range(bundle.num_graphs)])

    assert merged.num_graphs == bundle.num_graphs
    assert torch.equal(merged.physics_latent, bundle.physics_latent)
    assert torch.equal(merged.atom_force_mean, bundle.atom_force_mean)
    assert torch.equal(merged.atom_to_residue, bundle.atom_to_residue)
    assert torch.equal(merged.residue_batch_index, bundle.residue_batch_index)
    assert torch.equal(merged.atom_batch_index, bundle.atom_batch_index)
    assert torch.equal(merged.frames.rotation, bundle.frames.rotation)


def test_a_split_bundle_is_a_standalone_single_graph(extractor):
    bundle = extractor(batch(sizes=(6, 5)))
    second = split_bundle(bundle, 1)
    assert second.num_graphs == 1
    assert second.num_residues == 5
    assert int(second.residue_batch_index.max()) == 0
    assert int(second.atom_to_residue.min()) == 0
    assert int(second.atom_to_residue.max()) == 4


def test_merging_refuses_bundles_with_different_irreps(extractor, tmp_path):
    wide = FrozenPhase1Extractor.from_checkpoint(
        write_checkpoint(
            tmp_path / "wide.pt",
            config=dataclasses.replace(
                small_config(),
                encoder=dataclasses.replace(
                    small_config().encoder,
                    irreps=IrrepsConfig(scalar_channels=8, vector_channels=2,
                                        tensor_channels=1),
                ),
            ),
        )
    )
    a = split_bundle(extractor(batch(sizes=(4,))), 0)
    b = split_bundle(wide(batch(sizes=(4,))), 0)
    with pytest.raises(ValueError, match="disagree on irreps"):
        merge_bundles([a, b])


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def cache_for(tmp_path, *, checkpoint="ck", config="cfg") -> Phase1FeatureCache:
    return Phase1FeatureCache(
        str(tmp_path / "cache"), checkpoint_sha256=checkpoint, config_hash=config
    )


def test_cache_round_trips_a_frame(extractor, tmp_path):
    cache = cache_for(tmp_path)
    bundle = split_bundle(extractor(batch(sizes=(6,))), 0)
    key = Phase1FeatureCache.frame_key("1abcA00", "320", "0", 17)
    cache.save("1abcA00", {key: bundle})

    restored = cache.load("1abcA00")[key]
    assert torch.equal(restored.physics_latent, bundle.physics_latent)
    assert restored.num_graphs == 1


def test_cache_key_names_the_frame_not_the_pair():
    """One frame is the current state of many pairs; it is cached once."""
    key = Phase1FeatureCache.frame_key("1abcA00", "320", "3", 41)
    assert key == "1abcA00/320/3/41"


def test_cache_from_a_different_checkpoint_is_refused(extractor, tmp_path):
    bundle = split_bundle(extractor(batch(sizes=(5,))), 0)
    cache_for(tmp_path).save("dom", {"dom/320/0/1": bundle})

    other = cache_for(tmp_path, checkpoint="a-different-checkpoint")
    with pytest.raises(ValueError, match="written by a different Phase 1"):
        other.load("dom")


def test_cache_from_a_different_config_is_refused(extractor, tmp_path):
    bundle = split_bundle(extractor(batch(sizes=(5,))), 0)
    cache_for(tmp_path).save("dom", {"dom/320/0/1": bundle})

    other = cache_for(tmp_path, config="a-different-config")
    with pytest.raises(ValueError, match="written by a different Phase 1"):
        other.load("dom")


def test_cache_get_returns_none_when_a_frame_is_absent(extractor, tmp_path):
    cache = cache_for(tmp_path)
    bundle = split_bundle(extractor(batch(sizes=(5,))), 0)
    cache.save("dom", {"dom/320/0/1": bundle})

    assert cache.get("dom", ["dom/320/0/1"]) is not None
    assert cache.get("dom", ["dom/320/0/1", "dom/320/0/2"]) is None
    assert cache.get("missing-domain", ["x"]) is None


def test_cache_writes_leave_no_temporary_file(extractor, tmp_path):
    cache = cache_for(tmp_path)
    cache.save("dom", {"dom/320/0/1": split_bundle(extractor(batch(sizes=(5,))), 0)})
    leftovers = [p for p in os.listdir(cache.root) if p.endswith(".tmp")]
    assert leftovers == []


def test_cache_merges_new_frames_into_an_existing_shard(extractor, tmp_path):
    cache = cache_for(tmp_path)
    bundle = split_bundle(extractor(batch(sizes=(5,))), 0)
    cache.save("dom", {"dom/320/0/1": bundle})
    cache.save("dom", {"dom/320/0/2": bundle})
    assert set(cache.load("dom")) == {"dom/320/0/1", "dom/320/0/2"}


def test_cache_holds_frames_of_different_sizes(extractor, tmp_path):
    """Ragged: two domains with different atom and residue counts."""
    cache = cache_for(tmp_path)
    joint = extractor(batch(sizes=(6, 4)))
    first, second = split_bundle(joint, 0), split_bundle(joint, 1)
    cache.save("small", {"small/320/0/0": second})
    cache.save("large", {"large/320/0/0": first})

    assert cache.load("small")["small/320/0/0"].num_residues == 4
    assert cache.load("large")["large/320/0/0"].num_residues == 6


def test_cache_refuses_a_multi_graph_entry(extractor, tmp_path):
    with pytest.raises(ValueError, match="graphs; cache one"):
        cache_for(tmp_path).save("dom", {"dom/320/0/1": extractor(batch(sizes=(4, 4)))})


# --------------------------------------------------------------------------
# the real Phase 1 checkpoint
# --------------------------------------------------------------------------


@pytest.mark.mdcath
@pytest.mark.skipif(not os.path.exists(REAL_CHECKPOINT), reason="no Phase 1 checkpoint")
def test_the_real_checkpoint_restores_its_published_contract():
    extractor = FrozenPhase1Extractor.from_checkpoint(
        REAL_CHECKPOINT,
        expect={
            "physics_latent_irreps": "64x0e+16x1o+8x2e",
            "physics_latent_dim": 152,
            "frame": "global",
            "target_scope": "heavy_atom",
            "num_cycles": 2,
            "lmax": 2,
        },
    )
    assert extractor.metadata["step"] == 120000
    assert all(not p.requires_grad for p in extractor.phase1.parameters())
    assert sum(p.numel() for p in extractor.phase1.parameters()) == 1_177_753
