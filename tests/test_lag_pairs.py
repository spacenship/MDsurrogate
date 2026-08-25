"""Raw-trajectory lag pairs (Phase 1.5, Checkpoint 1).

Most of these run offline against a **fake shard** written in the mdCATH schema,
because the properties being checked are properties of the index -- lag
arithmetic, trajectory boundaries, quarantine handling, determinism -- and those
must be testable without 623 GB on disk. The tests marked ``mdcath`` then confirm
the same arithmetic on real files.

The single most important assertion in this file is
``test_phase1_subsample_cannot_supply_a_one_nanosecond_lag``: Phase 1's index is
~12-13 frames apart, so building "1 ns" pairs from adjacent Phase 1 entries would
produce a 12-13 ns transition and nothing in the shapes or the loss would reveal
it.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from force_md.data import FrameGeometry, SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.data import residue_constants as rc  # noqa: E402
from force_md.data.adapters import (  # noqa: E402
    LagPairConfig,
    LagPairDataset,
    LagPairManifest,
    MdCathConfig,
    MdCathDataset,
    build_lag_pair_manifest,
    collate_lag_pairs,
    exact_lag_frames,
    restore_phase1_split,
)
from force_md.data.contracts import HierarchicalProteinBatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
HAS_DATA = bool(glob.glob(os.path.join(DATA_DIR, "*.h5")))

_Z_TO_ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


# --------------------------------------------------------------------------
# a minimal shard in the mdCATH schema
# --------------------------------------------------------------------------


def write_fake_shard(
    directory,
    domain: str,
    *,
    num_residues: int = 6,
    num_frames: int = 12,
    temperatures=("320", "348"),
    replicas=("0", "1"),
    seed: int = 0,
) -> str:
    """Write one shard with the fields the adapter actually reads.

    Geometry comes from :func:`synthetic_batch`, so CA-CA distances are physical
    and the PBC check passes; frames differ by a small deterministic drift, which
    keeps every frame a plausible protein while making the frames distinguishable.
    """
    batch = synthetic_batch([SyntheticSpec(num_residues)], seed=seed,
                            include_hydrogens=True, plm_dim=32)
    positions = batch.atoms.positions.numpy().astype(np.float64)
    a2r = batch.atoms.atom_to_residue.numpy()
    z = batch.atoms.atomic_number.numpy()
    names = [rc.ATOM_NAMES[int(i)] for i in batch.atoms.atom_name_id]
    resnames = [rc.RESIDUE_TYPES[int(t)] for t in batch.residues.residue_type]

    n_atom = len(positions)
    rng = np.random.default_rng(seed)
    drift = rng.normal(scale=0.02, size=positions.shape)

    pdb_lines = []
    for i, name in enumerate(names):
        res = int(a2r[i])
        pdb_lines.append(
            f"ATOM  {i + 1:5d} {name:<4s}{resnames[res]:>3s} A{res + 1:4d}    "
            f"{positions[i, 0]:8.3f}{positions[i, 1]:8.3f}{positions[i, 2]:8.3f}"
        )

    path = os.path.join(str(directory), f"mdcath_dataset_{domain}.h5")
    with h5py.File(path, "w") as handle:
        group = handle.create_group(domain)
        group.attrs["numResidues"] = num_residues
        group.attrs["numProteinAtoms"] = n_atom
        group.attrs["numChains"] = 1
        group.create_dataset("resid", data=(a2r + 1).astype(np.int64))
        group.create_dataset(
            "element", data=np.array([_Z_TO_ELEMENT[int(v)] for v in z], dtype="S2")
        )
        group.create_dataset(
            "resname", data=np.array([resnames[int(r)] for r in a2r], dtype="S4")
        )
        group.create_dataset("chain", data=np.array(["A"] * n_atom, dtype="S1"))
        group.create_dataset("z", data=z.astype(np.int64))
        group.create_dataset("pdbProteinAtoms", data="\n".join(pdb_lines))
        for temp in temperatures:
            tgroup = group.create_group(temp)
            for rep in replicas:
                rgroup = tgroup.create_group(rep)
                rgroup.attrs["numFrames"] = num_frames
                coords = np.stack(
                    [positions + drift * t for t in range(num_frames)]
                )
                forces = np.stack(
                    [rng.normal(scale=20.0, size=positions.shape) for _ in range(num_frames)]
                )
                rgroup.create_dataset("coords", data=coords.astype(np.float32))
                rgroup.create_dataset("forces", data=forces.astype(np.float32))
    return path


def fake_mdcath_config(directory, **overrides) -> MdCathConfig:
    base = dict(
        data_dir=str(directory),
        esm2_cache_dir=None,
        allow_fake_plm=True,
        plm_dim=32,
        ps_per_frame=1000.0,
        max_residues=None,
    )
    base.update(overrides)
    return MdCathConfig(**base)


@pytest.fixture
def shard_dir(tmp_path):
    write_fake_shard(tmp_path, "aaaaA00", num_frames=12, seed=0)
    write_fake_shard(tmp_path, "bbbbA00", num_frames=12, seed=1)
    return tmp_path


def build(config: LagPairConfig, domains) -> LagPairManifest:
    return build_lag_pair_manifest(config, domains, split="test")


# --------------------------------------------------------------------------
# lag arithmetic
# --------------------------------------------------------------------------


def test_lag_in_frames_is_exact():
    assert exact_lag_frames(1000.0, 1000.0) == 1
    assert exact_lag_frames(4000.0, 1000.0) == 4
    assert exact_lag_frames(16000.0, 1000.0) == 16


@pytest.mark.parametrize("lag_ps", [1500.0, 999.0, 1.0])
def test_non_integer_lag_is_refused_not_rounded(lag_ps):
    with pytest.raises(ValueError, match="not an integer"):
        exact_lag_frames(lag_ps, 1000.0)


def test_config_requires_ps_per_frame(shard_dir):
    with pytest.raises(ValueError, match="ps_per_frame"):
        LagPairConfig(mdcath=fake_mdcath_config(shard_dir, ps_per_frame=None))


def test_config_rejects_unsupported_history_length(shard_dir):
    with pytest.raises(ValueError, match="history_length"):
        LagPairConfig(mdcath=fake_mdcath_config(shard_dir), history_length=3)


def test_pairs_are_separated_by_exactly_the_lag(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    manifest = build(config, ["aaaaA00", "bbbbA00"])
    assert len(manifest) > 0
    by_lag = {1000.0: 1, 4000.0: 4}
    for pair in manifest.pairs:
        assert pair.future_frame - pair.current_frame == by_lag[pair.lag_ps]
        assert pair.lag_frames == by_lag[pair.lag_ps]
        assert pair.history_frames == (pair.current_frame - 1,)
        assert pair.current_frame - 1 >= 0


def test_phase1_subsample_cannot_supply_a_one_nanosecond_lag(tmp_path):
    """Two adjacent Phase 1 index entries are ~12-13 ns apart, not 1 ns."""
    write_fake_shard(tmp_path, "ccccA00", num_frames=500, seed=2,
                     temperatures=("320",), replicas=("0",))
    phase1 = MdCathDataset(
        fake_mdcath_config(tmp_path, frames_per_trajectory=40)
    )
    frames = [f for _, _, _, f in phase1.index]
    phase1.close()
    gaps = np.diff(sorted(frames))
    assert gaps.min() >= 12, (
        "Phase 1's evenly spaced index is the wrong source for a 1 ns lag; "
        f"its smallest gap here is {gaps.min()} frames"
    )

    config = LagPairConfig(
        mdcath=fake_mdcath_config(tmp_path), lags_ps=(1000.0,)
    )
    manifest = build(config, ["ccccA00"])
    assert {p.future_frame - p.current_frame for p in manifest.pairs} == {1}


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------


def test_no_pair_crosses_a_trajectory_boundary(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    manifest = build(config, ["aaaaA00", "bbbbA00"])
    n_frames = 12
    for pair in manifest.pairs:
        assert 0 <= min(pair.all_frames)
        assert max(pair.all_frames) <= n_frames - 1
    # every pair names exactly one (domain, temperature, replica)
    assert len(manifest.trajectories) == 2 * 2 * 2
    assert {p.temperature for p in manifest.pairs} == {"320", "348"}
    assert {p.replica for p in manifest.pairs} == {"0", "1"}


def test_last_admissible_start_is_the_last_frame_minus_the_lag(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), lags_ps=(4000.0,))
    manifest = build(config, ["aaaaA00"])
    starts = sorted(p.current_frame for p in manifest.pairs
                    if p.temperature == "320" and p.replica == "0")
    assert starts[0] == 1              # history needs t-1 >= 0
    assert starts[-1] == 12 - 1 - 4    # future must be a stored frame
    assert max(p.future_frame for p in manifest.pairs) == 11


@pytest.mark.parametrize("num_frames,expected", [(4, 0), (6, 1), (7, 2)])
def test_short_trajectories_yield_no_impossible_pairs(tmp_path, num_frames, expected):
    write_fake_shard(tmp_path, "ddddA00", num_frames=num_frames,
                     temperatures=("320",), replicas=("0",))
    config = LagPairConfig(mdcath=fake_mdcath_config(tmp_path), lags_ps=(4000.0,))
    manifest = build(config, ["ddddA00"])
    assert len(manifest) == expected


def test_history_length_one_asks_for_no_past_frame(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), history_length=1)
    manifest = build(config, ["aaaaA00"])
    assert all(p.history_frames == () for p in manifest.pairs)
    assert min(p.current_frame for p in manifest.pairs) == 0


# --------------------------------------------------------------------------
# quarantine
# --------------------------------------------------------------------------


def _write_json(path, payload) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return str(path)


def test_coordinate_quarantined_frames_are_never_touched(tmp_path):
    write_fake_shard(tmp_path, "eeeeA00", num_frames=12,
                     temperatures=("320",), replicas=("0",))
    bad = [3, 4, 11]
    path = _write_json(tmp_path / "coord.json",
                       {"quarantine": {"eeeeA00": {"320/0": bad}}})
    config = LagPairConfig(
        mdcath=fake_mdcath_config(tmp_path, coord_quarantine_path=path)
    )
    manifest = build(config, ["eeeeA00"])
    assert len(manifest) > 0
    for pair in manifest.pairs:
        assert not set(pair.all_frames) & set(bad), pair.pair_id


def test_force_quarantined_trajectory_is_excluded_by_default(tmp_path):
    write_fake_shard(tmp_path, "ffffA00", num_frames=12,
                     temperatures=("320",), replicas=("0", "1"))
    path = _write_json(tmp_path / "force.json",
                       {"quarantine": {"ffffA00": ["320/0"]}})
    base = fake_mdcath_config(tmp_path, quarantine_path=path)

    fair = build(LagPairConfig(mdcath=base), ["ffffA00"])
    assert {p.replica for p in fair.pairs} == {"1"}
    assert fair.metadata["trajectories_skipped_for_force_quarantine"] == 1

    production = build(
        LagPairConfig(mdcath=base, require_current_force_labels=False), ["ffffA00"]
    )
    assert {p.replica for p in production.pairs} == {"0", "1"}


def test_a_configured_quarantine_file_that_is_missing_fails_closed(tmp_path):
    write_fake_shard(tmp_path, "ggggA00", num_frames=12)
    config = LagPairConfig(
        mdcath=fake_mdcath_config(
            tmp_path, coord_quarantine_path=str(tmp_path / "absent.json")
        )
    )
    with pytest.raises(FileNotFoundError):
        build(config, ["ggggA00"])


# --------------------------------------------------------------------------
# determinism and the manifest file
# --------------------------------------------------------------------------


def test_manifest_is_deterministic_for_one_config(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    a = build(config, ["aaaaA00", "bbbbA00"])
    b = build(config, ["aaaaA00", "bbbbA00"])
    assert a.content_hash() == b.content_hash()
    assert [p.pair_id for p in a.pairs] == [p.pair_id for p in b.pairs]


def test_random_selection_depends_on_the_seed(shard_dir):
    kw = dict(mdcath=fake_mdcath_config(shard_dir), pairs_per_trajectory=3,
              selection="random")
    a = build(LagPairConfig(seed=0, **kw), ["aaaaA00"])
    b = build(LagPairConfig(seed=0, **kw), ["aaaaA00"])
    c = build(LagPairConfig(seed=1, **kw), ["aaaaA00"])
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_even_selection_caps_pairs_per_trajectory(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir),
                           pairs_per_trajectory=2, lags_ps=(1000.0,))
    manifest = build(config, ["aaaaA00"])
    per_traj: dict[tuple, int] = {}
    for pair in manifest.pairs:
        key = (pair.domain, pair.temperature, pair.replica)
        per_traj[key] = per_traj.get(key, 0) + 1
    assert set(per_traj.values()) == {2}


def test_max_pairs_keeps_every_domain(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), max_pairs=6)
    manifest = build(config, ["aaaaA00", "bbbbA00"])
    assert len(manifest) <= 6
    assert set(manifest.domains) == {"aaaaA00", "bbbbA00"}


def test_manifest_round_trips_through_json(shard_dir, tmp_path):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    manifest = build(config, ["aaaaA00"])
    path = str(tmp_path / "manifest.json")
    digest = manifest.save(path)
    reloaded = LagPairManifest.load(path)
    assert reloaded.content_hash() == digest == manifest.content_hash()
    assert [p.pair_id for p in reloaded.pairs] == [p.pair_id for p in manifest.pairs]
    assert reloaded.metadata["lag_frames"] == [1, 4]
    assert reloaded.metadata["ps_per_frame"] == 1000.0
    assert reloaded.metadata["seed"] == 0
    assert reloaded.metadata["quarantine"]["force_path"] is None


def test_edited_manifest_is_rejected(shard_dir, tmp_path):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    path = str(tmp_path / "manifest.json")
    build(config, ["aaaaA00"]).save(path)
    payload = json.load(open(path))
    payload["pairs"][0][3] += 1
    json.dump(payload, open(path, "w"))
    with pytest.raises(ValueError, match="content hash mismatch"):
        LagPairManifest.load(path)


def test_pair_id_names_every_coordinate_of_the_transition(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), lags_ps=(4000.0,))
    pair = build(config, ["aaaaA00"]).pairs[0]
    assert pair.pair_id == f"aaaaA00/320/0/t{pair.current_frame}/f{pair.future_frame}/lag4000ps"


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_splits_do_not_share_a_domain(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    train = build_lag_pair_manifest(config, ["aaaaA00"], split="train")
    val = build_lag_pair_manifest(config, ["bbbbA00"], split="val")
    train.assert_disjoint(val)
    assert not set(train.domains) & set(val.domains)


def test_a_shared_domain_between_splits_is_an_error(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    train = build_lag_pair_manifest(config, ["aaaaA00", "bbbbA00"], split="train")
    val = build_lag_pair_manifest(config, ["bbbbA00"], split="val")
    with pytest.raises(ValueError, match="appear in both"):
        train.assert_disjoint(val)


def test_split_restoration_is_by_domain_and_reproducible(shard_dir):
    config = fake_mdcath_config(shard_dir)
    train, val = restore_phase1_split(config)
    assert set(train) | set(val) == {"aaaaA00", "bbbbA00"}
    assert not set(train) & set(val)
    assert (train, val) == restore_phase1_split(config)


def test_split_restoration_refuses_a_snapshot_it_cannot_reproduce(shard_dir, tmp_path):
    snapshot = str(tmp_path / "config_snapshot.json")
    json.dump({"split": {"train_domains": ["aaaaA00", "bbbbA00"], "val_domains": []}},
              open(snapshot, "w"))
    with pytest.raises(ValueError, match="does not match"):
        restore_phase1_split(fake_mdcath_config(shard_dir), snapshot_path=snapshot)


# --------------------------------------------------------------------------
# the dataset itself
# --------------------------------------------------------------------------


def test_dataset_item_aligns_history_current_and_future(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    manifest = build(config, ["aaaaA00"])
    dataset = LagPairDataset(config, manifest)
    try:
        example = dataset[0]
        state = example.current.batch
        state.validate()
        assert len(example.history) == 1
        example.history[0].matches(state)
        example.future.matches(state)
        assert int(state.frame_index[0]) == example.pair.current_frame
        assert int(example.history[0].frame_index[0]) == example.pair.current_frame - 1
        assert int(example.future.frame_index[0]) == example.pair.future_frame
        # different frames, same topology
        assert not torch.allclose(example.future.positions, state.atoms.positions)
        assert example.future.positions.shape == state.atoms.positions.shape
    finally:
        dataset.close()


def test_future_is_a_label_not_a_state(shard_dir):
    """The future frame must not be constructible as an encoder input."""
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    dataset = LagPairDataset(config, build(config, ["aaaaA00"]))
    try:
        example = dataset[0]
        assert isinstance(example.future, FrameGeometry)
        assert not isinstance(example.future, HierarchicalProteinBatch)
        assert not hasattr(example.future, "forces")
        assert not hasattr(example.future, "atoms")
    finally:
        dataset.close()


def test_collated_batch_keeps_the_ragged_contract(shard_dir):
    config = LagPairConfig(mdcath=fake_mdcath_config(shard_dir))
    manifest = build(config, ["aaaaA00", "bbbbA00"])
    dataset = LagPairDataset(config, manifest)
    try:
        picks = [0, len(dataset) // 2, len(dataset) - 1]
        batch = collate_lag_pairs([dataset[i] for i in picks])
        batch.validate()
        assert batch.num_graphs == 3
        assert batch.future.positions.shape == batch.current.atoms.positions.shape
        assert batch.lag_frames.tolist() == [p.lag_frames for p in batch.pairs]
        assert torch.equal(
            batch.future.frame_index,
            batch.current.frame_index + batch.lag_frames,
        )
    finally:
        dataset.close()


def test_dataset_refuses_a_manifest_built_with_another_history_length(shard_dir):
    built = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), history_length=2)
    manifest = build(built, ["aaaaA00"])
    other = LagPairConfig(mdcath=fake_mdcath_config(shard_dir), history_length=1)
    with pytest.raises(ValueError, match="history_length"):
        LagPairDataset(other, manifest)


# --------------------------------------------------------------------------
# real shards
# --------------------------------------------------------------------------


@pytest.mark.mdcath
@pytest.mark.skipif(not HAS_DATA, reason="no mdCATH shards in data/")
def test_real_pairs_are_one_and_four_nanoseconds():
    domains = [
        os.path.basename(p)[len("mdcath_dataset_") : -len(".h5")]
        for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))[:2]
    ]
    config = LagPairConfig(
        mdcath=MdCathConfig(
            data_dir=DATA_DIR,
            esm2_cache_dir=os.path.join(ROOT, "esm2_cache"),
            quarantine_path=os.path.join(ROOT, "mdcath_force_quarantine.json"),
            coord_quarantine_path=os.path.join(ROOT, "mdcath_coord_quarantine.json"),
            temperatures=(320,), replicas=(0,), ps_per_frame=1000.0,
        ),
        pairs_per_trajectory=4,
    )
    manifest = build_lag_pair_manifest(config, domains, split="smoke")
    assert manifest.counts_by_lag() == {1000.0: 8, 4000.0: 8}
    for pair in manifest.pairs:
        assert pair.future_frame - pair.current_frame == int(pair.lag_ps // 1000)


@pytest.mark.mdcath
@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "runs/phase1_full/config_snapshot.json")),
    reason="no Phase 1 run snapshot",
)
def test_phase1_split_is_reproduced_exactly():
    config = MdCathConfig(data_dir=DATA_DIR, max_residues=250, ps_per_frame=1000.0)
    train, val = restore_phase1_split(
        config, snapshot_path=os.path.join(ROOT, "runs/phase1_full/config_snapshot.json")
    )
    assert (len(train), len(val)) == (726, 181)
    assert not set(train) & set(val)
