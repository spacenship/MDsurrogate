"""The mdCATH adapter, against real shards.

Skipped when no shard is present. These check the things that would otherwise
only surface as a quietly wrong training run: quarantined force labels reaching
the loss, a domain appearing in both splits, hydrogens leaking into a heavy-atom
representation, or the residue frame being built from a CHARMM cap atom.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from force_md.data import collate_batches  # noqa: E402
from force_md.data.adapters import MdCathConfig, MdCathDataset, split_domains  # noqa: E402
from force_md.geometry import frames_from_batch  # noqa: E402
from force_md.graph import build_hierarchical_graph  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "esm2_cache")
QUARANTINE = os.path.join(ROOT, "mdcath_force_quarantine.json")
COORD_QUARANTINE = os.path.join(ROOT, "mdcath_coord_quarantine.json")
HAS_DATA = bool(glob.glob(os.path.join(DATA_DIR, "*.h5")))

pytestmark = [
    pytest.mark.mdcath,
    pytest.mark.skipif(not HAS_DATA, reason="no mdCATH shards in data/"),
]


def make_config(**overrides) -> MdCathConfig:
    base = dict(
        data_dir=DATA_DIR,
        esm2_cache_dir=CACHE_DIR if os.path.isdir(CACHE_DIR) else None,
        quarantine_path=QUARANTINE if os.path.exists(QUARANTINE) else None,
        max_domains=3,
        frames_per_trajectory=2,
        temperatures=(320,),
        replicas=(0, 4),
        allow_fake_plm=not os.path.isdir(CACHE_DIR),
    )
    base.update(overrides)
    return MdCathConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    ds = MdCathDataset(make_config())
    yield ds
    ds.close()


def test_dataset_builds_an_index(dataset):
    assert len(dataset) > 0
    assert len(dataset.domains) <= 3
    domain, temp, rep, frame = dataset.index[0]
    assert temp == "320" and rep in ("0", "4") and frame >= 0


def test_item_satisfies_the_contract(dataset):
    example = dataset[0]
    example.batch.validate()
    b = example.batch
    assert b.num_graphs == 1
    assert b.atoms.positions.shape == (b.num_atoms, 3)
    assert b.atoms.forces.shape == (b.num_atoms, 3)
    assert b.residues.plm_embedding.shape[0] == b.num_residues
    assert b.units.length == "angstrom"
    assert b.units.force == "kcal/mol/angstrom"


def test_heavy_atom_scope_contains_no_hydrogen(dataset):
    b = dataset[0].batch
    assert bool(b.atoms.is_heavy.all()), "hydrogens leaked into the heavy-atom scope"
    assert not bool((b.atoms.atomic_number == 1).any())


def test_all_atom_scope_contains_hydrogen():
    ds = MdCathDataset(make_config(represented_scope="all_atom"))
    try:
        b = ds[0].batch
        assert int((~b.atoms.is_heavy).sum()) > 0
        assert ds[0].hidden_force_target is None, (
            "nothing is omitted in all-atom mode, so there is no residual"
        )
    finally:
        ds.close()


def test_hidden_force_target_is_the_hydrogen_sum(dataset):
    example = dataset[0]
    assert example.hidden_force_target is not None
    assert example.hidden_force_target.shape == (example.batch.num_residues, 3)
    magnitude = float(example.hidden_force_target.norm(dim=-1).mean())
    assert 5.0 < magnitude < 200.0, f"implausible residual magnitude {magnitude}"


def test_residue_frames_are_valid_and_use_ca_not_cay(dataset):
    from force_md.data import residue_constants as rc

    b = dataset[0].batch
    frames = frames_from_batch(b)
    assert float(frames.valid.float().mean()) > 0.95
    # CA is a real atom of the residue, and CAY is present but is not it
    names = {rc.ATOM_NAMES[int(i)] for i in b.atoms.atom_name_id}
    assert "CA" in names
    if "CAY" in names:
        ca_rows = b.atoms.atom_name_id == rc.atom_name_id("CA")
        cay_rows = b.atoms.atom_name_id == rc.atom_name_id("CAY")
        assert not bool((ca_rows & cay_rows).any())


def test_ca_ca_distances_are_physical(dataset):
    """Also the PBC check: a broken chain would show up as a huge CA-CA gap."""
    b = dataset[0].batch
    ca = b.backbone.ca_positions[b.backbone.frame_valid]
    d = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1)
    within_chain = d[d < 5.0]
    assert float(within_chain.mean()) == pytest.approx(3.83, abs=0.15)


def test_graph_builds_on_a_real_protein(dataset):
    b = dataset[0].batch
    graph = build_hierarchical_graph(b)
    graph.validate(b)
    counts = graph.edge_counts()
    assert counts["residue__contains__atom"] == b.num_atoms
    assert counts["backbone__owns__residue"] == b.num_residues
    assert counts["atom__spatial__atom"] > counts["atom__bonded__atom"]
    per_atom = counts["atom__spatial__atom"] / b.num_atoms
    assert 10.0 < per_atom < 40.0, f"{per_atom:.1f} neighbours/atom at 5.0 A"


def test_quarantined_trajectories_are_marked_invalid():
    """The five known-corrupt domains must never contribute a force label."""
    if not os.path.exists(QUARANTINE):
        pytest.skip("no quarantine file")
    payload = json.load(open(QUARANTINE))
    quarantine = payload.get("quarantine", {})
    if not quarantine:
        pytest.skip("quarantine list is empty")

    domain = sorted(quarantine)[0]
    bad_key = sorted(quarantine[domain])[0]
    temp, rep = bad_key.split("/")
    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(int(temp),), replicas=(int(rep),)),
        domains=[domain],
    )
    try:
        assert len(ds) > 0
        example = ds[0]
        assert not bool(example.batch.atoms.force_valid.any()), (
            f"{domain} {bad_key} is quarantined but its forces are marked valid"
        )
    finally:
        ds.close()


def test_clean_trajectory_is_marked_valid():
    """Replica 4 is the one never affected by the upstream defect."""
    if not os.path.exists(QUARANTINE):
        pytest.skip("no quarantine file")
    quarantine = json.load(open(QUARANTINE)).get("quarantine", {})
    if not quarantine:
        pytest.skip("quarantine list is empty")
    domain = sorted(quarantine)[0]
    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(320,), replicas=(4,)),
        domains=[domain],
    )
    try:
        assert bool(ds[0].batch.atoms.force_valid.all())
    finally:
        ds.close()


def test_missing_plm_cache_raises_unless_opted_out():
    with pytest.raises(FileNotFoundError, match="precompute_esm2"):
        ds = MdCathDataset(
            make_config(esm2_cache_dir="/nonexistent", allow_fake_plm=False)
        )
        ds[0]


def test_fake_plm_must_be_opted_into():
    ds = MdCathDataset(make_config(esm2_cache_dir=None, allow_fake_plm=True, plm_dim=16))
    try:
        assert ds[0].batch.residues.plm_embedding.shape[1] == 16
    finally:
        ds.close()


def test_real_plm_embedding_matches_the_residue_count(dataset):
    if not os.path.isdir(CACHE_DIR):
        pytest.skip("no ESM-2 cache")
    b = dataset[0].batch
    assert b.residues.plm_embedding.shape == (b.num_residues, 1280)


def test_collating_real_proteins_validates(dataset):
    batches = [dataset[i].batch for i in range(min(3, len(dataset)))]
    merged = collate_batches(batches)
    merged.validate()
    assert merged.num_graphs == len(batches)
    assert merged.num_atoms == sum(b.num_atoms for b in batches)


def test_domain_split_has_no_leak(dataset):
    train, val = split_domains(dataset.domains, val_fraction=0.34, seed=0)
    assert not set(train) & set(val)
    assert train and val


def test_frames_are_spread_across_the_trajectory():
    ds = MdCathDataset(make_config(frames_per_trajectory=5, max_domains=1))
    try:
        frames = sorted({f for _, _, _, f in ds.index})
        assert len(frames) >= 4
        assert max(frames) - min(frames) > 100, (
            "frames must span the trajectory, not cluster at the start"
        )
    finally:
        ds.close()


def test_max_residues_filter_is_applied():
    ds = MdCathDataset(make_config(max_domains=None, max_residues=100))
    try:
        if len(ds) == 0:
            pytest.skip("no domain under the residue cap in this subset")
        assert ds[0].batch.num_residues <= 100
    finally:
        ds.close()


def test_saturated_coordinate_frames_are_excluded():
    """A quarantined frame must never reach the index.

    This is the defect that killed the first full launch: every atom saturated at
    INT_MAX/1000 nm. It cannot be caught downstream -- the value is finite, so it
    passes every NaN guard and only appears as a NaN after the first tensor
    product, by which point nothing names the frame responsible.
    """
    if not os.path.exists(COORD_QUARANTINE):
        pytest.skip("no coordinate quarantine file")
    quarantine = json.load(open(COORD_QUARANTINE)).get("quarantine", {})
    if not quarantine:
        pytest.skip("coordinate quarantine list is empty")

    domain = sorted(quarantine)[0]
    traj = sorted(quarantine[domain])[0]
    temp, rep = traj.split("/")
    corrupt = set(quarantine[domain][traj])

    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(int(temp),), replicas=(int(rep),),
                    frames_per_trajectory=40,
                    coord_quarantine_path=COORD_QUARANTINE),
        domains=[domain],
    )
    try:
        sampled = {frame for _, _, _, frame in ds.index}
        assert sampled, f"{domain}/{traj} produced no usable frames"
        assert not (sampled & corrupt), (
            f"{domain}/{traj}: sampled corrupt frames {sorted(sampled & corrupt)}"
        )
        # And the surviving frames must actually load, at sane magnitudes.
        peak = max(float(ds[i].batch.atoms.positions.abs().max())
                   for i in range(min(len(ds), 5)))
        assert peak < 1.0e4, f"{domain}/{traj}: |coordinate| reaches {peak:.6g} A"
    finally:
        ds.close()


def test_unquarantined_saturated_frame_raises_rather_than_returning_nan():
    """Without the quarantine the adapter must fail loudly, not silently."""
    if not os.path.exists(COORD_QUARANTINE):
        pytest.skip("no coordinate quarantine file")
    quarantine = json.load(open(COORD_QUARANTINE)).get("quarantine", {})
    if not quarantine:
        pytest.skip("coordinate quarantine list is empty")

    domain = sorted(quarantine)[0]
    traj = sorted(quarantine[domain])[0]
    temp, rep = traj.split("/")
    frame = sorted(quarantine[domain][traj])[0]

    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(int(temp),), replicas=(int(rep),),
                    frames_per_trajectory=2, coord_quarantine_path=None),
        domains=[domain],
    )
    try:
        ds.index = [(domain, temp, rep, frame)]
        with pytest.raises(ValueError, match="INT_MAX"):
            ds[0]
    finally:
        ds.close()


def test_collapsed_frame_is_quarantined():
    """A frame with every atom at one point must never be sampled.

    This one frame caused every non-finite step in the first four runs. It defeats
    a magnitude check completely -- all coordinates are exactly (0,0,0), so
    ``|coord| = 0`` passes any threshold, however tight. The invariant that
    catches it is spatial extent: a protein occupies space.

    It is also the *last* frame of its trajectory, and ``np.linspace(0, n-1, k)``
    always includes ``n-1``, so evenly spaced sampling drew it once per epoch with
    certainty -- which is exactly the rate the runs showed.
    """
    if not os.path.exists(COORD_QUARANTINE):
        pytest.skip("no coordinate quarantine file")
    payload = json.load(open(COORD_QUARANTINE))
    quarantine = payload.get("quarantine", {})
    collapsed = payload.get("kinds", {}).get("collapsed", 0)
    if not collapsed:
        pytest.skip("no collapsed frames recorded")

    assert "2qenA03" in quarantine, "the known collapsed frame is not quarantined"
    traj, frames = next(iter(quarantine["2qenA03"].items()))
    temp, rep = traj.split("/")

    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(int(temp),), replicas=(int(rep),),
                    frames_per_trajectory=40,
                    coord_quarantine_path=COORD_QUARANTINE),
        domains=["2qenA03"],
    )
    try:
        sampled = {f for _, _, _, f in ds.index}
        assert sampled, "quarantine removed the whole trajectory"
        assert not (sampled & set(frames)), (
            f"collapsed frames {sorted(sampled & set(frames))} still sampled"
        )
        # Every surviving frame must actually occupy space.
        for i in range(min(len(ds), 6)):
            x = ds[i].batch.atoms.positions
            extent = float((x - x.mean(0)).norm(dim=-1).max())
            assert extent > 1.0, f"frame {ds.index[i]} has extent {extent:.3e} A"
    finally:
        ds.close()


def test_collapsed_frame_raises_without_quarantine():
    """The runtime backstop must catch by extent, not only by magnitude.

    The collapsed frame has coordinates of exactly (0,0,0), so the magnitude
    backstop passes it however tight the threshold. Checking magnitude alone here
    -- while the audit checked extent -- would leave the two halves disagreeing
    about what is loadable.
    """
    if not os.path.exists(COORD_QUARANTINE):
        pytest.skip("no coordinate quarantine file")
    payload = json.load(open(COORD_QUARANTINE))
    if not payload.get("kinds", {}).get("collapsed"):
        pytest.skip("no collapsed frames recorded")
    traj, frames = next(iter(payload["quarantine"]["2qenA03"].items()))
    temp, rep = traj.split("/")

    ds = MdCathDataset(
        make_config(max_domains=None, temperatures=(int(temp),), replicas=(int(rep),),
                    frames_per_trajectory=2, coord_quarantine_path=None),
        domains=["2qenA03"],
    )
    try:
        ds.index = [("2qenA03", temp, rep, int(frames[0]))]
        with pytest.raises(ValueError, match="no spatial extent"):
            ds[0]
    finally:
        ds.close()


def test_missing_quarantine_file_refuses_to_run():
    """A configured-but-absent quarantine file must fail, not silently disable.

    Returning an empty dict here meant a fresh clone or a re-download trained on
    known-bad frames with nothing in the log to say the filter was inactive.
    ``None`` remains an explicit, honoured opt-out.
    """
    with pytest.raises(FileNotFoundError, match="quarantine file but does not exist"):
        MdCathDataset(make_config(coord_quarantine_path="/nonexistent/quarantine.json"))

    with pytest.raises(FileNotFoundError, match="quarantine file but does not exist"):
        MdCathDataset(make_config(quarantine_path="/nonexistent/forces.json"))

    ds = MdCathDataset(make_config(quarantine_path=None, coord_quarantine_path=None))
    try:
        assert len(ds) > 0, "explicit opt-out must still build an index"
    finally:
        ds.close()
