"""Topology checks against real mdCATH shards.

Skipped when no shard is present, so the offline suite stays runnable. These are
the tests that stop a plausible-looking heuristic from being wrong on real data:
the distance-based bond builder is checked against the PSF bond list, which is
ground truth from the force field that generated the trajectories.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from force_md.data import residue_constants as rc  # noqa: E402
from force_md.data.psf import parse_psf_bonds  # noqa: E402
from force_md.graph import build_covalent_bonds  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SHARDS = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))[:3]

pytestmark = [
    pytest.mark.mdcath,
    pytest.mark.skipif(not SHARDS, reason="no mdCATH shards in data/"),
]


def _load(path):
    with h5py.File(path, "r") as f:
        dom = list(f.keys())[0]
        g = f[dom]
        n_atom = int(g.attrs["numProteinAtoms"])
        z = torch.tensor(g["z"][:], dtype=torch.int64)
        resid = np.asarray(g["resid"][:])
        names = [
            line[12:16].strip()
            for line in g["pdbProteinAtoms"][()].decode().split("\n")
            if line.startswith("ATOM")
        ]
        psf = g["psf"][()].decode()
        temp = sorted(k for k in g.keys() if k.isdigit())[0]
        rep = sorted(g[temp].keys())[0]
        x = torch.tensor(g[temp][rep]["coords"][0], dtype=torch.float64)
    _, first = np.unique(resid, return_index=True)
    a2r = torch.tensor(np.searchsorted(np.unique(resid), resid), dtype=torch.int64)
    return dom, n_atom, z, a2r, names, psf, x


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_psf_bonds_parse_and_cover_every_atom(path):
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    bonds = parse_psf_bonds(psf, n_atom)
    assert bonds.shape[0] == 2 and bonds.shape[1] > 0
    assert int(bonds.max()) < n_atom and int(bonds.min()) >= 0
    # every protein atom participates in at least one bond
    touched = torch.zeros(n_atom, dtype=torch.bool)
    touched[bonds.reshape(-1)] = True
    assert bool(touched.all()), f"{dom}: {int((~touched).sum())} atoms have no bond"


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_psf_bond_lengths_are_physical(path):
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    bonds = parse_psf_bonds(psf, n_atom)
    d = torch.linalg.norm(x[bonds[0]] - x[bonds[1]], dim=-1)
    assert float(d.min()) > 0.8, f"{dom}: bond shorter than 0.8 A"
    assert float(d.max()) < 2.0, f"{dom}: bond longer than 2.0 A"


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_distance_heuristic_reproduces_the_psf_bond_graph(path):
    """The fallback must agree with the force field's own topology."""
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    psf_bonds = parse_psf_bonds(psf, n_atom)
    truth = {(min(a, b), max(a, b)) for a, b in psf_bonds.t().tolist()}

    e = build_covalent_bonds(x, z, a2r, torch.zeros(n_atom, dtype=torch.int64))
    got = {(min(a, b), max(a, b)) for a, b in zip(e.src.tolist(), e.dst.tolist())}

    missing = truth - got
    extra = got - truth
    assert not missing, f"{dom}: heuristic missed {len(missing)} real bonds"
    assert len(extra) / len(truth) < 0.02, (
        f"{dom}: heuristic invented {len(extra)} bonds ({100*len(extra)/len(truth):.1f}%)"
    )


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_every_residue_has_a_frame_and_ca_is_not_cay(path):
    """CHARMM caps must not be mistaken for backbone atoms."""
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    n_res = int(a2r.max()) + 1
    for r in range(n_res):
        sel = [i for i in range(n_atom) if int(a2r[i]) == r]
        present = {names[i] for i in sel}
        assert {"N", "CA", "C"} <= present, f"{dom}: residue {r} lacks frame atoms"
    caps = {nm for nm in names if rc.is_cap_atom(nm)}
    assert caps, f"{dom}: expected CHARMM terminal caps"
    assert not any(rc.is_backbone_atom(nm) for nm in caps)


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_atoms_are_contiguous_per_residue(path):
    """The contract requires non-decreasing atom_to_residue; check real data
    satisfies it so the adapter does not silently need to re-sort."""
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    assert bool((a2r[1:] >= a2r[:-1]).all()), f"{dom}: atoms not grouped by residue"


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_all_residue_names_are_in_the_vocabulary(path):
    dom, n_atom, z, a2r, names, psf, x = _load(path)
    with h5py.File(path, "r") as f:
        g = f[list(f.keys())[0]]
        resnames = {r.decode() for r in g["resname"][:]}
    unknown = {r for r in resnames if rc.canonical_resname(r) == "UNK"}
    assert not unknown, f"{dom}: residue names not in the CHARMM vocabulary: {unknown}"


# --------------------------------------------------------------------------
# force projection on real labels
# --------------------------------------------------------------------------


def _load_frame(path, replica="4"):
    """Replica 4 is the one never affected by the upstream forces==coords defect."""
    with h5py.File(path, "r") as f:
        dom = list(f.keys())[0]
        g = f[dom]
        el = np.array([e.decode() for e in g["element"][:]])
        resid = np.asarray(g["resid"][:])
        a2r = torch.tensor(np.searchsorted(np.unique(resid), resid), dtype=torch.int64)
        temp = sorted(k for k in g.keys() if k.isdigit())[0]
        x = torch.tensor(g[temp][replica]["coords"][0], dtype=torch.float64)
        fo = torch.tensor(g[temp][replica]["forces"][0], dtype=torch.float64)
    return dom, el, a2r, x, fo


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_real_forces_are_not_a_copy_of_coordinates(path):
    dom, el, a2r, x, fo = _load_frame(path)
    assert not torch.equal(x, fo), f"{dom}: forces are a copy of coords"


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_hydrogen_residual_is_substantial_and_identifiable(path):
    """The omitted-atom residual is a real target on this dataset.

    mdCATH stores hydrogens, so heavy-atom and all-atom residue forces can both
    be built and their difference measured. Across 12 audited domains the
    residual magnitude is 0.58-0.71 of the heavy-only residue force -- far too
    large to fold into uncertainty, which is why residue_hidden_force is enabled
    here rather than disabled as an unidentifiable term.
    """
    from force_md.nn.irreps import scatter_sum

    dom, el, a2r, x, fo = _load_frame(path)
    n_res = int(a2r.max()) + 1
    heavy = torch.tensor(el != "H")
    f_heavy = scatter_sum(fo * heavy.unsqueeze(-1).double(), a2r, n_res)
    f_all = scatter_sum(fo, a2r, n_res)
    residual = f_all - f_heavy

    mag = lambda v: float(v.norm(dim=-1).mean())  # noqa: E731
    ratio = mag(residual) / mag(f_heavy)
    assert 0.3 < ratio < 1.2, f"{dom}: residual/heavy ratio {ratio:.2f} out of range"
    # and it is exactly the summed force on the hydrogens
    expected = scatter_sum(fo * (~heavy).unsqueeze(-1).double(), a2r, n_res)
    assert torch.allclose(residual, expected, atol=1e-9)


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: os.path.basename(p)[:-3])
def test_protein_net_force_is_not_zero(path):
    """The protein is an open subsystem: solvent forces act on it, so the total
    force does not vanish. A model that assumed a closed system would be wrong."""
    dom, el, a2r, x, fo = _load_frame(path)
    net = float(fo.sum(0).norm())
    per_atom = float(fo.norm(dim=-1).mean())
    assert net > 1e-3, f"{dom}: net force vanished, unexpected for an open subsystem"
    assert net < per_atom * fo.shape[0], f"{dom}: net force implausibly large"
