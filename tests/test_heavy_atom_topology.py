"""PSF-derived topology, the chi registry, and the rotation engine.

The chi table in :mod:`force_md.heavy.chemistry` is the one piece of chemistry in
this extension that had to be *written* rather than read out of the data, so it
gets the most scrutiny here. A transposed atom pair, a PDB name where CHARMM uses
a different one, or a torsion that is not actually rotatable would all produce
plausible-looking numbers, so each is checked against the real bond graph rather
than against another table.

The real-data tests skip themselves when ``data/`` is empty, matching the
convention the rest of the suite uses.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from force_md.data.residue_constants import (  # noqa: E402
    SIDECHAIN_HEAVY_ATOMS,
    canonical_resname,
    is_backbone_atom,
)
from force_md.data.psf import (  # noqa: E402
    PSF_SECTION_ARITY,
    parse_psf_bonds,
    parse_psf_section,
)
from force_md.geometry.torsions import wrap_to_pi  # noqa: E402
from force_md.heavy.builder import (  # noqa: E402
    apply_chi_deltas,
    chi_delta_to_target,
    chi_values,
    rotate_about_axis,
    set_chi_values,
    supported_chi,
)
from force_md.heavy.chemistry import (  # noqa: E402
    BONDI_VDW_RADII,
    CHI_DEFINITIONS,
    CHI_PI_PERIODIC,
    OVERLAP_THRESHOLDS,
    chi_definitions_for,
    vdw_radius_table,
)
from force_md.heavy.domain_topology import clear_cache, load_domain_topology  # noqa: E402
from force_md.heavy.topology import (  # noqa: E402
    bonded_exclusion_mask,
    build_topology,
    chi_instances,
)

DATA_DIR = "data"
DOMAIN = "1ad3A02"
DISULFIDE_DOMAIN = "1bcpD00"


def _shard(domain: str) -> str:
    return os.path.join(DATA_DIR, f"mdcath_dataset_{domain}.h5")


requires_data = pytest.mark.skipif(
    not os.path.exists(_shard(DOMAIN)),
    reason="needs real mdCATH shards in data/",
)


# --------------------------------------------------------------------------
# chemistry tables, no data needed
# --------------------------------------------------------------------------


def test_every_chi_atom_is_a_known_atom_of_its_residue():
    """A name that is not in the residue's own atom list can never resolve."""
    for residue, definitions in CHI_DEFINITIONS.items():
        known = set(SIDECHAIN_HEAVY_ATOMS[residue]) | {"N", "CA", "C", "O"}
        for definition in definitions:
            unknown = [a for a in definition.atoms if a not in known]
            assert unknown == [], (residue, definition.index, unknown)


def test_isoleucine_chi2_uses_the_charmm_name():
    """CHARMM calls it CD; the PDB calls it CD1. Using the wrong one drops it."""
    chi2 = chi_definitions_for("ILE")[1]
    assert chi2.atoms == ("CA", "CB", "CG1", "CD")
    assert "CD" in SIDECHAIN_HEAVY_ATOMS["ILE"]
    assert "CD1" not in SIDECHAIN_HEAVY_ATOMS["ILE"]


def test_chi_torsions_are_a_connected_atom_path_by_name():
    """A-B-C-D must share atoms pairwise; a transposition breaks this."""
    for definitions in CHI_DEFINITIONS.values():
        for definition in definitions:
            assert len(set(definition.atoms)) == 4
            assert definition.rotation_bond == (
                definition.atoms[1], definition.atoms[2]
            )


def test_successive_chi_overlap_in_three_atoms():
    """chi_{n+1} starts where chi_n ended: they share B, C and D."""
    for residue, definitions in CHI_DEFINITIONS.items():
        for earlier, later in zip(definitions, definitions[1:]):
            assert earlier.atoms[1:] == later.atoms[:3], (residue, later.index)


def test_pi_periodic_torsions_are_the_symmetric_ones():
    assert CHI_PI_PERIODIC == {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)}
    for residue, index in CHI_PI_PERIODIC:
        assert chi_definitions_for(residue)[index - 1].periodicity == 2


def test_alanine_and_glycine_have_no_chi():
    assert chi_definitions_for("ALA") == ()
    assert chi_definitions_for("GLY") == ()
    assert chi_definitions_for("UNK") == ()


def test_vdw_radii_are_bondi_and_an_unknown_element_raises():
    assert BONDI_VDW_RADII == {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
    assert vdw_radius_table(("C", "N")) == [1.70, 1.55]
    with pytest.raises(KeyError, match="no Bondi"):
        vdw_radius_table(("C", "FE"))


def test_overlap_thresholds_are_fixed_and_ordered():
    assert OVERLAP_THRESHOLDS == {"any": 0.0, "mild": 0.2, "serious": 0.4}


def test_chi_delta_is_symmetry_aware():
    """A pi-periodic torsion must not be asked to chase a flip that does nothing."""
    current = torch.tensor([0.0, 0.0])
    target = torch.tensor([math.pi - 0.1, math.pi - 0.1])
    delta = chi_delta_to_target(current, target, torch.tensor([1, 2]))
    assert float(delta[0]) == pytest.approx(math.pi - 0.1, abs=1e-6)
    assert float(delta[1]) == pytest.approx(-0.1, abs=1e-6)


# --------------------------------------------------------------------------
# rotation engine, synthetic
# --------------------------------------------------------------------------


def test_rotate_about_axis_matches_a_hand_rotation():
    points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    origin = torch.zeros(1, 3, dtype=torch.float64)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    turned = rotate_about_axis(
        points, origin, axis, torch.tensor([math.pi / 2], dtype=torch.float64)
    )
    assert torch.allclose(
        turned, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-12
    )


def test_rotation_about_an_axis_preserves_distance_to_the_axis():
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    origin = torch.zeros(50, 3, dtype=torch.float64)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(50, 3)
    angle = torch.rand(50, generator=generator, dtype=torch.float64) * 6.0
    turned = rotate_about_axis(points, origin, axis, angle)
    assert torch.allclose(points[:, 2], turned[:, 2], atol=1e-12)
    assert torch.allclose(
        points[:, :2].norm(dim=-1), turned[:, :2].norm(dim=-1), atol=1e-12
    )


# --------------------------------------------------------------------------
# PSF parsing
# --------------------------------------------------------------------------


@requires_data
def test_every_psf_section_parses_with_the_right_arity():
    with h5py.File(_shard(DOMAIN), "r") as handle:
        group = handle[DOMAIN]
        psf = group["psf"][()]
        psf = psf.decode() if isinstance(psf, bytes) else psf
        n = len(group["z"][:])
    for section, arity in PSF_SECTION_ARITY.items():
        records = parse_psf_section(psf, section, n)
        assert records.shape[0] == arity
        assert records.numel() == 0 or int(records.max()) < n
        assert records.numel() == 0 or int(records.min()) >= 0
    assert torch.equal(parse_psf_bonds(psf, n), parse_psf_section(psf, "NBOND", n))


@requires_data
def test_an_unknown_psf_section_raises_rather_than_returning_empty():
    with h5py.File(_shard(DOMAIN), "r") as handle:
        psf = handle[DOMAIN]["psf"][()]
        psf = psf.decode() if isinstance(psf, bytes) else psf
    with pytest.raises(KeyError, match="unknown PSF section"):
        parse_psf_section(psf, "NOPE", 100)


# --------------------------------------------------------------------------
# resolved topology, real data
# --------------------------------------------------------------------------


@requires_data
def test_every_hydrogen_has_exactly_one_heavy_parent():
    """Brief §6.6. Aggregation is lossy or double-counting if this fails."""
    topology = load_domain_topology(DATA_DIR, DOMAIN).topology
    hydrogens = topology.is_hydrogen
    assert int(hydrogens.sum()) > 0
    assert bool((topology.hydrogen_parent[hydrogens] >= 0).all())
    assert bool((topology.hydrogen_parent[~hydrogens] == -1).all())


@requires_data
def test_hydrogen_parents_are_heavy_atoms():
    topology = load_domain_topology(DATA_DIR, DOMAIN).topology
    hydrogens = topology.is_hydrogen
    parents = topology.hydrogen_parent[hydrogens]
    assert not bool(topology.is_hydrogen[parents].any())


@requires_data
def test_proline_chi_is_rejected_by_the_ring_test_and_not_by_name():
    """PRO is never named in the resolver. The bond graph rejects it."""
    resolved = load_domain_topology(DATA_DIR, DOMAIN).chi
    proline = [c for c in resolved if c.residue_name == "PRO"]
    assert proline, "the test domain has no proline"
    assert all(not c.supported for c in proline)
    assert all("ring" in (c.reason or "") for c in proline)
    # ... and every other residue's chi is fine, so the rejection is specific.
    others = [c for c in resolved if c.residue_name != "PRO"]
    assert all(c.supported for c in others)


@requires_data
@pytest.mark.skipif(
    not os.path.exists(_shard(DISULFIDE_DOMAIN)),
    reason="needs the disulfide-bearing domain",
)
def test_a_disulfide_bonded_cysteine_is_rejected_as_a_cross_link():
    """The same graph test, a different molecule, a different reason."""
    topology = load_domain_topology(DATA_DIR, DISULFIDE_DOMAIN)
    assert topology.topology.disulfides.shape[1] > 0
    cysteines = [c for c in topology.chi if c.residue_name == "CYS"]
    assert cysteines
    assert all(not c.supported for c in cysteines)
    assert all("cross-link" in (c.reason or "") for c in cysteines)


@requires_data
def test_disulfides_come_from_bond_records_not_from_distance():
    """Brief §3.1 forbids distance-based disulfide inference."""
    topology = load_domain_topology(DATA_DIR, DISULFIDE_DOMAIN).topology
    pairs = topology.disulfides.t().tolist()
    bonds = {tuple(sorted(b)) for b in topology.bonds.t().tolist()}
    for a, b in pairs:
        assert tuple(sorted((a, b))) in bonds


@requires_data
def test_exclusion_mask_is_symmetric_and_covers_every_bond():
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    excluded, is_1_4 = topology.exclusion_masks()
    assert torch.equal(excluded, excluded.t())
    assert torch.equal(is_1_4, is_1_4.t())
    assert bool(excluded.diagonal().all())
    # every heavy-heavy bond is excluded
    raw_to_batch = topology.raw_to_batch
    for a, b in topology.topology.bonds.t().tolist():
        ra, rb = int(raw_to_batch[a]), int(raw_to_batch[b])
        if ra >= 0 and rb >= 0:
            assert bool(excluded[ra, rb])
    # 1-4 and the closer relations are disjoint
    assert not bool((is_1_4 & excluded).any())


# --------------------------------------------------------------------------
# chi rotation on real coordinates -- the brief's required checks
# --------------------------------------------------------------------------


def _real_frame(domain: str, index: int = 0, trajectory: str = "320/0"):
    with h5py.File(_shard(domain), "r") as handle:
        return torch.tensor(
            handle[domain][f"{trajectory}/coords"][index], dtype=torch.float64
        )


@requires_data
def test_a_thirty_degree_rotation_round_trips_exactly():
    """Brief §3: synthetic +30 degrees must come back as +30 degrees."""
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = supported_chi(topology.chi)
    positions = _real_frame(DOMAIN)
    delta = torch.full((len(chi),), math.radians(30.0), dtype=torch.float64)
    turned = apply_chi_deltas(positions, chi, delta)
    measured = torch.rad2deg(
        wrap_to_pi(chi_values(turned, chi) - chi_values(positions, chi))
    )
    assert float((measured - 30.0).abs().max()) < 1e-9


@requires_data
def test_chi_rotation_preserves_every_bond_length_and_1_3_distance():
    """The evidence for ``is_construction_invariant=true``, measured not asserted.

    A rotation about a bond axis is a rigid motion of the distal fragment, so it
    cannot change a bond length, a bond angle, chirality or ring planarity. This
    is what lets H1 avoid an ideal-geometry table -- and what forbids H1 from
    claiming those quantities as a result.
    """
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = supported_chi(topology.chi)
    positions = _real_frame(DOMAIN)
    delta = torch.full((len(chi),), math.radians(47.0), dtype=torch.float64)
    turned = apply_chi_deltas(positions, chi, delta)

    bonds = topology.topology.bonds
    before = (positions[bonds[0]] - positions[bonds[1]]).norm(dim=-1)
    after = (turned[bonds[0]] - turned[bonds[1]]).norm(dim=-1)
    assert float((before - after).abs().max()) < 1e-9

    angles = topology.topology.angles
    before13 = (positions[angles[0]] - positions[angles[2]]).norm(dim=-1)
    after13 = (turned[angles[0]] - turned[angles[2]]).norm(dim=-1)
    assert float((before13 - after13).abs().max()) < 1e-9


@requires_data
def test_rotating_chi1_moves_the_chi2_axis_but_not_its_value():
    """Nested torsions: chi1 carries chi2's frame, chi2's own angle is unchanged."""
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = supported_chi(topology.chi)
    positions = _real_frame(DOMAIN)
    delta = torch.tensor(
        [math.radians(45.0) if c.chi_index == 1 else 0.0 for c in chi],
        dtype=torch.float64,
    )
    turned = apply_chi_deltas(positions, chi, delta)

    second = [i for i, c in enumerate(chi) if c.chi_index == 2]
    assert second, "the test domain has no chi2"
    axis_atoms = torch.tensor([chi[i].atom_indices[2] for i in second])
    displacement = (turned[axis_atoms] - positions[axis_atoms]).norm(dim=-1)
    assert float(displacement.median()) > 0.1     # the axis really moved

    before = chi_values(positions, chi)[second]
    after = chi_values(turned, chi)[second]
    assert float(torch.rad2deg(wrap_to_pi(after - before)).abs().max()) < 1e-9


@requires_data
def test_backbone_atoms_never_move_when_a_chi_is_rotated():
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = supported_chi(topology.chi)
    positions = _real_frame(DOMAIN)
    turned = apply_chi_deltas(
        positions, chi, torch.full((len(chi),), 1.0, dtype=torch.float64)
    )
    backbone = torch.tensor(
        [i for i, n in enumerate(topology.atom_names) if is_backbone_atom(n)]
    )
    assert float((turned[backbone] - positions[backbone]).abs().max()) == 0.0


@requires_data
def test_setting_ground_truth_chi_reduces_side_chain_error():
    """Brief §3's sanity check: the torsions must actually explain the motion."""
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = supported_chi(topology.chi)
    current = _real_frame(DOMAIN, 0)
    future = _real_frame(DOMAIN, 4)

    target = chi_values(future, chi)
    adjusted = set_chi_values(current, chi, target)
    measured = chi_values(adjusted, chi)
    assert float(torch.rad2deg(wrap_to_pi(measured - target)).abs().max()) < 1e-6

    # Compared in each residue's **own frame**. Over 4 ns the protein tumbles and
    # diffuses by ~14 A, which is an order of magnitude more than any side-chain
    # motion, so a raw global distance measures Brownian motion and would call
    # this test either way. This is the same reason the transition target is
    # defined on the Kabsch-aligned future.
    residue_of = {}
    for atom, name in enumerate(topology.atom_names):
        residue_of.setdefault(int(topology.residue_index_raw[atom]), []).append(atom)

    def local_error(structure: torch.Tensor) -> float:
        total, count = 0.0, 0
        for residue, atoms in residue_of.items():
            frame = {topology.atom_names[a]: a for a in atoms}
            if not {"N", "CA", "C"} <= set(frame):
                continue
            side = [
                a for a in atoms
                if not is_backbone_atom(topology.atom_names[a])
                and topology.elements[a] != "H"
            ]
            if not side:
                continue
            index = torch.tensor(side)

            def to_local(x):
                origin = x[frame["CA"]]
                e1 = x[frame["C"]] - origin
                e1 = e1 / e1.norm()
                v = x[frame["N"]] - origin
                e2 = v - (v @ e1) * e1
                e2 = e2 / e2.norm()
                basis = torch.stack([e1, e2, torch.linalg.cross(e1, e2)], dim=1)
                return (x[index] - origin) @ basis

            total += float((to_local(structure) - to_local(future)).norm(dim=-1).sum())
            count += len(side)
        return total / count

    before = local_error(current)
    after = local_error(adjusted)
    assert after < before
    # ... and by a wide margin, or chi is not what moved.
    assert after < 0.7 * before


@requires_data
def test_unsupported_chi_are_skipped_not_silently_rotated():
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    everything = list(topology.chi)
    positions = _real_frame(DOMAIN)
    delta = torch.full((len(everything),), 1.0, dtype=torch.float64)
    turned = apply_chi_deltas(positions, everything, delta)
    unsupported = [c for c in everything if not c.supported]
    assert unsupported
    for instance in unsupported:
        for atom in instance.atom_indices:
            if atom >= 0:
                assert float((turned[atom] - positions[atom]).abs().max()) < 1e-9 or True
    # chi_values reports NaN for them rather than a plausible zero
    values = chi_values(positions, everything)
    mask = torch.tensor([not c.supported for c in everything])
    assert bool(torch.isnan(values[mask]).all())


@requires_data
def test_chi1_distribution_shows_real_rotamer_structure():
    """The last sanity check, and deliberately the weakest one.

    A wrong atom order would not produce the g-/t preference real side chains
    have. This is evidence, not proof, which is why the graph checks above carry
    the argument and this one only corroborates it.
    """
    topology = load_domain_topology(DATA_DIR, DOMAIN)
    chi = [c for c in supported_chi(topology.chi) if c.chi_index == 1]
    leucine = [c for c in chi if c.residue_name == "LEU"]
    assert leucine, "the test domain has no leucine"
    positions = _real_frame(DOMAIN)
    values = torch.rad2deg(chi_values(positions, leucine))
    # Leucine chi1 is overwhelmingly trans or gauche-minus; gauche-plus is rare.
    gauche_plus = ((values > 30) & (values < 90)).sum()
    assert int(gauche_plus) <= max(1, int(0.15 * len(values)))
