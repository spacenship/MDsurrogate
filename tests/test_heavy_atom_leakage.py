"""Nothing deployable may read the future. Enforced by perturbing it.

The heavy-atom stages sit next to three quantities that *do* legitimately read
``t + lag``: the supervision, the two H0 oracles, and the H2 GT-force arm. That
proximity is the risk. A feature builder that picked up one future field would
not crash, would not change any shape, and would improve every metric -- the
failure mode is a better-looking result, which is the one nothing else catches.

So the test is behavioural rather than structural: build the same batch against
two *different* futures and require the deployable outputs to be bit-identical.
A structural check ("this function does not import that module") would pass for
a leak routed through a shared object; this one cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from force_md.heavy.backmapping import (  # noqa: E402
    CONSTRUCTION_MODES,
    heavy_atom_placements,
)
from force_md.heavy.refiner import (  # noqa: E402
    current_peptide_geometry,
    reconstruct_from_frames,
    refiner_features,
)
from force_md.transition import build_transition_target  # noqa: E402
from test_heavy_atom_equivariance import a_prediction  # noqa: E402
from test_transition_targets import make_batch, perturbed_future  # noqa: E402

#: The modes whose construction is allowed to consult ``t + lag``. Read from the
#: module's own table rather than restated, so a mode added later is classified
#: by its declaration instead of by this test's memory of it.
ORACLE_MODES = frozenset(
    mode
    for mode, info in CONSTRUCTION_MODES.items()
    if info["uses_future_for_scoring_only"]
)


def two_futures(seed: int = 0):
    """One batch, two genuinely different futures."""
    batch = make_batch(seed=seed)
    first = perturbed_future(batch, seed=seed + 1, scale=0.3)
    second = perturbed_future(batch, seed=seed + 2, scale=1.4)
    moved = (first.ca_positions - second.ca_positions).norm(dim=-1).max()
    assert float(moved) > 0.5, "the two futures must actually differ"
    return batch, first, second


def test_refiner_features_ignore_the_future():
    batch, first, second = two_futures()
    prediction = a_prediction(batch.num_residues)
    lag = torch.full((batch.num_graphs,), 1.0, dtype=torch.float64)

    a = refiner_features(prediction, build_transition_target(batch, first), lag)
    b = refiner_features(prediction, build_transition_target(batch, second), lag)
    assert torch.equal(a, b), (
        "a refiner feature moved when only the future changed. H1b's input must "
        "be the current structure and the coarse prediction, nothing else."
    )


def test_the_peptide_geometry_reference_is_measured_at_t():
    """H1b regularises towards the *current* frame's own peptide geometry.

    That is a measurement of the model's input, so it is not leakage -- but only
    as long as it is actually the current structure being measured.
    """
    batch, first, second = two_futures()
    values = []
    for future in (first, second):
        target = build_transition_target(batch, future)
        n, ca, c = reconstruct_from_frames(
            target.current_ca,
            target.current_frames.rotation,
            target.local_n,
            target.local_c,
        )
        values.append(current_peptide_geometry(n, ca, c, target.following))
    for key in ("bond_c_n", "angle_ca_c_n", "angle_c_n_ca"):
        assert torch.equal(values[0][key], values[1][key]), (
            f"{key} changed with the future; it is supposed to be measured at t"
        )


def test_the_deployable_placement_ignores_the_future():
    batch, first, second = two_futures()
    prediction = a_prediction(batch.num_residues)

    placements = [
        heavy_atom_placements(
            prediction, build_transition_target(batch, future), batch, future
        )
        for future in (first, second)
    ]
    for mode in CONSTRUCTION_MODES:
        a, b = placements[0][mode], placements[1][mode]
        same = torch.equal(a.positions, b.positions)
        if mode in ORACLE_MODES:
            assert not same, (
                f"{mode} is declared an oracle but did not move when the future "
                "did, so it is not reading the future it claims to"
            )
        else:
            assert same, (
                f"{mode} is declared future-free but changed when only the "
                "future changed"
            )


def test_the_identity_placement_ignores_the_model_as_well():
    """The baseline must depend on neither the future nor the prediction.

    If a prediction reached it, the baseline would improve, every delta measured
    against it would shrink, and the report would quietly understate the model.
    """
    batch, future, _ = two_futures()
    target = build_transition_target(batch, future)
    first = heavy_atom_placements(
        a_prediction(batch.num_residues, seed=1), target, batch, future
    )["identity_current_atoms"]
    second = heavy_atom_placements(
        a_prediction(batch.num_residues, seed=99), target, batch, future
    )["identity_current_atoms"]
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.positions, batch.atoms.positions)
