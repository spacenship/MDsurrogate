"""The Stage M1 evaluator's two invariants, end to end on a real probe (CPU).

The metric tests check arithmetic and the analysis tests check aggregation.
These two check the properties that would invalidate the whole re-evaluation if
they were false, and that only show up when a real model is in the loop:

* **the future never reaches the model.** The extended metrics read
  ``t + lag`` -- that is what a metric is for -- so the only defence against
  leaking it into the conditioning path is that no such path exists. Asserted
  against the actual signatures and by giving a conditioner a target and watching
  it fail.

* **no weight moves.** The point of Stage M is that the Stage B checkpoints are
  re-scored, not re-trained. Checked by hashing every parameter tensor before and
  after a full scoring pass and requiring bitwise equality.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from force_md.transition import (  # noqa: E402
    ExtendedMetricConfig,
    RecordContext,
    build_transition_target,
    extended_metric_records,
)
from force_md.transition.probe import TransitionProbe  # noqa: E402
from test_transition_probe import (  # noqa: E402
    displaced,
    extractor,  # noqa: F401 - a module-scoped pytest fixture
    make_probe,
)
from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402

PLM_DIM = 32


class FakePair:
    def __init__(self, index: int):
        self.domain = f"d{index}"
        self.temperature = "320"
        self.replica = "0"
        self.current_frame = 2
        self.future_frame = 3
        self.lag_ps = 1000.0
        self.pair_id = f"d{index}/320/0/t2/f3/lag1000ps"


CONTEXT = RecordContext(
    arm="structure_only", canonical_arm="S0_structure_history", oracle=False, seed=0,
    manifest_hash="m", phase1_checkpoint_hash="p", transition_checkpoint_hash="t",
    config_hash="c", git_commit="g", git_dirty=True, source_diff_hash="d",
)


@pytest.fixture
def batch_and_frames():
    batch = synthetic_batch(
        [SyntheticSpec(14), SyntheticSpec(11)], seed=0, plm_dim=PLM_DIM
    )
    history = displaced(batch, scale=0.2, seed=3, offset=-1)
    future = displaced(batch, scale=0.6, seed=4, offset=4)
    return batch, history, future


def _parameter_digest(module: torch.nn.Module) -> str:
    """One hash over every parameter and buffer, in a fixed order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(
        list(module.named_parameters()) + list(module.named_buffers())
    ):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# 27. the evaluation target never reaches the conditioner
# --------------------------------------------------------------------------


def test_the_probe_signature_cannot_accept_a_future_frame(extractor):  # noqa: F811
    """Structural, not behavioural: there is no parameter to pass one to."""
    parameters = set(inspect.signature(TransitionProbe.forward).parameters)
    assert parameters == {"self", "batch", "bundle", "history", "lag_ps"}
    assert "future" not in parameters and "target" not in parameters


def test_the_metric_path_takes_no_model(extractor, batch_and_frames):  # noqa: F811
    """The one place the future is read has nothing to give it to."""
    parameters = set(inspect.signature(extended_metric_records).parameters)
    assert parameters == {"prediction", "target", "pairs", "context", "config",
                          "identity"}
    assert not any("model" in p or "probe" in p for p in parameters)


def test_scoring_does_not_change_what_the_model_would_predict(
    extractor, batch_and_frames  # noqa: F811
):
    """The strongest available end-to-end statement about leakage.

    Score the model against the real future, then against a **wrecked** future,
    and require the prediction to be identical both times. If any future
    coordinate reached the conditioning path the two predictions would differ.
    """
    import dataclasses

    batch, history, future = batch_and_frames
    probe = make_probe(extractor).eval()

    with torch.no_grad():
        bundle = extractor(batch)
        first = probe(batch, bundle, history=[history], lag_ps=torch.tensor([1000.0, 1000.0]))
        # A *conformational* wreck, not a translation: a rigid shift of the
        # future is removed by the target's Kabsch step and would leave every
        # metric unchanged -- correctly, but then this test would prove nothing.
        g = torch.Generator().manual_seed(9)

        def noisy(x):
            return x + 4.0 * torch.randn(x.shape, generator=g, dtype=x.dtype)

        wrecked = dataclasses.replace(
            future,
            positions=noisy(future.positions),
            n_positions=noisy(future.n_positions),
            ca_positions=noisy(future.ca_positions),
            c_positions=noisy(future.c_positions),
        )
        target_a = build_transition_target(batch, future)
        target_b = build_transition_target(batch, wrecked)
        rows_a = extended_metric_records(
            first, target_a, pairs=[FakePair(0), FakePair(1)], context=CONTEXT,
            config=ExtendedMetricConfig(),
        )
        rows_b = extended_metric_records(
            first, target_b, pairs=[FakePair(0), FakePair(1)], context=CONTEXT,
            config=ExtendedMetricConfig(),
        )
        second = probe(batch, bundle, history=[history], lag_ps=torch.tensor([1000.0, 1000.0]))

    assert torch.equal(first.translation_local, second.translation_local)
    assert torch.equal(first.rotation, second.rotation)
    # The metrics did see the difference, or the test above proves nothing.
    assert rows_a[0]["ca_rmsd"] != pytest.approx(rows_b[0]["ca_rmsd"], rel=1e-3)
    # ... but the edge set, which is a function of the current frame only, did not.
    assert rows_a[0]["n_current_spatial_edges"] == rows_b[0]["n_current_spatial_edges"]


# --------------------------------------------------------------------------
# 28. no weight moves
# --------------------------------------------------------------------------


def test_a_full_scoring_pass_leaves_every_parameter_bitwise_identical(
    extractor, batch_and_frames  # noqa: F811
):
    """Test 28. Bitwise, over parameters *and* buffers."""
    batch, history, future = batch_and_frames
    probe = make_probe(extractor).eval()
    before = _parameter_digest(probe)
    extractor_before = _parameter_digest(extractor.phase1)

    with torch.no_grad():
        for _ in range(3):
            bundle = extractor(batch)
            prediction = probe(
                batch, bundle, history=[history], lag_ps=torch.tensor([1000.0, 1000.0])
            )
            target = build_transition_target(batch, future)
            extended_metric_records(
                prediction, target, pairs=[FakePair(0), FakePair(1)],
                context=CONTEXT, config=ExtendedMetricConfig(),
            )

    assert _parameter_digest(probe) == before
    assert _parameter_digest(extractor.phase1) == extractor_before


def test_scoring_the_same_batch_twice_gives_identical_records(
    extractor, batch_and_frames  # noqa: F811
):
    """Test 25, at the level the evaluator actually runs at."""
    batch, history, future = batch_and_frames
    probe = make_probe(extractor).eval()

    def score():
        with torch.no_grad():
            bundle = extractor(batch)
            prediction = probe(
                batch, bundle, history=[history], lag_ps=torch.tensor([1000.0, 1000.0])
            )
            return extended_metric_records(
                prediction, build_transition_target(batch, future),
                pairs=[FakePair(0), FakePair(1)], context=CONTEXT,
                config=ExtendedMetricConfig(),
            )

    assert score() == score()


def test_the_probes_own_graph_settings_drive_the_spatial_edge_subset(
    extractor, batch_and_frames  # noqa: F811
):
    """The evaluator reads k and the cutoff from the checkpoint, not a constant.

    A run whose ``backbone_cutoff`` differed would otherwise be scored on edges it
    never had, and the number would look perfectly reasonable.
    """
    batch, history, future = batch_and_frames
    probe = make_probe(extractor).eval()
    target = build_transition_target(batch, future)
    with torch.no_grad():
        prediction = probe(
            batch, extractor(batch), history=[history],
            lag_ps=torch.tensor([1000.0, 1000.0]),
        )

    counts = []
    for cutoff in (6.0, 30.0):
        config = ExtendedMetricConfig(
            spatial_knn=probe.config.residue_knn, spatial_cutoff=cutoff
        )
        rows = extended_metric_records(
            prediction, target, pairs=[FakePair(0), FakePair(1)],
            context=CONTEXT, config=config,
        )
        counts.append(rows[0]["n_current_spatial_edges"])
    assert counts[0] < counts[1]
