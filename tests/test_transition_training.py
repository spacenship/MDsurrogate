"""The Phase 1.5 trainer and its results contract (Checkpoint 6).

Runs entirely offline: batches are assembled from synthetic proteins into
:class:`LagPairBatch` directly, so the loop, the provenance record, the
checkpointing and the results aggregation are all testable without an mdCATH
shard.

The properties under test are the ones that make an ablation an ablation:
provenance that can prove two arms saw the same experiment, validation that
reports micro **and** domain-macro beside the identity baseline, and a checkpoint
that restores exactly what it saved.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

torch = pytest.importorskip("torch")

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.data.adapters.lag_pairs import LagPair, LagPairBatch, LagPairManifest  # noqa: E402
from force_md.data.contracts import FrameGeometry  # noqa: E402
from force_md.geometry import frame_atom_indices, so3_exp_map  # noqa: E402
from force_md.models.local_physics import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn.hierarchical_encoder import EncoderConfig  # noqa: E402
from force_md.nn.irreps import IrrepsConfig  # noqa: E402
from force_md.training.transition_module import (  # noqa: E402
    TransitionTrainConfig,
    TransitionTrainer,
)
from force_md.transition import (  # noqa: E402
    ConditionerConfig,
    FrozenPhase1Extractor,
    TransitionProbe,
    TransitionProbeConfig,
)

PLM_DIM = 32


@pytest.fixture(scope="module")
def extractor(tmp_path_factory):
    config = LocalPhysicsConfig(
        encoder=EncoderConfig(
            plm_dim=PLM_DIM, num_cycles=1,
            irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
        ),
        use_energy_branch=False,
    )
    model = LocalPhysicsModel(config)
    path = str(tmp_path_factory.mktemp("ck") / "phase1.pt")
    torch.save(
        {"state_dict": model.state_dict(), "model_config": config, "step": 42,
         "latent_contract": model.latent_contract()},
        path,
    )
    return FrozenPhase1Extractor.from_checkpoint(path)


def _frame(batch, positions, offset):
    indices, complete = frame_atom_indices(batch)
    safe = indices.clamp(min=0)
    return FrameGeometry(
        positions=positions,
        n_positions=positions[safe[:, 0]],
        ca_positions=positions[safe[:, 1]],
        c_positions=positions[safe[:, 2]],
        frame_valid=batch.backbone.frame_valid & complete,
        atom_batch_index=batch.atoms.batch_index,
        residue_batch_index=batch.residues.batch_index,
        frame_index=batch.frame_index + offset,
    )


def _moved(batch, *, scale, seed, offset):
    g = torch.Generator().manual_seed(seed)
    dtype = batch.atoms.positions.dtype
    shift = torch.randn(batch.num_residues, 3, dtype=dtype, generator=g) * scale
    turn = so3_exp_map(
        torch.randn(batch.num_residues, 3, dtype=dtype, generator=g) * scale * 0.3
    )
    a2r = batch.atoms.atom_to_residue
    ca = batch.backbone.ca_positions
    local = batch.atoms.positions - ca[a2r]
    return _frame(batch, torch.einsum("nij,nj->ni", turn[a2r], local) + ca[a2r] + shift[a2r], offset)


def make_lag_batch(domains=("a", "b"), sizes=(6, 5), lag_ps=(1000.0, 4000.0), seed=0):
    """A LagPairBatch assembled from synthetic proteins -- no mdCATH needed."""
    batch = synthetic_batch(
        [SyntheticSpec(n) for n in sizes], seed=seed, plm_dim=PLM_DIM,
        include_hydrogens=True,
    )
    pairs = tuple(
        LagPair(domain=d, temperature="320", replica="0", current_frame=5,
                lag_frames=int(lag // 1000), lag_ps=float(lag), past_frames=1)
        for d, lag in zip(domains, lag_ps)
    )
    dtype = batch.atoms.positions.dtype
    return LagPairBatch(
        pairs=pairs,
        current=batch,
        hidden_force_target=None,
        history=(_moved(batch, scale=0.15, seed=seed + 11, offset=-1),),
        future=_moved(batch, scale=0.4, seed=seed + 12, offset=4),
        lag_ps=torch.tensor([float(v) for v in lag_ps], dtype=dtype),
        lag_frames=torch.tensor([int(v // 1000) for v in lag_ps], dtype=torch.int64),
    )


def make_trainer(extractor, *, arm="physics_latent", manifest=None, **overrides):
    torch.manual_seed(0)
    probe = TransitionProbe(
        TransitionProbeConfig(
            arm=arm,
            plm_dim=PLM_DIM,
            num_blocks=2,
            irreps=IrrepsConfig(scalar_channels=16, vector_channels=4, tensor_channels=2),
            conditioner=ConditionerConfig(d_cond=16, hidden=32, atom_message_dim=16),
        ),
        latent_irreps=extractor.contract["physics_latent_irreps"],
    )
    defaults = dict(device="cpu", max_steps=6, warmup_steps=2, eval_every=3, log_every=100)
    defaults.update(overrides)
    return TransitionTrainer(
        probe, extractor, TransitionTrainConfig(**defaults), manifest=manifest
    )


@pytest.fixture
def loader():
    return [make_lag_batch(seed=0), make_lag_batch(domains=("c", "d"), sizes=(7, 4), seed=5)]


# --------------------------------------------------------------------------
# provenance -- what makes two runs comparable
# --------------------------------------------------------------------------


def test_provenance_records_everything_needed_to_compare_two_arms(extractor, tmp_path):
    manifest = LagPairManifest(
        pairs=(LagPair("a", "320", "0", 5, 1, 1000.0, 1),), metadata={"num_domains": 1}
    )
    trainer = make_trainer(extractor, manifest=manifest)
    record = trainer.provenance()

    assert record["arm"] == "physics_latent"
    assert record["manifest_hash"] == manifest.content_hash()
    assert record["phase1_sha256"] == extractor.metadata["checkpoint_sha256"]
    assert record["phase1_step"] == 42
    assert record["parameter_count"] == trainer.module.parameter_count()
    assert record["trainable_parameter_count"] == record["parameter_count"]
    assert record["latent_contract"] == extractor.contract
    assert record["train"]["seed"] == 0


def test_arms_differ_only_in_the_conditioner_share_of_the_parameters(extractor):
    counts = {}
    for arm in ("structure_only", "physics_latent", "force_pattern_shape"):
        record = make_trainer(extractor, arm=arm).provenance()
        counts[arm] = record["parameter_breakdown"]
    backbones = {c["blocks"] for c in counts.values()}
    heads = {c["heads"] for c in counts.values()}
    assert len(backbones) == 1, "arms must share one backbone"
    assert len(heads) == 1, "arms must share one head"
    assert counts["structure_only"]["conditioner"] == 0
    assert counts["force_pattern_shape"]["conditioner"] > counts["physics_latent"]["conditioner"]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def test_a_training_step_reports_physical_units(extractor, loader):
    trainer = make_trainer(extractor)
    components = trainer.train_step(loader[0])
    for key in ("total", "translation", "rotation", "translation_rmse_angstrom",
                "rotation_error_deg", "grad_norm", "lr"):
        assert key in components
    assert components["translation_rmse_angstrom"] > 0
    assert trainer.step == 1


def test_training_reduces_the_loss_on_a_repeated_batch(extractor, loader):
    torch.set_num_threads(1)
    trainer = make_trainer(extractor, max_steps=40, warmup_steps=5, learning_rate=3e-3)
    first = trainer.train_step(loader[0])["total"]
    for _ in range(39):
        last = trainer.train_step(loader[0])["total"]
    assert last < first


def test_the_probe_starts_at_the_identity_baseline(extractor, loader):
    """Before any step, the arm's Ca RMSD must equal the identity baseline."""
    trainer = make_trainer(extractor)
    summary, _ = trainer.evaluate(loader)
    for row in summary["rows"]:
        assert row["ca_rmsd_micro"] == pytest.approx(
            row["ca_rmsd_identity_micro"], rel=1e-5
        )


def test_fit_runs_and_records_history(extractor, loader):
    trainer = make_trainer(extractor)
    history = trainer.fit(loader, loader, log=None)
    assert trainer.step == trainer.config.max_steps
    assert history and all("step" in row for row in history)


def test_frozen_phase1_is_untouched_by_training(extractor, loader):
    before = [p.detach().clone() for p in extractor.phase1.parameters()]
    trainer = make_trainer(extractor)
    trainer.fit(loader, None, log=None)
    for old, new in zip(before, extractor.phase1.parameters()):
        assert torch.equal(old, new)
    assert all(p.grad is None for p in extractor.phase1.parameters())


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def test_evaluation_reports_both_aggregations_and_the_baseline(extractor, loader):
    trainer = make_trainer(extractor)
    summary, records = trainer.evaluate(loader)

    assert summary["rows"], "no aggregated rows"
    for row in summary["rows"]:
        for key in ("ca_rmsd_micro", "ca_rmsd_domain_macro",
                    "ca_rmsd_identity_micro", "rotation_geodesic_deg_micro"):
            assert key in row, key
        assert row["domain_count"] >= 1
    assert {r["lag_ns"] for r in summary["rows"]} == {1.0, 4.0}
    assert len(records) == sum(b.num_graphs for b in loader)
    assert {r["domain"] for r in records} == {"a", "b", "c", "d"}


def test_evaluation_does_not_update_the_probe(extractor, loader):
    trainer = make_trainer(extractor)
    before = [p.detach().clone() for p in trainer.module.parameters()]
    trainer.evaluate(loader)
    for old, new in zip(before, trainer.module.parameters()):
        assert torch.equal(old, new)


def test_evaluation_can_be_capped(extractor, loader):
    trainer = make_trainer(extractor)
    _, all_records = trainer.evaluate(loader)
    _, capped = trainer.evaluate(loader, max_batches=1)
    assert len(capped) < len(all_records)


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_the_run(extractor, loader, tmp_path):
    trainer = make_trainer(extractor)
    trainer.fit(loader, None, log=None)
    path = str(tmp_path / "last.pt")
    trainer.save_checkpoint(path)

    probe, restored = TransitionTrainer.load_checkpoint(path, extractor, device="cpu")
    assert restored.step == trainer.step
    assert probe.config.arm == trainer.module.config.arm
    for old, new in zip(trainer.module.parameters(), probe.parameters()):
        assert torch.equal(old, new)

    with torch.no_grad():
        a = trainer._forward(loader[0], model=trainer.module)[0]
        b = restored._forward(loader[0], model=restored.module)[0]
    assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_checkpoints_carry_their_provenance(extractor, loader, tmp_path):
    manifest = LagPairManifest(pairs=(LagPair("a", "320", "0", 5, 1, 1000.0, 1),), metadata={})
    trainer = make_trainer(extractor, manifest=manifest)
    path = str(tmp_path / "last.pt")
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["provenance"]["manifest_hash"] == manifest.content_hash()
    assert payload["provenance"]["phase1_sha256"] == extractor.metadata["checkpoint_sha256"]


def test_a_non_finite_model_is_never_checkpointed(extractor, tmp_path):
    trainer = make_trainer(extractor)
    with torch.no_grad():
        next(iter(trainer.module.parameters())).fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite weights"):
        trainer.save_checkpoint(str(tmp_path / "last.pt"))
    assert not os.path.exists(str(tmp_path / "last.pt"))


def test_resuming_continues_from_the_saved_step(extractor, loader, tmp_path):
    trainer = make_trainer(extractor)
    trainer.fit(loader, None, log=None)
    path = str(tmp_path / "last.pt")
    trainer.save_checkpoint(path)

    fresh = make_trainer(extractor)
    assert fresh.step == 0
    fresh.load_state(path)
    assert fresh.step == trainer.step


# --------------------------------------------------------------------------
# the results table
# --------------------------------------------------------------------------


def test_result_rows_carry_every_required_column(extractor, loader):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_phase1_5_ablation import COLUMNS, rows_for  # noqa: PLC0415

    trainer = make_trainer(extractor)
    summary, _ = trainer.evaluate(loader)
    rows = rows_for("physics_latent", 0, 6, trainer.provenance(), summary, 1.23, "val")

    assert rows
    required = {
        "arm", "seed", "lag_ns", "split", "step", "parameter_count",
        "trainable_parameter_count", "domain_count", "pair_count", "ca_rmsd",
        "translation_rmse", "rotation_geodesic_deg", "pair_distance_mae",
        "contact_f1", "clash_rate", "train_loss", "val_loss",
    }
    for row in rows:
        assert required <= set(row), required - set(row)
        assert set(row) <= set(COLUMNS)
    assert {row["aggregation"] for row in rows} == {"micro", "domain_macro"}


def test_a_finite_but_absurd_loss_counts_as_divergence(extractor, loader):
    """The guard the probe's body-order-3 blowup slipped past.

    That failure reached a loss of 1e14 with every value a perfectly good float,
    so testing finiteness alone let it through and the run would have spent its
    whole budget on it. ``max_loss`` is what closes that.
    """
    trainer = make_trainer(extractor, max_loss=10.0, max_consecutive_skips=2)
    before = [p.detach().clone() for p in trainer.module.parameters()]

    real = trainer._forward

    def absurd(batch):
        total, components, a, b, c = real(batch)
        return total * 1e14, {**components, "total": 1e14}, a, b, c

    trainer._forward = absurd

    trainer.train_step(loader[0])
    assert trainer.skipped_steps == 1
    for old, new in zip(before, trainer.module.parameters()):
        assert torch.equal(old, new), "a skipped step must not move a weight"

    trainer.train_step(loader[0])
    with pytest.raises(RuntimeError, match="divergence, not a bad frame"):
        trainer.train_step(loader[0])


def test_an_ordinary_bad_batch_is_not_mistaken_for_divergence(extractor, loader):
    """max_loss sits far above anything training produces: 1e4 vs ~1.2 healthy."""
    trainer = make_trainer(extractor)
    components = trainer.train_step(loader[0])
    assert abs(components["total"]) < 100.0
    assert trainer.skipped_steps == 0


def test_evaluation_defaults_to_the_whole_loader(extractor, loader):
    """`eval_batches` must not leak into a final measurement.

    It once did, and because the validation loader is unshuffled the 72,080-pair
    / 181-domain validation silently became 1,600 pairs from the same four
    domains on every arm and every seed -- internally consistent, and wrong.
    """
    trainer = make_trainer(extractor, eval_batches=1)
    full, _ = trainer.evaluate(loader, split="val")
    capped, _ = trainer.evaluate(loader, split="val", max_batches=1)

    seen = {row["lag_ns"]: row["graph_count"] for row in full["rows"]}
    capped_seen = {row["lag_ns"]: row["graph_count"] for row in capped["rows"]}
    assert sum(seen.values()) > sum(capped_seen.values()), (
        "evaluate() ignored batches beyond config.eval_batches"
    )
    assert sum(seen.values()) == sum(
        b.num_graphs for b in loader
    ), "evaluate() did not cover every pair in the loader"
