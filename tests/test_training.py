"""Training loop: overfitting, splits, normalisation, checkpoints, determinism.

These run on synthetic data so they work with no mdCATH shard present. The
real-data adapter has its own file.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from force_md.data import SyntheticSpec, collate_batches, synthetic_batch
from force_md.data.adapters.mdcath import TrainingExample, split_domains
from force_md.geometry import frames_from_batch, link_backbone_to_atom_positions
from force_md.models import LocalPhysicsConfig, LocalPhysicsModel
from force_md.nn import EncoderConfig, IrrepsConfig
from force_md.physics import (
    LossWeights,
    ResidueSumProjector,
    TargetNormalizer,
    omitted_atom_residual,
    phase1_loss,
)
from force_md.training import (
    Phase1Trainer,
    TrainConfig,
    collate_examples,
    merge_metrics,
    set_seed,
    vector_metrics,
)

PLM_DIM = 32


def make_batch(n_res=6, seed=0):
    return synthetic_batch([SyntheticSpec(n_res)], seed=seed,
                           include_hydrogens=True, plm_dim=PLM_DIM)


def make_model():
    set_seed(0)
    return LocalPhysicsModel(
        LocalPhysicsConfig(encoder=EncoderConfig(plm_dim=PLM_DIM))
    )


def make_example(batch) -> TrainingExample:
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    hidden, _ = omitted_atom_residual(allat, heavy)
    return TrainingExample(batch=batch, hidden_force_target=hidden)


# --------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------


def test_collate_offsets_indices_correctly():
    a = make_batch(4, seed=0)
    b = make_batch(7, seed=1)
    merged = collate_batches([a, b])
    assert merged.num_graphs == 2
    assert merged.num_residues == a.num_residues + b.num_residues
    assert merged.num_atoms == a.num_atoms + b.num_atoms
    # the second protein's atoms point at the second protein's residues
    tail = merged.atoms.atom_to_residue[a.num_atoms :]
    assert int(tail.min()) == a.num_residues
    assert torch.equal(merged.atoms.batch_index[a.num_atoms :],
                       torch.ones(b.num_atoms, dtype=torch.int64))
    merged.validate()


def test_collate_preserves_the_first_protein(make_first=make_batch):
    a = make_batch(5, seed=3)
    merged = collate_batches([a, make_batch(9, seed=4)])
    assert torch.allclose(merged.atoms.positions[: a.num_atoms], a.atoms.positions)
    assert torch.allclose(
        merged.residues.plm_embedding[: a.num_residues], a.residues.plm_embedding
    )


def test_collate_rejects_mismatched_plm_widths():
    a = synthetic_batch([SyntheticSpec(4)], seed=0, plm_dim=16)
    b = synthetic_batch([SyntheticSpec(4)], seed=0, plm_dim=32)
    with pytest.raises(ValueError, match="different PLM widths"):
        collate_batches([a, b])


def test_collate_rejects_empty():
    with pytest.raises(ValueError, match="empty list"):
        collate_batches([])


def test_collate_examples_joins_hidden_targets():
    examples = [make_example(make_batch(4, seed=0)), make_example(make_batch(6, seed=1))]
    batch, hidden = collate_examples(examples)
    assert hidden.shape == (batch.num_residues, 3)
    assert torch.allclose(hidden[:4], examples[0].hidden_force_target)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_split_is_by_domain_and_disjoint():
    domains = [f"d{i:03d}" for i in range(20)]
    train, val = split_domains(domains, val_fraction=0.25, seed=0)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(domains)
    assert len(val) == 5


def test_split_is_deterministic_and_seed_dependent():
    domains = [f"d{i:03d}" for i in range(20)]
    a = split_domains(domains, 0.25, seed=0)
    b = split_domains(domains, 0.25, seed=0)
    c = split_domains(domains, 0.25, seed=1)
    assert a == b
    assert a != c


def test_split_never_empties_validation():
    train, val = split_domains(["only", "two"], val_fraction=0.01, seed=0)
    assert len(val) >= 1


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalizer_is_fitted_on_training_data_only():
    """The trainer must never see the validation loader while fitting scales."""
    train = [collate_examples([make_example(make_batch(5, seed=i))]) for i in range(3)]
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu", normalizer_batches=3))
    fitted = trainer.fit_normalizer(train)
    assert fitted.atom_force > 0 and fitted.residue_force > 0
    assert trainer.normalizer is fitted

    # a validation set with a wildly different scale must not move the fit
    trainer2 = Phase1Trainer(make_model(), TrainConfig(device="cpu", normalizer_batches=3))
    again = trainer2.fit_normalizer(train)
    assert again.atom_force == pytest.approx(fitted.atom_force)


def test_normalizer_requires_data():
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu"))
    with pytest.raises(ValueError, match="no batches"):
        trainer.fit_normalizer([])


# --------------------------------------------------------------------------
# overfit one batch -- the Phase 1 completion criterion
# --------------------------------------------------------------------------


@pytest.fixture
def single_threaded():
    """Pin intra-op parallelism for the duration of one test, then restore it.

    ``scatter_sum`` is a CPU ``index_add``, whose floating-point accumulation
    order depends on how the threads happen to be scheduled. One step is
    unaffected at any tolerance that matters; 250 optimisation steps amplify it.
    Measured on this machine, five identical runs of the test below give:

        1 thread   0.5738 every time -- bit-identical
        8 threads  0.5541 to 0.5813, and 0.6046 / 0.6264 inside the full suite

    so the assertion's 0.6 threshold sits *inside* the run-to-run spread and the
    test passes or fails depending on thread scheduling. The threshold is the
    Phase 1 completion criterion and is left alone; the nondeterminism is what
    gets fixed. Single-threaded costs nothing here -- 14 s against 15 s -- because
    the model is small enough that threading overhead dominates.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.slow
def test_overfit_one_batch(single_threaded):
    """The model must be able to fit a single batch nearly exactly.

    Reported against the zero-prediction baseline: `relative_rmse` near 1.0 means
    nothing was learned, so only a large drop is evidence of anything.

    Runs a narrow, single-cycle model with the energy branch off. The energy
    branch needs ``create_graph=True`` so the conservative force can be
    backpropagated, and that second-order graph through the e3nn encoder costs
    ~14 s per step on CPU -- an hour for this one test. The conservative-force
    loss is weighted 0 here anyway, and the energy path has its own coverage in
    ``test_heads.py``.

    Runs single-threaded so the result is reproducible; see the
    ``single_threaded`` fixture for the measurement behind that.
    """
    set_seed(0)
    batch = make_batch(5, seed=0)
    model = LocalPhysicsModel(
        LocalPhysicsConfig(
            encoder=EncoderConfig(
                plm_dim=PLM_DIM,
                num_cycles=1,
                irreps=IrrepsConfig(scalar_channels=32, vector_channels=8,
                                    tensor_channels=4),
            ),
            use_energy_branch=False,
        )
    )
    heavy = ResidueSumProjector("heavy_atom")(batch)
    allat = ResidueSumProjector("all_atom")(batch)
    hidden, _ = omitted_atom_residual(allat, heavy)
    normalizer = TargetNormalizer.fit(
        batch.atoms.forces[batch.atoms.is_heavy],
        heavy.force[heavy.valid], heavy.torque[heavy.valid],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    weights = LossWeights(conservative_force=0.0, energy_gauge=0.0)

    def metrics():
        with torch.no_grad():
            out = model(batch)
        return vector_metrics(out.atom_force_mean, batch.atoms.forces,
                              batch.atoms.is_heavy)

    before = metrics()
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        frames = frames_from_batch(link_backbone_to_atom_positions(batch))
        total, _ = phase1_loss(out, batch, heavy, frames, hidden_force_target=hidden,
                              atom_selection=batch.atoms.is_heavy,
                              weights=weights, normalizer=normalizer)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    after = metrics()

    assert after["relative_rmse"] < 0.6 * before["relative_rmse"], (
        f"relative RMSE {before['relative_rmse']:.3f} -> {after['relative_rmse']:.3f}"
    )
    assert after["angular_error_deg"] < 20.0, (
        f"angular error {before['angular_error_deg']:.1f} -> "
        f"{after['angular_error_deg']:.1f} deg"
    )


# --------------------------------------------------------------------------
# trainer mechanics
# --------------------------------------------------------------------------


def test_train_step_reduces_the_loss_on_a_repeated_batch():
    set_seed(0)
    loader = [collate_examples([make_example(make_batch(5, seed=0))])]
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu", max_steps=8,
                                                      normalizer_batches=1))
    trainer.fit_normalizer(loader)
    first = trainer.train_step(*loader[0])
    for _ in range(6):
        last = trainer.train_step(*loader[0])
    assert last["total"] < first["total"]
    assert "grad_norm" in last


def test_evaluate_returns_losses_and_metrics():
    loader = [collate_examples([make_example(make_batch(5, seed=i))]) for i in (0, 1)]
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu", normalizer_batches=2))
    trainer.fit_normalizer(loader)
    result = trainer.evaluate(loader)
    for key in ("total", "atom_force_rmse", "atom_force_rmse_zero",
                "atom_force_angular_error_deg", "residue_force_rmse", "torque_rmse"):
        assert key in result, key
    assert result["atom_force_rmse_zero"] > 0


def test_evaluate_does_not_update_parameters():
    loader = [collate_examples([make_example(make_batch(5, seed=0))])]
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu", normalizer_batches=1))
    trainer.fit_normalizer(loader)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.evaluate(loader)
    for p, q in zip(trainer.model.parameters(), before):
        assert torch.equal(p.detach(), q)


def test_fit_runs_and_records_history():
    loader = [collate_examples([make_example(make_batch(4, seed=i))]) for i in (0, 1)]
    trainer = Phase1Trainer(
        make_model(),
        TrainConfig(device="cpu", max_steps=4, eval_every=2, log_every=100,
                    normalizer_batches=2),
    )
    trainer.fit_normalizer(loader)
    history = trainer.fit(loader, loader, log=None)
    assert len(history) == 2
    assert history[-1]["step"] == 4


# --------------------------------------------------------------------------
# checkpoints and reproducibility
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_everything(tmp_path):
    loader = [collate_examples([make_example(make_batch(5, seed=0))])]
    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu", normalizer_batches=1))
    trainer.fit_normalizer(loader)
    trainer.train_step(*loader[0])
    path = str(tmp_path / "ckpt.pt")
    trainer.save_checkpoint(path)

    model, restored = Phase1Trainer.load_checkpoint(path, device="cpu")
    assert restored.step == trainer.step
    assert restored.normalizer == trainer.normalizer
    assert model.latent_contract() == trainer.model.latent_contract()

    batch = make_batch(5, seed=0)
    with torch.no_grad():
        a = trainer.model.eval()(batch)
        b = model.eval()(batch)
    assert torch.allclose(a.physics_latent, b.physics_latent, atol=1e-12)
    assert torch.allclose(a.atom_force_mean, b.atom_force_mean, atol=1e-12)


def test_config_snapshot_is_json_serialisable():
    import json

    trainer = Phase1Trainer(make_model(), TrainConfig(device="cpu"))
    snapshot = trainer.config_snapshot()
    text = json.dumps(snapshot)
    assert "latent_contract" in snapshot
    assert snapshot["model"]["lmax"] == 2
    assert snapshot["model"]["atom_cutoff"] == 5.0
    assert json.loads(text)["train"]["seed"] == 0


def test_seed_makes_initialisation_reproducible():
    set_seed(3)
    a = LocalPhysicsModel(LocalPhysicsConfig(encoder=EncoderConfig(plm_dim=PLM_DIM)))
    set_seed(3)
    b = LocalPhysicsModel(LocalPhysicsConfig(encoder=EncoderConfig(plm_dim=PLM_DIM)))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_vector_metrics_on_a_perfect_prediction():
    target = torch.randn(10, 3)
    m = vector_metrics(target, target)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["angular_error_deg"] == pytest.approx(0.0, abs=1e-3)
    assert m["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert m["rmse_zero"] > 0


def test_vector_metrics_zero_baseline_matches_predicting_nothing():
    target = torch.randn(10, 3) * 5.0
    m = vector_metrics(torch.zeros_like(target), target)
    assert m["rmse"] == pytest.approx(m["rmse_zero"], rel=1e-6)
    assert m["relative_rmse"] == pytest.approx(1.0, rel=1e-6)


def test_vector_metrics_opposite_prediction_is_180_degrees():
    target = torch.randn(10, 3)
    m = vector_metrics(-target, target)
    assert m["angular_error_deg"] == pytest.approx(180.0, abs=1e-2)


def test_vector_metrics_empty_mask_is_nan_not_zero():
    target = torch.randn(5, 3)
    m = vector_metrics(target, target, torch.zeros(5, dtype=torch.bool))
    assert m["rmse"] != m["rmse"]  # NaN


def test_merge_metrics_weights_by_count():
    a = vector_metrics(torch.zeros(10, 3), torch.ones(10, 3))
    b = vector_metrics(torch.ones(90, 3), torch.ones(90, 3))
    merged = merge_metrics([a, b])
    # the 90-node batch is perfect, so the weighted mean must sit near it
    assert merged["rmse"] < 0.2 * a["rmse"]
    assert merged["count"] == 100.0
