from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from force_md.heavy_flow import ForceNormalizer, build_domain_chunks, domain_order
from experiments.heavy_flow.train_physics_chunks import (
    _ChunkProgress,
    _parse_args,
    _load_normalizer_artifact,
    _save_normalizer_artifact,
)


def test_checkpoint_defaults_are_step_only(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_physics_chunks.py"])
    args = _parse_args()
    assert args.checkpoint_every == 500
    assert args.checkpoint_seconds == 0


def test_checkpoint_timer_remains_explicitly_configurable(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_physics_chunks.py", "--checkpoint-seconds", "900"])
    assert _parse_args().checkpoint_seconds == 900


def test_500_domain_chunks_are_non_overlapping_and_ordered() -> None:
    domains = [f"d{i:04d}" for i in range(1201)]

    chunks = build_domain_chunks(domains, chunk_size=500)

    assert [chunk.num_domains for chunk in chunks] == [500, 500, 201]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [domain for chunk in chunks for domain in chunk.domains] == domains
    assert all(chunk.chunk_size == 500 for chunk in chunks)
    assert all(chunk.total_chunks == 3 for chunk in chunks)
    assert set(chunks[0].domains).isdisjoint(chunks[1].domains)
    assert set(chunks[1].domains).isdisjoint(chunks[2].domains)


def test_manifest_order_is_authoritative_and_extra_files_fail_closed(tmp_path) -> None:
    domains = ["first", "second", "third"]
    for domain in domains:
        (tmp_path / f"mdcath_dataset_{domain}.h5").touch()
    manifest = tmp_path / "mdcath_manifest.json"
    manifest.write_text(json.dumps({
        "shards": [
            {"path": "data/mdcath_dataset_second.h5", "size": 0},
            {"path": "data/mdcath_dataset_first.h5", "size": 0},
            {"path": "data/mdcath_dataset_third.h5", "size": 0},
        ]
    }))

    assert domain_order(tmp_path, manifest_path=manifest) == [
        "second", "first", "third"
    ]

    (tmp_path / "mdcath_dataset_extra.h5").touch()
    with pytest.raises(ValueError, match="not listed"):
        domain_order(tmp_path, manifest_path=manifest)


def test_standalone_normalizer_artifact_roundtrip_and_spec_guard(tmp_path: Path) -> None:
    path = tmp_path / "normalizer.json"
    spec = {"data": "same", "train_key_sha256": "abc"}
    expected = ForceNormalizer(scale=2.5, count=7)

    _save_normalizer_artifact(path, expected, spec)

    loaded = _load_normalizer_artifact(path, spec)
    assert loaded == expected
    with pytest.raises(ValueError, match="different data/split"):
        _load_normalizer_artifact(path, {"data": "changed", "train_key_sha256": "abc"})


def _legacy_fit_spec() -> dict:
    return {
        "config": {
            "geometry": {"num_blocks": 4},
            "context": {"atom_refine_blocks": 2},
            "physics": {"physics_blocks": 3},
            "data": {"represented_scope": "heavy_atom", "seed": 0},
        },
        "data_dir": "/dataset",
        "force_unit": "kcal/mol/angstrom",
        "represented_scope": "heavy_atom",
        "quarantine_path": "/force_quarantine.json",
        "coord_quarantine_path": "/coord_quarantine.json",
        "frames_per_trajectory": 4,
        "max_train_frames_per_chunk": None,
        "chunk_size": 500,
        "domain_order": ["first", "second"],
        "split": {"train_count": 8, "train_key_sha256": "abc"},
    }


def test_legacy_normalizer_reused_after_edge_memory_tuning(tmp_path: Path) -> None:
    path = tmp_path / "normalizer.json"
    saved = _legacy_fit_spec()
    expected = ForceNormalizer(scale=26.979844952075947, count=67470557)
    _save_normalizer_artifact(path, expected, saved)
    original = path.read_bytes()
    current = deepcopy(saved)
    for section, prefix in (("geometry", ""), ("context", "atom_refine_"), ("physics", "")):
        current["config"][section][prefix + "edge_chunk_size"] = 2048
        current["config"][section][prefix + "activation_checkpoint"] = True

    assert _load_normalizer_artifact(path, current) == expected
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", list(_legacy_fit_spec()))
def test_normalizer_still_rejects_changed_data_identity(tmp_path: Path, field: str) -> None:
    path = tmp_path / "normalizer.json"
    saved = _legacy_fit_spec()
    _save_normalizer_artifact(path, ForceNormalizer(scale=2.5, count=7), saved)
    current = deepcopy(saved)
    if field == "config":
        current[field]["data"]["seed"] = 1
    elif field == "split":
        current[field]["train_key_sha256"] = "different-frames"
    else:
        current[field] = "changed"
    with pytest.raises(ValueError, match="different data/split.*changed fields"):
        _load_normalizer_artifact(path, current)


def test_progress_logs_interval_timeout_and_final_eta(monkeypatch, capsys) -> None:
    ticks = iter([0.0, 2.0, 3.0, 65.0, 66.0, 70.0])
    monkeypatch.setattr("experiments.heavy_flow.train_physics_chunks.time.perf_counter",
                        lambda: next(ticks))
    progress = _ChunkProgress(chunk_index=1, steps=5, steps_per_epoch=3,
                              step_offset=10, global_step_target=15,
                              log_every=4, log_seconds=60)
    for step in range(1, 6):
        progress(step, 0.5)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["chunk_step"] for row in rows] == [1, 3, 4, 5]
    assert rows[1]["seconds_per_step"] == 31.5
    assert rows[1]["eta_seconds"] == 63.0
    assert rows[-1]["global_step"] == 15
    assert rows[-1]["epoch"] == 2
    assert rows[-1]["progress_percent"] == 100.0
    assert rows[-1]["eta_seconds"] == 0.0


def test_stream_progress_reports_completed_optimizer_steps() -> None:
    import torch
    from force_md.heavy_flow import ForceHead, HeavyFlowPhysicsModel, HeavyFlowSample, HeavyFlowTargets
    from experiments.heavy_flow.train_physics import train_physics_stream_steps
    from tests.heavy_flow.test_stage2_seq_geo import condition, context_model
    from tests.heavy_flow.test_stage3_physics import small_predictor

    torch.manual_seed(37)
    c = condition()
    context = context_model()
    predictor = small_predictor(context(c))
    model = HeavyFlowPhysicsModel(context, predictor, ForceHead(12, vector_channels=2))
    sample = HeavyFlowSample(c, HeavyFlowTargets(
        force_current=torch.zeros_like(c.current_positions),
        x_future=c.current_positions.clone(), force_mask=c.atom_mask.clone(),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    observed = []

    def report(step, loss):
        adam_steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
        assert adam_steps and max(adam_steps) == step
        observed.append((step, loss))

    losses = train_physics_stream_steps(model, [sample], optimizer, steps=2,
                                       step_offset=10, progress_callback=report)
    assert observed == list(enumerate(losses, start=1))
