from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.heavy_flow.physics_runtime import batch_indices, batches, capture_rng, restore_rng
from experiments.heavy_flow.train_physics import build_stage3_model, train_physics_stream_steps
from force_md.heavy_flow import ForceNormalizer, HeavyFlowSample, HeavyFlowTargets
from force_md.heavy_flow.atom_graph import _top_k_spatial
from tests.heavy_flow.test_stage2_seq_geo import condition
from tests.heavy_flow.test_stage3_stage4_handoff import _stage3_config


def runtime_config(dropout=0.0):
    config = _stage3_config()
    config["context"]["dropout"] = dropout
    config["context"]["geometry_modality_dropout"] = 0.35 if dropout else 0.0
    for section, prefix in (("geometry", ""), ("context", "atom_refine_"), ("physics", "")):
        config[section][prefix + "edge_chunk_size"] = 3
        config[section][prefix + "activation_checkpoint"] = True
    return config


def samples(n=5):
    result = []
    for i in range(n):
        c = condition()
        target = torch.full_like(c.current_positions, (i + 1) * 0.1)
        result.append(HeavyFlowSample(c, HeavyFlowTargets(
            force_current=target, x_future=c.current_positions.clone(), force_mask=c.atom_mask.clone())))
    return result


@pytest.mark.parametrize("n,batch,world", [(5, 1, 2), (1, 1, 2), (7, 2, 3), (8, 2, 2)])
def test_rank_partition_exact_coverage(n, batch, world):
    steps = math.ceil(n / (batch * world))
    for epoch in range(2):
        seen = [i for s in range(steps) for r in range(world)
                for i in batch_indices(n, batch, r, world, s + epoch * steps)]
        assert seen == list(range(n))


@pytest.mark.parametrize("n", [1, 6, 31, 270])
def test_tiled_neighbors_match_legacy(n):
    torch.manual_seed(72)
    positions = torch.randn(n, 3)
    if n > 2:
        positions[1] = positions[2]  # deterministic tie handling
    distances = torch.cdist(positions, positions)
    valid = (distances <= 1.7) & ~torch.eye(n, dtype=torch.bool)
    expected = []
    degrees = []
    for destination in range(n):
        sources = valid[:, destination].nonzero().flatten().tolist()
        degrees.append(len(sources))
        sources.sort(key=lambda source: (float(distances[source, destination]), source))
        expected.extend((s, destination) for s in sources[:3])
    actual, candidates, truncated, maximum = _top_k_spatial(positions, cutoff=1.7, max_neighbors=3)
    assert actual == expected
    assert candidates == sum(degrees)
    assert truncated == sum(d > 3 for d in degrees)
    assert maximum == min(3, max(degrees))


def test_prefetch_matches_serial_and_propagates_errors():
    data = samples(3)
    for depth in (0, 1, 2):
        got = list(batches(data, batch_size=1, start_step=1, steps=4, prefetch=depth))
        assert [float(s.targets.force_current[0, 0, 0]) for s, _ in got] == pytest.approx([0.2, 0.3, 0.1])
    class Broken:
        def __len__(self):
            return 2
        def __getitem__(self, i):
            raise ValueError("broken frame")
    with pytest.raises(ValueError, match="broken frame"):
        list(batches(Broken(), batch_size=1, start_step=0, steps=2, prefetch=1))


def _ddp_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(91)
        config = runtime_config()
        model = build_stage3_model(config, device="cpu")
        # SGD tests gradient weighting directly; Adam can amplify different
        # floating-point reduction noise in nearly-zero gradients.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        losses = train_physics_stream_steps(model, samples(), optimizer, steps=3, prefetch=1)
        if rank == 0:
            torch.save({"model": model.state_dict(), "losses": losses}, Path(directory) / "ddp.pt")
        # Dropout and checkpoint recomputation exercise per-rank RNG resume.
        torch.manual_seed(73 + rank)
        config = runtime_config(dropout=0.2)
        model = build_stage3_model(config, device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        saved = {}
        def checkpoint(step, loss):
            if step == 1:
                from copy import deepcopy
                saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), rng=capture_rng())
        full = train_physics_stream_steps(model, samples(), optimizer, steps=3, progress_callback=checkpoint)
        resumed = build_stage3_model(config, device="cpu")
        resumed.load_state_dict(saved["model"], strict=True)
        opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
        opt.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        tail = train_physics_stream_steps(resumed, samples(), opt, steps=3, start_step=1)
        assert tail == full[1:]
        for key, value in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_matches_global_batch_and_resumes_rng(tmp_path):
    mp.spawn(_ddp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(91)
    model = build_stage3_model(runtime_config(), device="cpu")
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    losses = train_physics_stream_steps(model, samples(), opt, steps=3, batch_size=2, prefetch=0)
    ddp = torch.load(tmp_path / "ddp.pt", weights_only=False)
    assert ddp["losses"] == pytest.approx(losses, rel=2e-5, abs=2e-5)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(ddp["model"][key], value, rtol=2e-4, atol=2e-5)


def test_nonfinite_gradient_never_updates_optimizer():
    model = build_stage3_model(runtime_config(), device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_hook(lambda g: g * float("nan"))
    with pytest.raises(FloatingPointError, match="optimizer not updated"):
        train_physics_stream_steps(model, samples(1), opt, steps=1)
    assert not opt.state


def test_runner_mid_chunk_checkpoint_resume(monkeypatch, tmp_path):
    import sys
    import yaml
    from experiments.heavy_flow import train_physics_chunks as runner

    config = runtime_config(dropout=0.2)
    config["data"] = {"seed": 11, "temporal_block_size": 1,
                      "same_domain_validation_fraction": 0.0,
                      "unseen_domain_validation_fraction": 0.0}
    cfg = tmp_path / "config.yaml"
    from experiments.heavy_flow.physics_runtime import rank_world
    rank, world = rank_world()
    if rank == 0:
        cfg.write_text(yaml.safe_dump(config))
    if world > 1:
        dist.barrier()
    monkeypatch.setattr(runner, "domain_order", lambda *a, **kw: ["a", "b"])
    class Reader:
        index = [(d, "320", "0", i) for d in ("a", "b") for i in range(3)]
        def close(self):
            pass
    monkeypatch.setattr(runner, "_reader", lambda *a: Reader())
    data = samples(3)
    monkeypatch.setattr(runner, "HeavyFlowPhysicsFrameDataset", lambda reader, keys, **kwargs: [data[k.frame] for k in keys])
    monkeypatch.setattr(runner.ForceNormalizer, "fit_from_force_arrays", lambda *a, **k: ForceNormalizer(scale=1.0, count=18))
    def run(output, resume=None):
        argv = ["train", "--config", str(cfg), "--device", "cpu", "--chunk-size", "1",
                "--output", str(output), "--normalizer-path", str(tmp_path / "normalizer.json"),
                "--checkpoint-every", "1", "--checkpoint-seconds", "0", "--fit-normalizer"]
        if resume:
            argv.extend(["--resume", str(resume)])
        monkeypatch.setattr(sys, "argv", argv)
        runner._run(runner._parse_args())

    full_path = tmp_path / "full.pt"
    run(full_path)
    save = runner._atomic_save_stage3
    def interrupt(*a, **kw):
        save(*a, **kw)
        if kw["chunk_rotation"]["step_in_chunk"] == 1:
            raise RuntimeError("simulated interruption after save")
    monkeypatch.setattr(runner, "_atomic_save_stage3", interrupt)
    partial = tmp_path / "partial.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(partial)
    raw = torch.load(partial, weights_only=False)
    assert raw["step"] == 1
    assert raw["chunk_rotation"]["next_chunk"] == 0
    assert raw["chunk_rotation"]["step_in_chunk"] == 1
    assert all(not key.startswith("module.") for key in raw["upstream"])
    monkeypatch.setattr(runner, "_atomic_save_stage3", save)
    run(partial, partial)
    full = torch.load(full_path, weights_only=False)
    resumed = torch.load(partial, weights_only=False)
    expected_steps = 2 * math.ceil(3 / world)
    assert full["step"] == resumed["step"] == expected_steps
    assert full["chunk_rotation"]["chunk_metrics"] == resumed["chunk_rotation"]["chunk_metrics"]
    for key, value in full["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][key], value, rtol=0, atol=0)
    for pid, state in full["optimizer_state"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(resumed["optimizer_state"]["state"][pid][key], value, rtol=0, atol=0)


def _ddp_runner_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            test_runner_mid_chunk_checkpoint_resume(monkeypatch, Path(directory))
    finally:
        dist.destroy_process_group()


def test_ddp_runner_periodic_atomic_save_and_resume(tmp_path):
    mp.spawn(_ddp_runner_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)


def test_corpus_runner_carries_optimizer_and_scale(monkeypatch, tmp_path):
    import json
    import sys
    import yaml
    from experiments.heavy_flow import train_physics_chunks as runner
    from force_md.heavy_flow.corpus_plan import build_plan

    config = runtime_config(dropout=0.0)
    config['data'] = dict(seed=0, temporal_block_size=1,
                          same_domain_validation_fraction=0., unseen_domain_validation_fraction=0.)
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    plan = build_plan([(f'data/mdcath_dataset_{d}.h5', 10) for d in ('a', 'b')],
                      repo_id='test/data', revision='fixed', chunk_size=1,
                      validation_fraction=0, test_fraction=0)
    manifest = tmp_path / 'plan.json'
    manifest.write_text(json.dumps(plan))
    data = samples(2)
    class Reader:
        def __init__(self, domains):
            self.index = [(d, '320', '0', i) for d in domains for i in range(2)]
        def close(self):
            pass
    monkeypatch.setattr(runner, '_reader', lambda args, cfg, domains: Reader(domains))
    monkeypatch.setattr(runner, 'HeavyFlowPhysicsFrameDataset', lambda reader, keys, **kwargs: [data[k.frame] for k in keys])
    monkeypatch.setattr(runner.ForceNormalizer, 'fit_from_force_arrays',
                        lambda *a, **k: ForceNormalizer(scale=2., count=18))
    first = tmp_path / 'first.pt'
    second = tmp_path / 'second.pt'
    for i, output in enumerate((first, second)):
        monkeypatch.setattr(runner, 'domain_order', lambda *a, **kw: [plan['shards'][i]['domain']])
        argv = ['train', '--config', str(cfg), '--device', 'cpu', '--chunk-size', '1',
                '--corpus-manifest', str(manifest), '--corpus-chunk-index', str(i),
                '--output', str(output), '--prefetch', '0', '--fit-normalizer']
        if i:
            argv += ['--initialize-from', str(first)]
            monkeypatch.setattr(runner.ForceNormalizer, 'fit_from_force_arrays',
                                lambda *a, **k: pytest.fail('must reuse checkpoint normalizer'))
        monkeypatch.setattr(sys, 'argv', argv)
        runner._run(runner._parse_args())
    one = torch.load(first, weights_only=False)
    two = torch.load(second, weights_only=False)
    assert one['step'] == 2 and two['step'] == 4
    assert one['normalizer'] == two['normalizer']
    assert two['chunk_rotation']['corpus_chunk_index'] == 1
    assert two['chunk_rotation']['corpus_plan_digest'] == plan['digest']
    for key, state in two['optimizer_state']['state'].items():
        assert state['step'] == one['optimizer_state']['state'][key]['step'] + 2
