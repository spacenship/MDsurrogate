"""Bounded real-config GPU smoke; no training checkpoint is overwritten."""
from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.heavy_flow.train_physics import build_stage3_model, load_yaml, train_physics_stream_steps
from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow import ForceLossConfig, ForceNormalizer, HeavyFlowPhysicsFrameDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heavy_flow/stage3.yaml")
    parser.add_argument("--domain", default="1a0rP01")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--esm2-cache-dir", default="esm2_cache")
    parser.add_argument("--normalizer-path", default="outputs/heavy_flow/stage3/chunk_rotation_normalizer.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-resume", action="store_true",
                        help="Also save/reload model + Adam in a temporary directory and run one update")
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("at least two steps: cold and warmed")
    from pathlib import Path
    config = load_yaml(args.config)
    normalizer = ForceNormalizer.from_dict(json.loads(Path(args.normalizer_path).read_text())["normalizer"])
    reader = MdCathDataset(MdCathConfig(
        data_dir=args.data_dir, temperatures=(320,), replicas=(0,), frames_per_trajectory=1,
        esm2_cache_dir=args.esm2_cache_dir, quarantine_path="mdcath_force_quarantine.json",
        coord_quarantine_path="mdcath_coord_quarantine.json"), domains=[args.domain])
    try:
        dataset = HeavyFlowPhysicsFrameDataset(reader, reader.index[:1])
        config_model = build_stage3_model(config, device=args.device)
        opt = torch.optim.AdamW(config_model.parameters(), lr=1e-4)
        cuda = torch.device(args.device).type == "cuda"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        durations = []
        def report(step, loss):
            nonlocal started
            if cuda:
                torch.cuda.synchronize()
            now = time.perf_counter()
            durations.append(now - started)
            print(json.dumps({"event": "benchmark_step", "step": step, "loss": loss,
                              "seconds": durations[-1]}), flush=True)
            started = now
        train_physics_stream_steps(config_model, dataset, opt, normalizer=normalizer,
            loss_config=ForceLossConfig(**config.get("loss", {})), steps=args.steps,
            progress_callback=report, prefetch=1)
        print(json.dumps({"event": "benchmark_complete", "config": args.config, "domain": args.domain,
            "atoms": int(dataset[0].condition.atom_mask.sum()), "steps": args.steps,
            "warm_seconds_per_step": sum(durations[1:]) / len(durations[1:]),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None,
            "note": "one repeated real frame; not a full-dataset OOM or DDP throughput guarantee"}), flush=True)
        if args.check_resume:
            import gc
            import tempfile
            from experiments.heavy_flow.train_physics import save_stage3_checkpoint
            from experiments.heavy_flow.train_physics_chunks import _load_resume
            from force_md.heavy_flow.physics_dataset import PhysicsSplitManifest
            from experiments.heavy_flow.physics_runtime import capture_rng, restore_rng
            rng = capture_rng()
            with tempfile.TemporaryDirectory(prefix="stage3-resume-smoke-") as directory:
                checkpoint = Path(directory) / "checkpoint.pt"
                save_stage3_checkpoint(checkpoint, config_model, normalizer=normalizer,
                    split_manifest=PhysicsSplitManifest(dataset.keys, [], []), config=config,
                    optimizer=opt, step=args.steps, chunk_rotation={"next_chunk": 0})
                del config_model, opt
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                resumed, opt, restored_normalizer, _, restored_step, _ = _load_resume(
                    checkpoint, config, device=args.device, lr=1e-4)
                restore_rng(rng)
                loss = train_physics_stream_steps(resumed, dataset, opt, normalizer=restored_normalizer,
                    loss_config=ForceLossConfig(**config.get("loss", {})), steps=1, step_offset=restored_step)
                print(json.dumps({"event": "benchmark_resume_complete", "loaded_step": restored_step,
                                  "next_step": restored_step + 1, "loss": loss[-1], "device": args.device}), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
