"""Bounded Stage 4 sampling CLI and a small programmatic sampling helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow import HeavyAtomFlowModel, HeavyFlowCondition, FlowConditionBundle
from force_md.heavy_flow.physics_dataset import HeavyFlowPhysicsFrameDataset
from force_md.heavy_flow.history import HistoryConfig
from experiments.heavy_flow.train_flow import (
    load_stage4_checkpoint as _load_stage4_checkpoint,
)

__all__ = ["sample_model", "load_stage4_checkpoint"]


def sample_model(
    model: HeavyAtomFlowModel,
    condition: HeavyFlowCondition | FlowConditionBundle,
    *,
    num_samples: int = 8,
    seed: Optional[int] = 0,
    solver: str = "heun",
    steps: int = 20,
) -> torch.Tensor:
    """Public inference API: no future coordinates or teacher force is needed."""
    return model.sample(condition, num_samples=num_samples, seed=seed, solver=solver, steps=steps)


def load_stage4_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    stage3_checkpoint: str | Path | dict[str, Any] | None = None,
) -> tuple[HeavyAtomFlowModel, dict[str, Any]]:
    model, checkpoint = _load_stage4_checkpoint(
        path,
        device=device,
        stage3_checkpoint=stage3_checkpoint,
    )
    return model, checkpoint["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage3-checkpoint", default=None, help="optional upstream source; architecture compatibility is always enforced")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--esm2-cache-dir", default=None)
    parser.add_argument("--allow-fake-plm", action="store_true")
    parser.add_argument("--frames-per-trajectory", type=int, default=1)
    parser.add_argument("--max-domains", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lag-frames", type=float, default=1.0, help="decoder prediction interval in stored frames (1000 ps/frame for mdCATH)")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="outputs/heavy_flow/stage4_v2/samples.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, _ = load_stage4_checkpoint(
        args.checkpoint,
        device=args.device,
        stage3_checkpoint=args.stage3_checkpoint,
    )
    reader = MdCathDataset(MdCathConfig(
        data_dir=args.data_dir,
        esm2_cache_dir=args.esm2_cache_dir,
        represented_scope="heavy_atom",
        frames_per_trajectory=args.frames_per_trajectory,
        max_domains=args.max_domains,
        allow_fake_plm=args.allow_fake_plm,
        check_pbc=True,
    ))
    dataset = HeavyFlowPhysicsFrameDataset(reader, reader.index, history_config=HistoryConfig.from_config(model.upstream_config))
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError("sample-index is outside the dataset")
    condition = dataset[args.sample_index].condition.to(args.device)
    if args.lag_frames <= 0:
        raise ValueError("lag-frames must be positive")
    condition.lag.fill_(args.lag_frames)
    samples = sample_model(model, condition, num_samples=args.num_samples, seed=args.seed, solver=args.solver, steps=args.steps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples.cpu(), args.output)
    print(json.dumps({"output": args.output, "shape": list(samples.shape), "solver": args.solver, "steps": args.steps}, indent=2))


if __name__ == "__main__":
    main()
