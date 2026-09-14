"""Evaluate a Stage 3 atom-force checkpoint with controls and domain splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from force_md.data.adapters.mdcath import MdCathConfig, MdCathDataset
from force_md.heavy_flow.force_losses import ForceNormalizer
from force_md.heavy_flow.physics_dataset import HeavyFlowPhysicsFrameDataset, PhysicsSplitManifest
from force_md.heavy_flow.physics_metrics import (
    compute_force_metrics,
    mean_force_baseline,
    unconditional_mean_variance_baseline,
    zero_force_baseline,
)
from .train_physics import build_stage3_model


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataset: Iterable[Any], *, normalizer: Optional[ForceNormalizer] = None, train_force: Optional[torch.Tensor] = None) -> dict[str, Any]:
    samples = list(dataset)
    if not samples:
        raise ValueError("cannot evaluate an empty Stage 3 split")
    predictions, logvars, targets, masks, positions, residue_ids, domains = [], [], [], [], [], [], []
    model.eval()
    for sample in samples:
        output = model(sample.condition)
        predictions.append(output.force_mean)
        logvars.append(output.force_logvar)
        targets.append(sample.targets.force_current)
        masks.append(sample.targets.force_mask if sample.targets.force_mask is not None else sample.condition.atom_mask)
        positions.append(sample.condition.current_positions)
        residue_ids.append(sample.condition.atom_to_residue)
        domains.append(str(sample.condition.provenance.domain[0]) if sample.condition.provenance and sample.condition.provenance.domain else "unknown")
    mean = torch.cat(predictions)
    logvar = torch.cat(logvars)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    result = compute_force_metrics(mean, target, force_logvar=logvar, force_mask=mask, positions=torch.cat(positions), atom_to_residue=torch.cat(residue_ids), domains=domains, normalizer=normalizer)
    if train_force is not None:
        train_mask = torch.ones(train_force.shape[:2], dtype=torch.bool, device=train_force.device)
        result["controls"] = {
            "zero": zero_force_baseline(target, force_mask=mask),
            "train_mean": mean_force_baseline(train_force, target, train_mask=train_mask, force_mask=mask),
            "unconditional_mean_variance": unconditional_mean_variance_baseline(train_force, target, train_mask=train_mask, force_mask=mask, normalizer=normalizer),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--esm2-cache-dir", default=None)
    parser.add_argument("--allow-fake-plm", action="store_true")
    parser.add_argument("--max-domains", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from force_md.heavy_flow.checkpoint import validate_checkpoint
    from force_md.heavy_flow.history import HistoryConfig
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    validate_checkpoint(checkpoint)
    config = checkpoint["config"]
    model = build_stage3_model(config, device=args.device)
    model.load_state_dict(checkpoint["model_state"])
    normalizer = ForceNormalizer.from_dict(checkpoint["normalizer"])
    manifest = PhysicsSplitManifest(**checkpoint["split_manifest"])
    reader = MdCathDataset(MdCathConfig(data_dir=args.data_dir, max_domains=args.max_domains, esm2_cache_dir=args.esm2_cache_dir, allow_fake_plm=args.allow_fake_plm))
    train_data = HeavyFlowPhysicsFrameDataset(reader, manifest.train, history_config=HistoryConfig.from_config(config))
    train_force = torch.cat([train_data[i].targets.force_current for i in range(len(train_data))])
    outputs = {}
    for name, keys in (("same_domain", manifest.same_domain_validation), ("unseen_domain", manifest.unseen_domain_validation)):
        dataset = HeavyFlowPhysicsFrameDataset(reader, keys, history_config=HistoryConfig.from_config(config))
        outputs[name] = evaluate_model(model, dataset, normalizer=normalizer, train_force=train_force)
    destination = args.output or str(Path(args.checkpoint).with_name("stage3_metrics.json"))
    Path(destination).write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": destination, "splits": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
