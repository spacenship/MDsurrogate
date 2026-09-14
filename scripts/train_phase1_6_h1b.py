#!/usr/bin/env python
"""H1b -- train a small SE(3) refiner on top of a frozen Phase 1.6 arm.

    # one-batch smoke, nothing is claimed from it
    python scripts/train_phase1_6_h1b.py --config configs/phase1_6_h1b_smoke.yaml \
        --run runs/phase1_6_bounded_seed0 --out runs/phase1_6_h1b_smoke --device cuda:1

    # tiny overfit: can the refiner move at all, on 1 batch for 200 steps
    python scripts/train_phase1_6_h1b.py --config configs/phase1_6_h1b_smoke.yaml \
        --run runs/phase1_6_bounded_seed0 --out runs/phase1_6_h1b_overfit \
        --overfit-batches 1 --steps 200 --device cuda:1

Stage M found the Phase 1.6 arms physically worse than "nothing moves" on every
validity metric -- peptide C-N bond 4-5x, backbone angles 3.5-5x, Ca clash rate
361-1651x -- because the probe predicts an independent rigid update per residue
and its loss has ``clash: 0.0`` and no bond term. Those are all **inter-residue**
quantities. H1b is the smallest thing that can address them: a per-residue
residual rotation and translation, conditioned on the residue and its two
sequence neighbours, composed onto the coarse frames.

H0 then measured how much room there is. Replacing the predicted frames with the
true ones drops heavy-atom RMSD by 2.449 A (63.9%) while replacing the side-chain
conformer drops it by 0.187 A (4.9%) -- a 13:1 split, and still 7:1 on side-chain
RMSD alone. The frame is where the error is, which is why H1b runs before H1a.

**The coarse arm is frozen and never written.** Its checkpoint is opened
read-only and its sha256 is compared before and after. The optimiser is given the
refiner's parameters only, and the arm is put in ``eval()`` under ``no_grad``, so
there is no path by which a gradient could reach it.

**The shortcut this must not take.** The identity baseline has near-perfect
peptide geometry because it *is* a real MD frame. A refiner rewarded for physical
geometry can therefore score well by simply undoing the coarse prediction. Three
things guard against it: the primary transition loss is kept in the objective at
full weight, the correction magnitude is logged every step, and the run reports
how far the refined prediction sits from the coarse one. A refiner whose
correction magnitude grows while its transition loss degrades is reverting, not
learning, and the log says so.

**What the geometry loss is measured against.** There is no table of ideal bond
lengths or angles in this repository and inventing one was forbidden. There is
something better: the current frame is a real MD snapshot, so its own peptide
geometry is physical by construction. The regulariser asks the refined structure
to keep ``|C_i - N_{i+1}|`` and the two angles across that bond near the values
measured at time ``t``. Reading the current structure is not leakage; it is the
model's input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.heavy.refiner import (  # noqa: E402
    BackboneFrameConstraintRefiner,
    RefinerConfig,
    compose,
    correction_magnitude,
    current_peptide_geometry,
    peptide_geometry_loss,
    prediction_from_frames,
    reconstruct_from_frames,
    refiner_feature_dim,
    refiner_features,
    sequence_separation_exclusion,
    soft_overlap_penalty,
)
from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import CANONICAL_ARMS  # noqa: E402
from force_md.transition.losses import (  # noqa: E402
    TransitionLossWeights,
    transition_loss,
)
from force_md.transition.targets import (  # noqa: E402
    apply_prediction,
    build_transition_target,
)
from evaluate_phase1_6_extended import (  # noqa: E402
    environment,
    fix_numerics,
    git_state,
    sha256_file,
)
from train_transition import (  # noqa: E402
    build_configs,
    build_datasets,
    build_extractor,
    make_loader,
)

ROOT = Path(__file__).resolve().parents[1]
GIT = git_state()

#: Backbone atoms carry a Bondi radius of 1.55 (N), 1.70 (C), 1.70 (C'). The
#: training penalty uses carbon's for all three rather than a per-atom table:
#: it is the larger of the two, so the penalty is conservative, and the *reported*
#: overlap always comes from the per-element PSF-based H0 metric instead.
TRAINING_BACKBONE_RADIUS = 1.70


def log(message: str) -> None:
    print(message, flush=True)


def coarse_forward(trainer, coarse, batch):
    """The frozen arm's prediction. Always under ``no_grad``."""
    with torch.no_grad():
        bundle = (
            trainer.extractor.oracle_bundle(batch.current)
            if coarse.conditioner.requires_oracle
            else trainer.extractor(batch.current)
        )
        return coarse(
            batch.current, bundle, history=batch.history, lag_ps=batch.lag_ps
        )


@torch.no_grad()
def evaluate(
    refiner, trainer, coarse, loader, weights, settings, device, max_batches: int
) -> dict:
    """Mean of every logged quantity over held-out batches.

    Training loss cannot distinguish a refiner that learned the geometry from one
    that memorised the training pairs, and the overfit probe deliberately does
    the latter. Nothing about H1b should be reported without this.

    Sample-weighted by residue count, so a batch of long domains does not count
    the same as a batch of short ones.
    """
    was_training = refiner.training
    refiner.eval()
    totals: dict[str, float] = {}
    weight = 0.0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        target = build_transition_target(batch.current, batch.future)
        residues = float(target.valid.sum())
        if residues == 0:
            continue
        _, parts = refined_step(
            refiner,
            coarse_forward(trainer, coarse, batch),
            target,
            batch.lag_ps / 1000.0,
            weights,
            settings,
        )
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * residues
        weight += residues
    if was_training:
        refiner.train()
    if not weight:
        raise SystemExit("validation produced no valid residue")
    return {key: value / weight for key, value in totals.items()}


# --------------------------------------------------------------------------
# one forward pass
# --------------------------------------------------------------------------


def refined_step(
    refiner: BackboneFrameConstraintRefiner,
    coarse_prediction,
    target,
    lag_ns: torch.Tensor,
    weights: TransitionLossWeights,
    settings: dict,
) -> tuple[torch.Tensor, dict]:
    """Total loss and its parts for one batch.

    The coarse prediction arrives already detached from the frozen arm; every
    gradient below flows to the refiner and nowhere else.
    """
    features = refiner_features(coarse_prediction, target, lag_ns)
    correction = refiner(features)

    coarse_origin, coarse_rotation = apply_prediction(coarse_prediction, target)
    origin, rotation = compose(coarse_origin, coarse_rotation, correction)
    refined = prediction_from_frames(origin, rotation, target)

    # -- 1. the original objective, unchanged -------------------------------
    primary, components = transition_loss(refined, target, weights=weights)

    # -- 2. peptide geometry, against the current frame's own values --------
    n_refined, ca_refined, c_refined = reconstruct_from_frames(
        origin, rotation, target.local_n, target.local_c
    )
    n_current, ca_current, c_current = reconstruct_from_frames(
        target.current_ca, target.current_frames.rotation, target.local_n, target.local_c
    )
    geometry = peptide_geometry_loss(
        current_peptide_geometry(n_refined, ca_refined, c_refined, target.following),
        current_peptide_geometry(n_current, ca_current, c_current, target.following),
        bond_weight=settings["bond_weight"],
        angle_weight=settings["angle_weight"],
    )
    geometry_total = geometry["bond"] + geometry["angle_ca_c_n"] + geometry["angle_c_n_ca"]

    # -- 3. steric overlap on the reconstructed backbone --------------------
    backbone = torch.cat([n_refined, ca_refined, c_refined], dim=0)
    residue = torch.arange(target.valid.numel(), device=backbone.device).repeat(3)
    graph = target.residue_batch_index.repeat(3)
    excluded = sequence_separation_exclusion(
        residue, graph, min_separation=settings["overlap_min_separation"]
    )
    overlap = soft_overlap_penalty(
        backbone,
        torch.full_like(backbone[:, 0], TRAINING_BACKBONE_RADIUS),
        excluded,
        target.valid.repeat(3),
        tolerance=settings["overlap_tolerance"],
    )

    total = (
        settings["primary_weight"] * primary
        + settings["geometry_weight"] * geometry_total
        + settings["overlap_weight"] * overlap
    )

    with torch.no_grad():
        magnitude = correction_magnitude(correction)
        valid = target.valid

        def norm_summary(values: torch.Tensor) -> dict[str, float]:
            values = values[valid].to(torch.float64)
            return {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90": float(torch.quantile(values, 0.90)),
                "p95": float(torch.quantile(values, 0.95)),
                "p99": float(torch.quantile(values, 0.99)),
                "max": float(values.max()),
            }

        raw_translation = norm_summary(magnitude["raw_translation_norm"])
        clipped_translation = norm_summary(
            magnitude["correction_translation_norm"]
        )
        raw_rotation = norm_summary(magnitude["raw_rotation_deg"])
        clipped_rotation = norm_summary(magnitude["correction_rotation_deg"])
        coarse_loss, coarse_components = transition_loss(
            coarse_prediction, target, weights=weights
        )
        parts = {
            "total": float(total),
            "primary": float(primary),
            "primary_coarse": float(coarse_loss),
            "geometry": float(geometry_total),
            "geometry_bond": float(geometry["bond"]),
            "geometry_angle_ca_c_n": float(geometry["angle_ca_c_n"]),
            "geometry_angle_c_n_ca": float(geometry["angle_c_n_ca"]),
            "overlap": float(overlap),
            "correction_translation_mean_a": float(
                magnitude["correction_translation_norm"][valid].mean()
            ),
            "correction_translation_max_a": float(
                magnitude["correction_translation_norm"][valid].max()
            ),
            "correction_rotation_mean_deg": float(
                magnitude["correction_rotation_deg"][valid].mean()
            ),
            "correction_rotation_max_deg": float(
                magnitude["correction_rotation_deg"][valid].max()
            ),
            "raw_translation_norm_mean_a": raw_translation["mean"],
            "raw_translation_norm_median_a": raw_translation["median"],
            "raw_translation_norm_p90_a": raw_translation["p90"],
            "raw_translation_norm_p95_a": raw_translation["p95"],
            "raw_translation_norm_p99_a": raw_translation["p99"],
            "raw_translation_norm_max_a": raw_translation["max"],
            "clipped_translation_norm_mean_a": clipped_translation["mean"],
            "clipped_translation_norm_max_a": clipped_translation["max"],
            "raw_rotation_norm_mean_deg": raw_rotation["mean"],
            "raw_rotation_norm_median_deg": raw_rotation["median"],
            "raw_rotation_norm_p90_deg": raw_rotation["p90"],
            "raw_rotation_norm_p95_deg": raw_rotation["p95"],
            "raw_rotation_norm_p99_deg": raw_rotation["p99"],
            "raw_rotation_norm_max_deg": raw_rotation["max"],
            "clipped_rotation_norm_mean_deg": clipped_rotation["mean"],
            "clipped_rotation_norm_max_deg": clipped_rotation["max"],
            # What fraction of residues sits at the cap. The overfit probe hit
            # both caps exactly, which on one memorised batch says nothing -- a
            # model allowed a bound will use it. Whether the cap is *binding*
            # during real training is a different question, and this is the
            # number that answers it, rather than a guess made in advance.
            "translation_at_cap_fraction": float(
                (
                    magnitude["correction_translation_norm"][valid]
                    >= refiner.config.max_translation * (1.0 - 1e-6)
                ).to(torch.float64).mean()
            ),
            "rotation_at_cap_fraction": float(
                (
                    magnitude["correction_rotation_deg"][valid]
                    >= math.degrees(refiner.config.max_rotation_rad) * (1.0 - 1e-6)
                ).to(torch.float64).mean()
            ),
            # Physical units, so the log says whether the refiner moved the
            # metric or only the loss. Keyed by name because a typo here would
            # silently log NaN and the trade-off would go unmeasured -- which is
            # what happened on the first H1b probe.
            "translation_rmse_a": components["translation_rmse_angstrom"],
            "translation_rmse_a_coarse": coarse_components["translation_rmse_angstrom"],
            "rotation_error_deg": components["rotation_error_deg"],
            "rotation_error_deg_coarse": coarse_components["rotation_error_deg"],
        }
    return total, parts


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="the frozen Phase 1.6 run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--arm", default=None, help="canonical arm; else the config's")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--overfit-batches", type=int, default=None,
        help="cycle this many batches forever -- a capacity probe, not a result",
    )
    parser.add_argument("--resume", default=None, help="refiner checkpoint to continue")
    args = parser.parse_args()

    numerics = fix_numerics()
    raw = yaml.safe_load(open(args.config))
    settings = dict(raw["h1b"])
    canonical = args.arm or settings.pop("arm")
    settings.pop("arm", None)
    steps = args.steps if args.steps is not None else settings.pop("steps")
    settings.pop("steps", None)

    run = Path(args.run)
    arm_dir = run / canonical
    if not (arm_dir / "last.pt").exists():
        raise SystemExit(f"no checkpoint at {arm_dir / 'last.pt'}")
    checkpoint = arm_dir / "last.pt"
    before = sha256_file(checkpoint)

    provenance = json.loads((arm_dir / "provenance.json").read_text())
    implementation = provenance["arm"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs, _, _, data = build_configs(yaml.safe_load(open(args.config)), implementation)
    seed = provenance["seed"]
    pairs = type(pairs)(**{**pairs.__dict__, "seed": seed})
    train_ds, val_ds, train_manifest, _ = build_datasets(
        pairs, data, str(out / "manifests"), log=lambda *a: None
    )
    if train_manifest.content_hash() != provenance["manifest_hash"]:
        raise SystemExit(
            f"rebuilt manifest {train_manifest.content_hash()[:12]} != the "
            f"checkpoint's {provenance['manifest_hash'][:12]}. Refusing: H1b must "
            "sit on the split its coarse arm was trained against."
        )
    loader = make_loader(
        train_ds, data.get("batch_size", 4), shuffle=True,
        workers=data.get("num_workers", 0), seed=seed,
    )
    # Held out, unshuffled, and the same pairs Stage M and H0 scored, so an H1b
    # number can be put beside theirs without a caveat about the sample.
    val_loader = None if args.overfit_batches else make_loader(
        val_ds, data.get("batch_size", 4), shuffle=False,
        workers=data.get("num_workers", 0), seed=seed,
    )

    extractor = build_extractor(raw, implementation, args.device)
    _, trainer = TransitionTrainer.load_checkpoint(
        str(checkpoint), extractor, device=args.device
    )
    coarse = trainer.module
    coarse.eval()
    for parameter in coarse.parameters():
        parameter.requires_grad_(False)

    refiner = BackboneFrameConstraintRefiner(
        refiner_feature_dim(),
        RefinerConfig(**settings.pop("refiner", {})),
    ).to(args.device)
    optimiser = torch.optim.AdamW(
        refiner.parameters(),
        lr=settings.pop("learning_rate"),
        weight_decay=settings.pop("weight_decay", 0.0),
    )
    # Default to the *train* block's weights, i.e. literally the ones the frozen
    # arm was trained with. An H1b-specific set here would make "the primary loss
    # got worse" mean something different from the number the arm was optimised
    # against, which is the one comparison this stage depends on.
    weights = TransitionLossWeights(
        **settings.pop("loss_weights", raw["train"]["loss_weights"])
    )

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        refiner.load_state_dict(state["refiner"])
        optimiser.load_state_dict(state["optimiser"])
        start_step = int(state["step"])
        log(f"resumed from {args.resume} at step {start_step}")

    log(
        f"H1b on {canonical} ({implementation}), refiner "
        f"{sum(p.numel() for p in refiner.parameters()):,} parameters, "
        f"coarse {sum(p.numel() for p in coarse.parameters()):,} frozen"
    )
    if args.overfit_batches:
        log(
            f"OVERFIT: cycling {args.overfit_batches} batch(es). This is a "
            "capacity probe. It is not a result and must not be reported as one."
        )

    history: list[dict] = []
    validation: list[dict] = []
    best: dict = {"total": float("inf")}
    fixed = []
    started = time.time()
    step = start_step
    while step < steps:
        if args.overfit_batches:
            # Draw the fixed batches once, then cycle exactly those. Falling
            # through to the loader here would quietly train on the whole split
            # and report it as an overfit probe.
            if not fixed:
                for batch in loader:
                    fixed.append(batch.to(args.device))
                    if len(fixed) >= args.overfit_batches:
                        break
                if not fixed:
                    raise SystemExit("the loader produced no batch to overfit")
            source = fixed
        else:
            source = loader
        for batch in source:
            if step >= steps:
                break
            if not args.overfit_batches:
                batch = batch.to(args.device)
            target = build_transition_target(batch.current, batch.future)
            total, parts = refined_step(
                refiner,
                coarse_forward(trainer, coarse, batch),
                target,
                batch.lag_ps / 1000.0,
                weights,
                settings,
            )
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                refiner.parameters(), settings["grad_clip"]
            )
            optimiser.step()

            step += 1
            parts.update(step=step, grad_norm=float(grad_norm))
            history.append(parts)
            if step % settings["log_every"] == 0 or step == 1:
                log(
                    f"  step {step:5d} | total {parts['total']:.4f} | "
                    f"primary {parts['primary']:.4f} (coarse {parts['primary_coarse']:.4f}) | "
                    f"geom {parts['geometry']:.5f} | overlap {parts['overlap']:.5f} | "
                    f"|dt| {parts['correction_translation_mean_a']:.4f} A | "
                    f"|dR| {parts['correction_rotation_mean_deg']:.3f} deg"
                )
            if (
                val_loader is not None
                and settings.get("eval_every")
                and step % settings["eval_every"] == 0
            ):
                scores = evaluate(
                    refiner, trainer, coarse, val_loader, weights, settings,
                    args.device, settings.get("eval_batches", 40),
                )
                scores["step"] = step
                validation.append(scores)
                torch.save(
                    {
                        "refiner": refiner.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "step": step,
                        "refiner_config": asdict(refiner.config),
                        "canonical_arm": canonical,
                        "coarse_checkpoint_sha256": before,
                    },
                    out / "refiner_last.pt",
                )
                if scores["total"] < best["total"]:
                    best = dict(scores)
                    torch.save(
                        {"refiner": refiner.state_dict(), "step": step,
                         "refiner_config": asdict(refiner.config),
                         "canonical_arm": canonical,
                         "coarse_checkpoint_sha256": before},
                        out / "refiner_best.pt",
                    )
                log(
                    f"  VAL  {step:5d} | total {scores['total']:.4f} | "
                    f"primary {scores['primary']:.4f} "
                    f"(coarse {scores['primary_coarse']:.4f}, "
                    f"{scores['primary'] - scores['primary_coarse']:+.4f}) | "
                    f"geom {scores['geometry']:.5f} | overlap {scores['overlap']:.5f} | "
                    f"rot {scores['rotation_error_deg']:.3f} deg "
                    f"(coarse {scores['rotation_error_deg_coarse']:.3f}) | "
                    f"at-cap dt {scores['translation_at_cap_fraction']:.3f} "
                    f"dR {scores['rotation_at_cap_fraction']:.3f}"
                    + ("  <- best" if scores is best or scores == best else "")
                )

    after = sha256_file(checkpoint)
    if before != after:
        raise SystemExit(
            f"the frozen arm changed during H1b training "
            f"({before[:12]} -> {after[:12]}). Stopping."
        )

    torch.save(
        {
            "refiner": refiner.state_dict(),
            "optimiser": optimiser.state_dict(),
            "step": step,
            "refiner_config": asdict(refiner.config),
            "canonical_arm": canonical,
            "coarse_checkpoint_sha256": before,
        },
        out / "refiner_last.pt",
    )
    (out / "history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in history) + "\n"
    )
    if validation:
        (out / "validation.jsonl").write_text(
            "\n".join(json.dumps(row) for row in validation) + "\n"
        )
    first, last = history[0], history[-1]
    (out / "reproducibility_manifest.json").write_text(json.dumps({
        "stage": "H1b backbone frame constraint refiner",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "is_smoke": bool(args.overfit_batches) or steps <= settings.get("smoke_steps", 50),
        "overfit_batches": args.overfit_batches,
        "steps": step,
        "canonical_arm": canonical,
        "implementation": implementation,
        "coarse_checkpoint_sha256_before": before,
        "coarse_checkpoint_sha256_after": after,
        "coarse_weights_unchanged": before == after,
        "manifest_hash": provenance["manifest_hash"],
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "refiner_parameters": sum(p.numel() for p in refiner.parameters()),
        "feature_dim": refiner_feature_dim(),
        "loss_weights": weights.as_dict(),
        "settings": settings,
        "first_step": first,
        "last_step": last,
        "validation": validation,
        "best_validation": best if validation else None,
        "wall_seconds": round(time.time() - started, 1),
        "git": GIT,
        "numerics": numerics,
        "environment": environment(args.device, numerics),
    }, indent=2, default=str))

    log(
        f"\nstep {first['step']} -> {last['step']}: "
        f"total {first['total']:.4f} -> {last['total']:.4f}, "
        f"primary {first['primary']:.4f} -> {last['primary']:.4f} "
        f"(coarse {last['primary_coarse']:.4f}), "
        f"geometry {first['geometry']:.5f} -> {last['geometry']:.5f}, "
        f"overlap {first['overlap']:.5f} -> {last['overlap']:.5f}"
    )
    log(
        f"correction at the end: {last['correction_translation_mean_a']:.4f} A / "
        f"{last['correction_rotation_mean_deg']:.3f} deg mean"
    )
    if last["primary"] > last["primary_coarse"]:
        log(
            "NOTE: the refined transition loss is WORSE than the frozen coarse "
            "arm's. The refiner is paying transition accuracy for geometry; that "
            "is the trade this stage exists to measure, not a bug, but it must be "
            "reported as a cost and not omitted."
        )
    log("the frozen arm's weights were not changed")
    val_ds.close()
    train_ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
