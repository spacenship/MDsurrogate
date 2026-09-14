#!/usr/bin/env python
"""Stage M1 -- re-score the saved Phase 1.6 arms with the extended metric suite.

    python scripts/evaluate_phase1_6_extended.py \
        --run runs/phase1_6_bounded_seed0 \
        --config configs/phase1_6_bounded.yaml \
        --out runs/phase1_6_extended_metrics_seed0 \
        --device cuda:4

**No weight is touched and no optimizer is constructed.** Each arm's ``last.pt``
is loaded, its sha256 is taken before and after the pass and the two are required
to match, and the model runs in ``eval`` mode under ``torch.no_grad`` over the
same validation manifest the checkpoint was trained against -- asserted against
the checkpoint's own ``provenance.json`` rather than assumed. This mirrors
``scripts/reevaluate_phase1_6.py``, which did the same thing when Stage B's
records were found to be missing ``pair_id``.

Why a fresh inference pass rather than reusing the saved predictions: the saved
artefacts are *metric rows*, not predictions. ``val_records.json`` holds Cα RMSD
and a rotation angle per pair; there is no stored coordinate to compute a dRMSD
or a contact map from. Re-running the frozen checkpoints is the only way to get
them, and it is cheap -- 3,520 pairs per arm.

Output is one ``records.jsonl`` for every arm plus the identity baseline, written
atomically, with the reproducibility manifest beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Must be set before torch initialises cuBLAS, or `use_deterministic_algorithms`
# has nothing to select on the CUDA GEMM path. Harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from force_md.training.transition_module import TransitionTrainer  # noqa: E402
from force_md.transition import CANONICAL_ARMS  # noqa: E402
from force_md.transition.extended_metrics import (  # noqa: E402
    NOT_APPLICABLE,
    ExtendedMetricConfig,
    RecordContext,
    extended_metric_records,
)
from force_md.transition.targets import (  # noqa: E402
    build_transition_target,
    identity_prediction,
)
from train_transition import (  # noqa: E402
    build_configs,
    build_datasets,
    build_extractor,
    make_loader,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except Exception:  # noqa: BLE001 - a missing git must not stop an evaluation
        return ""


def git_state() -> dict:
    """Commit, dirtiness and a hash of the uncommitted patch.

    The working tree is dirty and stays that way -- that is the state the Stage B
    checkpoints were produced in. Hashing the patch is what makes "dirty"
    reproducible instead of merely honest: two runs with the same
    ``source_diff_hash`` saw the same source.
    """
    commit = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "source_diff_hash": hashlib.sha256(diff.encode()).hexdigest(),
        "status_hash": hashlib.sha256(status.encode()).hexdigest(),
        "dirty_files": [line for line in status.splitlines() if line.strip()],
    }


def fix_numerics() -> dict:
    """Pin the float32 knobs that change the answer, and report what they are.

    Two of them matter here and both were checked rather than assumed.

    *TF32.* On an H100 a float32 matmul can silently run in TF32, which has ~10
    bits of mantissa -- a relative error near 1e-3. That is the same order as the
    differences between these arms (0.005-0.05 A on a ~3 A RMSD), so it could
    invent or erase a result. This PyTorch build defaults ``matmul.allow_tf32``
    to False and ``float32_matmul_precision`` to "highest"; both are pinned here
    anyway, because a default is not a guarantee across versions.

    *Atomic scatter.* ``scatter_sum`` is ``Tensor.index_add``, which on CUDA
    accumulates with atomics: the summation order varies between runs, so two
    evaluations of the same batch differ at float32 rounding (~1e-7 relative).
    ``use_deterministic_algorithms`` selects the ordered kernel where one exists.
    ``warn_only=True`` because an op with no deterministic implementation should
    degrade to a warning in an evaluation, not abort a completed experiment.

    What is **not** fixed by this: CPU and CUDA still use different kernels and
    different reduction orders, so a CPU run and a GPU run agree to about 1e-6
    relative, not bitwise. The Stage B numbers being reproduced were produced on
    GPU, so the GPU run is the one that reproduces them; a CPU run is a code-path
    check, and its numbers are not reported.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
    }


def environment(device: str, numerics: dict) -> dict:
    import e3nn
    import numpy

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "e3nn": e3nn.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "device": device,
        "model_dtype": "float32",
        "metric_dtype": "float64 for pairwise distances, Kabsch SVD and angles",
        "numerics": numerics,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":")[1]) if ":" in device else 0
        info["gpu"] = torch.cuda.get_device_name(index)
        info["cuda"] = torch.version.cuda
    return info


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate_arm(
    *,
    canonical: str,
    arm_dir: Path,
    work_dir: Path,
    raw: dict,
    config_path: str,
    device: str,
    metric_config: ExtendedMetricConfig,
    want_identity: bool,
    max_batches: int | None = None,
    log=print,
) -> tuple[list[dict], list[dict], dict]:
    """``(arm_records, identity_records, provenance)`` for one saved checkpoint."""
    provenance = json.loads((arm_dir / "provenance.json").read_text())
    implementation = provenance["arm"]
    checkpoint = arm_dir / "last.pt"
    before = sha256_file(checkpoint)

    pairs, _, _, data = build_configs(yaml.safe_load(open(config_path)), implementation)
    seed = provenance["seed"]
    pairs = type(pairs)(**{**pairs.__dict__, "seed": seed})

    # Manifests go to the **output** directory, not into the Stage B run.
    # ``build_datasets`` writes ``manifest_train.json`` / ``manifest_val.json``
    # wherever it is pointed, and ``runs/phase1_6_bounded_seed0/<arm>/`` already
    # holds the pair that produced these checkpoints. A re-scoring pass has no
    # business overwriting them, even with identical content.
    work_dir.mkdir(parents=True, exist_ok=True)
    _, val_ds, train_manifest, val_manifest = build_datasets(
        pairs, data, str(work_dir), log=lambda *a: None
    )
    if train_manifest.content_hash() != provenance["manifest_hash"]:
        raise SystemExit(
            f"{canonical}: rebuilt manifest {train_manifest.content_hash()[:12]} != "
            f"the one this checkpoint trained on {provenance['manifest_hash'][:12]}. "
            "Refusing to score against different data."
        )
    val_loader = make_loader(
        val_ds, data.get("batch_size", 4), shuffle=False,
        workers=data.get("num_workers", 0), seed=seed,
    )

    extractor = build_extractor(raw, implementation, device)
    _, trainer = TransitionTrainer.load_checkpoint(
        str(checkpoint), extractor, device=device
    )
    model = trainer.module
    model.eval()

    # The probe's own graph, from the checkpoint's own config -- not from a
    # constant in the metric module. A run whose backbone_cutoff differed would
    # otherwise be scored on edges it never had.
    probe_config = model.config
    metric_config = type(metric_config)(**{
        **metric_config.__dict__,
        "spatial_knn": probe_config.residue_knn,
        "spatial_cutoff": probe_config.backbone_cutoff,
    })

    spec = CANONICAL_ARMS.get(canonical)
    context = RecordContext(
        arm=implementation,
        canonical_arm=canonical,
        oracle=bool(spec.oracle) if spec else bool(provenance.get("oracle")),
        seed=seed,
        manifest_hash=provenance["manifest_hash"],
        phase1_checkpoint_hash=provenance.get("phase1_sha256", ""),
        transition_checkpoint_hash=before,
        config_hash=provenance.get("config_hash", ""),
        git_commit=GIT["commit"],
        git_dirty=GIT["dirty"],
        source_diff_hash=GIT["source_diff_hash"],
        extra={"step": int(trainer.step), "split": "val"},
    )
    identity_context = RecordContext(**{
        **context.__dict__,
        "transition_checkpoint_hash": "",
        "extra": {"step": 0, "split": "val"},
    })

    arm_records: list[dict] = []
    identity_records: list[dict] = []
    started = time.time()
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = batch.to(device)
            target = build_transition_target(batch.current, batch.future)
            bundle = (
                trainer.extractor.oracle_bundle(batch.current)
                if model.conditioner.requires_oracle
                else trainer.extractor(batch.current)
            )
            prediction = model(
                batch.current, bundle, history=batch.history, lag_ps=batch.lag_ps
            )
            arm_records.extend(
                extended_metric_records(
                    prediction, target, pairs=batch.pairs,
                    context=context, config=metric_config,
                )
            )
            if want_identity:
                identity_records.extend(
                    extended_metric_records(
                        identity_prediction(target), target, pairs=batch.pairs,
                        context=identity_context, config=metric_config,
                        identity=True,
                    )
                )
    val_ds.close()

    after = sha256_file(checkpoint)
    if before != after:
        raise SystemExit(
            f"{canonical}: the checkpoint file changed during evaluation "
            f"({before[:12]} -> {after[:12]}). Nothing here writes weights, so "
            "something else did; stopping rather than reporting numbers whose "
            "source moved."
        )

    ids = [r["pair_id"] for r in arm_records]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise SystemExit(
            f"{canonical}: {len(ids) - len(set(ids))} duplicate pair_id(s), e.g. "
            f"{duplicates}. A paired analysis keyed on a duplicated id silently "
            "keeps one row per group; refusing to write the file."
        )
    log(
        f"  {len(arm_records)} records, {len(set(ids))} unique ids, "
        f"{len({r['domain_id'] for r in arm_records})} domains, "
        f"{time.time() - started:.0f}s | checkpoint {before[:12]} unchanged"
    )
    return arm_records, identity_records, {
        "canonical_arm": canonical,
        "implementation": implementation,
        "checkpoint_sha256": before,
        "step": int(trainer.step),
        "manifest_hash": provenance["manifest_hash"],
        "val_manifest_hash": val_manifest.content_hash(),
        "phase1_sha256": provenance.get("phase1_sha256"),
        "config_hash": provenance.get("config_hash"),
        "parameter_count": provenance.get("parameter_count"),
        "residue_knn": probe_config.residue_knn,
        "backbone_cutoff": probe_config.backbone_cutoff,
    }


GIT = git_state()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="a Phase 1.6 run directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="output directory for records")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument(
        "--no-identity", action="store_true",
        help="skip the identity baseline rows (they are otherwise always written)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Stage M0 smoke only. Scores a prefix of the loader, which is NOT a "
             "valid result -- the output is written to whatever --out names, so "
             "point it somewhere disposable.",
    )
    args = parser.parse_args()
    numerics = fix_numerics()
    if args.max_batches is not None:
        print(
            f"SMOKE: only the first {args.max_batches} batches will be scored. "
            "These records are not a result.",
            flush=True,
        )

    raw = yaml.safe_load(open(args.config))
    run = Path(args.run)
    arms = args.arms or [
        d for d in sorted(os.listdir(run))
        if d in CANONICAL_ARMS and (run / d / "last.pt").exists()
    ]
    if not arms:
        raise SystemExit(f"no arm checkpoints found under {run}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metric_config = ExtendedMetricConfig()

    records_path = out / "records.jsonl"
    temporary = records_path.with_suffix(".jsonl.partial")
    arm_provenance: list[dict] = []
    counts: dict[str, int] = {}
    identity_written = False

    with open(temporary, "w") as handle:
        for position, canonical in enumerate(arms):
            print(f"\n{'=' * 70}\n{canonical}\n{'=' * 70}", flush=True)
            arm_records, identity_records, provenance = evaluate_arm(
                canonical=canonical,
                arm_dir=run / canonical,
                work_dir=out / "manifests" / canonical,
                raw=raw,
                config_path=args.config,
                device=args.device,
                metric_config=metric_config,
                # The identity baseline depends only on the target, so it is
                # computed once, on the first arm, and reused. Recomputing it per
                # arm would be seven identical copies of the same rows.
                want_identity=(position == 0 and not args.no_identity),
                max_batches=args.max_batches,
            )
            for record in arm_records:
                handle.write(json.dumps(record) + "\n")
            counts[canonical] = len(arm_records)
            arm_provenance.append(provenance)
            if identity_records:
                for record in identity_records:
                    handle.write(json.dumps(record) + "\n")
                counts["identity_baseline"] = len(identity_records)
                identity_written = True

    # Checked **before** the atomic rename, so a run whose arms disagree leaves
    # no `records.jsonl` behind for someone to analyse by mistake.
    sizes = {v for k, v in counts.items() if k != "identity_baseline"}
    if len(sizes) > 1:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"arms produced different record counts: {counts}. Every arm must "
            "score the identical sample set; no records were written."
        )
    temporary.replace(records_path)

    manifest = {
        "stage": "M1 extended re-evaluation",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": str(run),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "records": str(records_path),
        "record_counts": counts,
        "identity_baseline_written": identity_written,
        "arms": arm_provenance,
        "metric_config": dict(metric_config.__dict__),
        "not_applicable": NOT_APPLICABLE,
        "git": GIT,
        "environment": environment(args.device, numerics),
        "retrained": False,
        "optimizer_constructed": False,
        "checkpoints_verified_unchanged": True,
    }
    (out / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str)
    )
    print(
        f"\nwrote {sum(counts.values())} records to {records_path}\n"
        f"manifest -> {out / 'reproducibility_manifest.json'}\n"
        "no weights were changed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
