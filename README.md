# force-md — Force-Conditioned Protein MD Ensemble Model

현재 작업 중인 **heavy-flow Stage 1–4 v2**의 아키텍처, 검증 결과, checkpoint
호환 범위와 학습 명령은 [heavy-flow v2 구현 문서](docs/heavy_flow_v2_architecture.md)에
정리되어 있다. 아래 Phase 1/1.5/1.6 설명과 기존 Stage 보고서는 이전 작업 기록이다.

Phase 1: a hierarchical **local-physics** model over an atom / residue /
backbone-frame graph, SE(3)-equivariant at `l_max = 2`, predicting atomic and
residue forces, torques, uncertainty and an invariant energy from mdCATH.

Phase 2 (not implemented) loads these modules unchanged and adds a temporal
stochastic transition on top of `physics_latent`.

```
                     p_theta( q_{t+D} | q_{t-k:t}, sequence, temperature, D )
                                          ^
                     Phase 2 -------------+
                                          |
   Phase 1:  q_t = {B_i, R_i, A_ia}  ->  physics_latent, forces, torques, U
```

---

## Layout

```
src/force_md/
  transition/  Phase 1.5/1.6 probe: conditioners, pair_physics, future_physics,
               arms registry, targets, metrics, frozen Phase 1 extractor
  data/        contracts, CHARMM vocabularies, units, synthetic fixtures,
               collation, PSF parsing, adapters/mdcath.py
  geometry/    residue frames, local coordinates, rigid-motion helpers
  graph/       six typed relations and their edge features
  conditioning/frozen ESM-2 cache, temperature, residue conditioner
  nn/          irreps, radial basis, uvu message blocks, vertical ops, encoder
  physics/     force projection, heads, energy, losses, output contract
  models/      LocalPhysicsModel
  training/    trainer, metrics
configs/       phase1_small.yaml
scripts/       download_mdcath, audit_mdcath_forces, precompute_esm2, train_phase1
examples/      hierarchy_sanity.py
docs/          phase1_hierarchy_and_contracts.md, phase2_interface.md,
               open_questions.md
tests/
```

## Setup

Python 3.11+, PyTorch, e3nn. Verified working set (conda env `md`):

```
torch 2.13.0+cu130   e3nn 0.6.0   numpy 2.4.6   scipy 1.17.1
h5py 3.16.0   transformers 5.15.0   huggingface_hub 1.27.0   pytest 9.1.1
```

```bash
pip install -r requirements.txt
pip install -e . --no-deps
pytest -q
```

The full test suite runs **without any mdCATH shard or ESM-2 checkpoint** —
real-data tests skip themselves. Equivariance tests run in float64 on CPU and are
the slowest part.

## Quick start

```bash
# 1. see the whole pipeline on one synthetic peptide
python examples/hierarchy_sanity.py

# 2. download a reproducible mdCATH subset (~669 GB for 1000 domains)
python scripts/download_mdcath.py --num-domains 1000 --dry-run

# 3. find trajectories whose forces are a copy of the coordinates
python scripts/audit_mdcath_forces.py

# 4. precompute frozen ESM-2 embeddings once (~2.5 GB download, 0.7 GB cache)
python scripts/precompute_esm2.py --dry-run

# 5. train a mini-subset
python scripts/train_phase1.py --config configs/phase1_small.yaml
```

Steps 2–4 report sizes with `--dry-run` before downloading anything.

### Phase 1.5 — the transition probe

Does Phase 1's force-supervised representation help predict structure 1–4 ns
ahead, against a structure-only baseline and an identity ("nothing moves")
reference? Five conditioner arms, one frozen Phase 1 checkpoint, one manifest.

```bash
# plumbing check: 8 domains, 5 arms, ~3 min
python scripts/run_phase1_5_ablation.py --config configs/phase1_5_smoke.yaml

# what each arm would see, without training a step
python scripts/run_phase1_5_ablation.py --config configs/phase1_5_short.yaml --dry-run

# bounded measurement: 120 domains, 6000 steps, one seed (~1.5 h, all five arms)
CUDA_VISIBLE_DEVICES=4 python scripts/run_phase1_5_ablation.py \
  --config configs/phase1_5_short.yaml

# one trained arm, re-evaluated against a saved manifest
python scripts/eval_transition.py --checkpoint runs/phase1_5_short_seed0/physics_latent/last.pt \
  --config configs/phase1_5_short.yaml
```

The full 3-seed experiment costs ~34 GPU-hours; its command, scale and storage
are in [docs/phase1_5_report.md](docs/phase1_5_report.md) §4. Every arm of one
ablation must see the identical manifest, seed, batch order and step budget — the
runner **asserts** this and aborts on a mismatch rather than producing a table
from runs that are merely similar.

Design and contracts: [docs/phase1_5_design.md](docs/phase1_5_design.md).

### Phase 1.6 — the pair-interaction ablation

Phase 1.5 asked whether Phase 1's *node* latent helps. Phase 1.6 asks whether the
**interaction** between two residues does — the per-edge message a message pass
computes and then throws away — and whether it beats a capacity-matched control
that sees the same edges with the physics removed.

```bash
# Stage A: every arm forward/backward, checkpoint, evaluate. ~4 min.
CUDA_VISIBLE_DEVICES=4 python scripts/run_phase1_6_ablation.py \
  --config configs/phase1_6_smoke.yaml

# Stage B: bounded screening, 120 domains, 6000 steps, one seed, seven arms
CUDA_VISIBLE_DEVICES=4 python scripts/run_phase1_6_ablation.py \
  --config configs/phase1_6_bounded.yaml

# statistics and figures: paired on sample id, bootstrapped over domains
python scripts/analyze_phase1_6.py --runs runs/phase1_6_bounded_seed0 \
  --out docs/phase1_6_results_bounded.md --plots docs/figures

# re-evaluate saved arms over the whole validation set, without retraining
python scripts/reevaluate_phase1_6.py --run runs/phase1_6_bounded_seed0 \
  --config configs/phase1_6_bounded.yaml

# Stage C: 3 seeds on GPUs 4,5,6. Only when the screening justifies it.
bash scripts/launch_phase1_6_full.sh
```

#### Stage M — extended metrics on the saved checkpoints

Stage B separated the arms on two numbers, and they disagreed: `P1` beat `P0` on
rotation while losing on Cα RMSD. Stage M re-scores the **same frozen
checkpoints** on pair geometry (dRMSD by sequence separation), contact
*formation* and *breakage*, backbone torsions including omega, and what physical
validity this reconstruction can actually support. It trains nothing and verifies
every checkpoint's sha256 before and after.

```bash
# waits for a free GPU, then runs the tests, the re-scoring and the report
bash scripts/run_phase1_6_extended_when_free.sh

# or by hand
python scripts/evaluate_phase1_6_extended.py --run runs/phase1_6_bounded_seed0 \
  --config configs/phase1_6_bounded.yaml \
  --out runs/phase1_6_extended_metrics_seed0 --device cuda:4
python scripts/analyze_phase1_6_extended.py \
  --records runs/phase1_6_extended_metrics_seed0/records.jsonl \
  --out docs/phase1_6_results_extended.md --plots docs/figures
```

The metric suite lives in
[src/force_md/transition/extended_metrics.py](src/force_md/transition/extended_metrics.py)
and extends `metrics.py` rather than replacing it: `ca_rmsd` and the rotation
geodesic keep the Stage B definitions bit for bit, which is asserted by test.
Audit and the decisions taken:
[docs/phase1_6_extended_metrics_audit.md](docs/phase1_6_extended_metrics_audit.md).

Arms carry two names: the canonical role (`P1_pair_physics_frozen`) and the
registered conditioner (`pair_physics`). Both appear in every row —
[src/force_md/transition/arms.py](src/force_md/transition/arms.py) is the mapping.

The pair latent is exposed by an **additive** interface:
`LocalPhysicsModel.forward(..., return_pair_messages=True)` populates
`Phase1Output.pair`, and the default path is bitwise what it was, so every Phase 1
checkpoint still loads and Phase 1.5's numbers stand.

Audit of what actually exists versus what the plan assumed:
[docs/phase1_6_audit.md](docs/phase1_6_audit.md).
Results: [docs/phase1_6_report.md](docs/phase1_6_report.md),
full tables in [docs/phase1_6_results_bounded.md](docs/phase1_6_results_bounded.md).

## What the model does

```
atom interaction        3-body, uvu tensor product, 5.0 A cutoff
atom -> residue pool    equivariant, size-normalised
residue -> backbone     pooled irreps + PLM and temperature scalars
backbone interaction    sequence +-1/+-2 and CA kNN(16)
backbone -> residue     gated global context
residue -> atom         gated broadcast
                        -> force / torque / uncertainty / energy heads
```

Node features are `64x0e + 16x1o + 8x2e` at every level, in the global frame.
Uncertainties are predicted in the **residue-local** frame, where a diagonal
covariance is meaningful.

## Things worth knowing before changing anything

* **Reflection is not a symmetry.** Chirality enters through the residue-frame
  local coordinates; mirroring changes the `l=0` features by 1.90 while a proper
  rotation leaves them unchanged to `7e-15`.
* **The neighbour list is not differentiable.** Gradients flow through edge
  vectors and distances only.
* **Ground-truth forces are labels, never inputs.** Asserted by test.
* **The full effective force is not a potential gradient.** The conservative and
  residual parts are separate tensors and are never equated.
* **mdCATH has no energy label and no per-frame timestamp.** Neither is invented.
* **5 of 1000 published domains ship corrupt force labels** (`forces == coords`,
  replicas 0–3). They are masked per trajectory, not dropped.
* **Use `uvu`, not fully-connected, tensor products.** The `uvw` version needs
  8064 weights per edge and exhausted an 80 GB card; `uvu` needs 288.
* **Never seed an equivariant channel with a raw Ångström quantity.** Each block's
  body-order-3 term squares its features, so stacked blocks compound magnitude as
  `|h|^(2^depth)`. Feeding a raw CA displacement took the peak gradient norm to
  1.4e11 at three blocks; feeding its direction plus a bounded magnitude encoding
  takes it to 3.8. Phase 1 is safe only because it interleaves its two backbone
  blocks with pooling layers instead of stacking them.

Full detail: [docs/phase1_hierarchy_and_contracts.md](docs/phase1_hierarchy_and_contracts.md).
Open questions: [docs/open_questions.md](docs/open_questions.md).
Phase 2 contract: [docs/phase2_interface.md](docs/phase2_interface.md).

## Status

Phase 1 and Phase 1.5 complete; Phase 1.6 screened. **581 tests pass** (~4 min;
real-data tests skip themselves if `data/` is empty).

**Phase 1.6 screening (1 seed, 30 held-out domains, 1,760 pairs per lag).** The
pair *architecture* beats the pair *physics*: a geometry-only control with the
same edges and 26 fewer parameters is the best arm on Cα RMSD at both lags, while
Phase 1's pair message beats it on rotation (+0.27% at 1 ns) and loses on RMSD
(−0.34%). The oracle again fails to beat structure-and-history, reproducing the
Phase 1.5 ceiling on a different manifest. Stage C not started — the gate did not
fire cleanly. [docs/phase1_6_report.md](docs/phase1_6_report.md).

**Stage M extended metrics (2026-08-27).** The seven Stage B checkpoints
re-scored on 3,520 pairs without retraining, reproducing the Stage B Cα RMSD and
rotation exactly. The result is not about the arms: **every trained arm is worse
than the identity baseline on all ten physical-validity cells** — peptide C–N
bond 4–5×, backbone angle 3.5–5×, consecutive Cα distance 5–7×, Cα clash rate
361–1651×. The probe predicts a rigid update per residue frame, each frame moves
independently, and the bonds *between* residues absorb the error, while the loss
carries `clash: 0.0` and no bond term. Among the arms, `P1`'s rotation gain over
`P0` (+0.10° at both lags) comes with **worse formed-contact F1 at both lags** —
an orientation-only effect that does not clear a Stage C gate.
[docs/phase1_6_results_extended.md](docs/phase1_6_results_extended.md).

**Phase 1.5 answered its question.** 3 seeds × 5 arms × 40,000 steps, evaluated on
181 held-out domains / 72,080 pairs. Full detail in
[docs/phase1_5_report.md](docs/phase1_5_report.md).

| arm | 1 ns Cα RMSD | vs identity | vs `structure_only` |
|---|---|---|---|
| identity ("nothing moves") | 2.9665 Å | — | — |
| `structure_only` | 2.7546 Å | −7.1% | — |
| `force_torque` | 2.7549 Å | −7.1% | no effect (sign flips across seeds) |
| `physics_latent` | **2.7495 Å** | −7.3% | −0.18% (3.6 sd), rotation −0.74% (4.7 sd) |
| `force_pattern_shape` | 2.7520 Å | −7.2% | beats A, does not beat `physics_latent` |
| `oracle_force` | 2.7520 Å | −7.2% | beats A, does not beat `physics_latent` |

Phase 1's learned latent is the best conditioner tested — it beats the explicit
force-moment features *and* the ground-truth-force oracle. But it buys **under 1%**,
below the pre-registered bar, and the oracle proves that is the ceiling: perfect
instantaneous force knowledge is worth ~0.2° of frame rotation at 1–4 ns. The 7–9%
that every arm takes off the identity baseline comes from structure and history,
not from force.

Mini-subset smoke result (40 domains, split by domain, 500
steps — **not** a performance claim):

| target | RMSE | zero baseline | relative | angular error |
|---|---|---|---|---|
| atom force | 26.39 | 46.69 | 0.566 | 37.8° |
| residue force | 46.02 | 60.4 | 0.762 | 48.9° |
| torque | 120.65 | 120.9 | 0.998 | — |

Atom-level force is genuinely learned on held-out proteins; torque is not learned
at 500 steps.
