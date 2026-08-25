# force-md — Force-Conditioned Protein MD Ensemble Model

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

Phase 1 and Phase 1.5 complete. **533 tests pass** (~4 min; real-data tests skip
themselves if `data/` is empty).

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
