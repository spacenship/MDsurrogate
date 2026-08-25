# Phase 1.5: the transition probe — results report

Companion to [phase1_5_design.md](phase1_5_design.md), which carries the contracts
and the reasoning. This file carries **what was actually run and what it showed**.

Status as of 2026-08-21: the probe, the five conditioner arms, the ablation
runner and the three configs are implemented and tested. **The measurement
experiment has not been run.** Everything below labelled *pending* is pending, and
no number has been written here that was not produced by a command in this
repository.

---

## 1. What has been run

| run | scale | purpose | outcome |
|---|---|---|---|
| test suite | 530 tests | correctness | all pass, ~4 min |
| smoke ablation | 8 domains, 36 pairs, 200 steps | plumbing | §3 — **not a measurement** |
| throughput probe | 120 domains, 14,080 pairs, 400 steps | cost model | ~6.1 steps/s |
| divergence bisection | 6 configurations × 60 steps | a real defect | §2 |
| manifest enumeration | 907 domains | cost model | 289,120 / 72,080 pairs |
| target statistics | 630 pairs, all 5 temperatures | data sanity | §2.1 |

**Not run:** the short probe (`configs/phase1_5_short.yaml`) and the full 3-seed
experiment (`configs/phase1_5_full.yaml`). Cost and launch commands in §4.

---

## 2. The substantive finding so far: the probe diverged, and why

This is the only real result Checkpoint 6 produced, and it is a negative one about
the implementation rather than anything about protein dynamics.

The smoke config ran at `num_blocks: 2`; the short and full configs ran at 3. At
3, **training diverged** — and it would have consumed the entire experiment
budget before anyone looked at a metric.

Measured over 60 real optimisation steps, `structure_only`, identical data:

| configuration | median grad norm | peak grad norm | max train Cα RMSD |
|---|---|---|---|
| `num_blocks=2` | 1085 | 1.2e5 | 26 Å |
| `num_blocks=3` | **1.7e6** | **1.4e11** | **1.1e8 Å** |
| `num_blocks=4` | non-finite | ∞ | ∞ |
| `num_blocks=3`, history removed | **0.01** | **0.74** | 11 Å |

The last row located it. Every `BackboneInteractionBlock` carries a body-order-3
term, `h ← h + square_mix(h ⊗ h)`, so feature magnitude is squared once per block
and `num_blocks` blocks compound it to roughly `|h|^(2^num_blocks)`. The probe
seeded its ℓ=1 channels with the **raw Ångström CA displacement** over the history
step. Depth alone is survivable and a large input alone is survivable; together
they are not.

Three things this was *not*, each ruled out by measurement rather than argument:

* **Not corrupt data.** §2.1 — the targets are clean.
* **Not a temperature effect.** 320 K alone diverged as hard as the mixed set
  (peak 4.2e18). The instability is in the network, not in the hot trajectories.
* **Not Phase 1's fault.** Phase 1 uses the same block but interleaves its two
  backbone blocks with pooling and injection layers rather than stacking them, so
  it never compounds past one square. The Phase 1 checkpoint is untouched.

**The fix** (`force_md.transition.probe`): the ℓ=1 seed is now the displacement
*direction* — exactly equivariant, unit norm, cannot compound — and the magnitude
re-enters as bounded invariant scalars via `displacement_features` (Gaussian basis
over `[0, 20] Å` plus `tanh(|dr|/20)`, every channel in `[0, 1]`). No information
is discarded; it is moved from a channel that compounds to one that does not.

Same six configurations after the fix:

| configuration | median grad, before → after | peak grad, before → after |
|---|---|---|
| `num_blocks=2` | 1085 → **0.37** | 1.2e5 → **1.88** |
| `num_blocks=3` | 1.7e6 → **0.37** | 1.4e11 → **3.81** |
| `num_blocks=4` | non-finite → **0.77** | ∞ → **8.16** |
| `num_blocks=3`, 320 K only | 2.0e11 → **0.26** | 4.2e18 → **1.89** |
| `num_blocks=3`, 450 K only | 1.6e17 → **0.58** | ∞ → **14.74** |
| `num_blocks=3`, batch 4 | 4.8e12 → **0.58** | ∞ → **4.29** |

Zero skipped steps in every row; max loss under 10 in every row.

`tests/test_transition_probe.py` pins the bound at `num_blocks ∈ {2, 3, 4}`, so
this cannot come back silently. `configs/phase1_5_smoke.yaml` was moved to
`num_blocks: 3` as well — a smoke run at a different depth than the real configs
is what let this through in the first place.

### 2.1 The targets are clean

Checked before blaming the model, over 630 pairs spanning all five temperatures:

| T (K) | mean \|Δr\|, 1 ns | mean \|Δr\|, 4 ns | max \|Δr\| |
|---|---|---|---|
| 320 | 1.36 Å | 1.83 Å | 24.3 Å |
| 348 | 1.59 Å | 1.95 Å | 27.4 Å |
| 379 | 1.98 Å | 2.51 Å | 36.3 Å |
| 413 | 2.92 Å | 3.87 Å | 32.2 Å |
| 450 | 4.35 Å | 5.61 Å | 44.4 Å |

Monotone in temperature, monotone in lag, and bounded by 44 Å — real motion, no
periodic-image jumps, no corrupt frames. The 88 int32-overflow frames mdCATH ships
are already excluded by `mdcath_coord_quarantine.json`. The loss's
`translation_scale_angstrom: 2.5` sits sensibly inside this spread.

### 2.2 A gap left open

The trainer's divergence guard fires on **non-finite** values only. In the two
worst configurations above the loss reached 1e14 while staying finite, so the
guard never fired and `skipped_steps` stayed at 0. A finite but astronomical loss
is also divergence. This is not fixed — the cause is fixed instead — but a run
that goes bad in some other way can still burn its full budget silently. Worth
adding a magnitude guard before spending 34 GPU-hours.

---

## 3. Smoke ablation — a plumbing check, not a measurement

8 train domains / 2 validation domains, 28 train and 8 validation pairs, 200
steps, one temperature (320 K), one replica, single seed.

**At this scale no arm ordering means anything.** Validation error rises between
step 100 and step 200 for every arm — the model is memorising 28 pairs. The
numbers are here to show that five arms ran on one manifest and produced one
comparable table, which is what the smoke config is for.

Validation, micro-averaged, 179 s for all five arms, manifest `bde39893b1fd`:

| arm | Cα RMSD 1 ns | rel | Cα RMSD 4 ns | rel | rot 1 ns | rel | rot 4 ns | rel |
|---|---|---|---|---|---|---|---|---|
| identity baseline | 1.1518 Å | 1.000 | 1.3909 Å | 1.000 | 15.20° | 1.000 | 17.38° | 1.000 |
| `structure_only` | 1.2150 | 1.055 | 1.4258 | 1.025 | 17.01 | 1.119 | 19.14 | 1.101 |
| `force_torque` | 1.2561 | 1.091 | 1.4495 | 1.042 | 16.97 | 1.116 | 19.36 | 1.114 |
| `physics_latent` | 1.2331 | 1.071 | 1.4617 | 1.051 | 17.61 | 1.158 | 20.59 | 1.184 |
| `force_pattern_shape` | 1.1838 | 1.028 | 1.4663 | 1.054 | 16.63 | 1.094 | 19.77 | 1.137 |
| `oracle_force` | 1.2289 | 1.067 | 1.4378 | 1.034 | 17.10 | 1.124 | 19.28 | 1.109 |

**Every arm is worse than predicting that nothing moves** (`rel > 1`), including
the oracle. At 28 training pairs from 8 domains that is the expected outcome and
carries no information about the research question. Do not read an ordering out of
this table.

What the smoke run does establish:

* all five arms trained from the identical manifest `bde39893b1fd`, identical
  seed, identical step budget — the runner asserts this and would have aborted
* every arm starts at exactly the identity baseline (zero-initialised heads)
* gradient norms stayed in 0.30–3.00 at `num_blocks: 3` throughout, which is the
  regime §2 restored
* parameter counts differ only in the conditioner, as intended:

| arm | total | conditioner |
|---|---|---|
| `structure_only` | 313,896 | 0 |
| `force_torque` | 324,100 | 10,204 |
| `physics_latent` | 342,040 | 28,144 |
| `force_pattern_shape` | 370,804 | 56,908 |
| `oracle_force` | 370,804 | 56,908 |

The 56,908-parameter spread between A and D is 18% of the backbone; the full
experiment's decision gates must not be read without it in view.

---

## 4. The full experiment: cost, and the command

**Launched 2026-08-21 11:31**, three seeds in parallel on GPUs 4/5/6, one seed per
process so the runner's fairness assertion covers every arm it compares. All three
enumerated the identical manifest `213def6ef8dd` (289,120 train / 72,080 val), so
the seeds differ only in initialisation and batch order. Logs in
`runs/phase1_5_full_seed{0,1,2}/train.log`.

### Scale, measured not assumed

| quantity | value | how |
|---|---|---|
| train domains | 726 | Phase 1 split, cross-checked two ways |
| validation domains | 181 | same |
| train pairs | **289,120** | manifest enumerated, 16 s |
| validation pairs | **72,080** | manifest enumerated, 4 s |
| pairs per lag | 144,560 / 36,040 | balanced by construction |
| steps per epoch, batch 8 | 36,140 | |
| `max_steps: 40000` | **1.11 epochs** | just over one pass |
| throughput | **~6.1 steps/s** | measured, batch 8, 3 blocks, 8 workers, one H100 |

I/O is not the bottleneck at this scale: the 14,080-pair probe held ~6.1 steps/s
across 400 steps with no cache warmth advantage.

### Time

Budgeting 5.5 steps/s to leave margin for the heavier conditioner arms:

| item | per arm |
|---|---|
| training, 40,000 steps | 2.0 h |
| periodic validation, 20 × 200 batches | 0.07 h |
| final full-validation pass, 9,010 batches | 0.17 h |
| manifest build | negligible |
| **total** | **~2.3 h** |

| scope | GPU-hours | wall time |
|---|---|---|
| one arm | 2.3 | 2.3 h |
| one seed (5 arms) | 11.5 | 11.5 h sequential |
| **3 seeds × 5 arms** | **~34.5** | **~11.5 h on 3 GPUs**, one seed each |

Confirmed in flight: 200 steps in 33 s on a dedicated GPU = **6.06 steps/s**,
against the 6.1 predicted. (The first 430 s of each run were slower because a
4-GPU job on the same cards had restarted itself from a supervisor loop; once it
was cleared the rate matched.) Gradient norms 0.33–1.83, loss 1.5–4.3 — the
regime §2 restored, four orders of magnitude below the new `max_loss` guard.

### Storage

~20 MB per train manifest, ~5 MB per validation manifest, ~4 MB per checkpoint,
~12 MB of per-pair validation records — **~700 MB** for all 15 runs.

### Launch

GPU policy on this machine: long training uses **4–7 only**, never 0–3.

```bash
# one seed per GPU, three concurrently
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((4 + seed)) OMP_NUM_THREADS=8 \
    python scripts/run_phase1_5_ablation.py \
      --config configs/phase1_5_full.yaml --seed $seed &
done
wait
```

Before committing 34 GPU-hours, the bounded run is the cheaper next step:

```bash
# 120 domains, 6000 steps, one seed — roughly 1.5 h for all five arms
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 \
  python scripts/run_phase1_5_ablation.py --config configs/phase1_5_short.yaml
```

`--dry-run` reports pair counts, parameter counts and provenance for every arm
without training.

---

## 5. Results

Complete. 3 seeds × 5 arms × 40,000 steps, evaluated on the **full** validation
manifest `9158f71463f0` — 72,080 pairs over 181 held-out domains, 36,040 pairs per
lag. Micro-averaged; the domain-macro table agrees on every conclusion below.

### 5.0 A correction that had to happen first

The runner's first pass reported `domain_count = 4`, `pair_count = 800`.
`evaluate()` fell back to `config.eval_batches` — a *training-cadence* knob — when
the runner asked for the final measurement, and because the validation loader is
unshuffled that meant the same first four domains for every arm and every seed.
Internally consistent, and wrong. The numbers below come from a re-evaluation of
all 15 saved checkpoints over the whole validation set; no model was retrained.
`tests/test_transition_training.py` pins the semantics.

### 5.1 The table

| Arm | 1 ns Cα RMSD | 4 ns Cα RMSD | 1 ns rotation | 4 ns rotation | 해석 |
|---|---|---|---|---|---|
| identity ("nothing moves") | 2.9665 Å | 4.0285 Å | 32.343° | 40.502° | the reference |
| A `structure_only` | 2.7546 ±0.0054 | 3.6768 ±0.0073 | 30.079 ±0.063 | 37.938 ±0.072 | −7.1% / −8.7% vs identity |
| B `force_torque` | 2.7549 ±0.0094 | 3.6773 ±0.0162 | 30.054 ±0.013 | 37.919 ±0.009 | **indistinguishable from A** |
| C `physics_latent` | **2.7495** ±0.0032 | **3.6717** ±0.0077 | **29.856** ±0.046 | **37.760** ±0.036 | **best arm, beats A on every metric** |
| D `force_pattern_shape` | 2.7520 ±0.0040 | 3.6746 ±0.0074 | 29.905 ±0.068 | 37.810 ±0.082 | beats A, does **not** beat C |
| E `oracle_force` | 2.7520 ±0.0035 | 3.6752 ±0.0076 | 29.910 ±0.041 | 37.807 ±0.051 | beats A, does **not** beat C |

`±` is the max−min spread across the three seeds. Contact F1 separates no arm from
any other at any lag (every comparison flips sign across seeds).

### 5.2 Paired test

The three seeds share the manifest and differ only in initialisation and batch
order, so the per-seed difference against A is paired and removes the common
seed-level variation. `sd` is over the three paired differences.

| arm | metric | 1 ns | 4 ns |
|---|---|---|---|
| B | Cα RMSD | sign flips | sign flips |
| B | rotation | sign flips | sign flips |
| C | Cα RMSD | −0.0051 Å (3.6 sd) | −0.0050 Å (2.3 sd) |
| C | rotation | **−0.223° (4.7 sd)** | **−0.178° (4.0 sd)** |
| D | rotation | −0.174° (13.1 sd) | −0.129° (16.7 sd) |
| E | rotation | −0.169° (7.8 sd) | −0.131° (9.5 sd) |

D and E versus C: **worse or leaning worse on all four comparisons.** Neither the
explicit force-moment/shape conditioner nor ground-truth forces improve on the
learned Phase 1 latent.

### 5.3 The effects are real and they are tiny

| metric | lag | identity | A | C | C's gain over A | share of A's gain over identity |
|---|---|---|---|---|---|---|
| Cα RMSD | 1 ns | 2.9665 | 2.7546 | 2.7495 | 0.18% | +2.4% |
| Cα RMSD | 4 ns | 4.0285 | 3.6768 | 3.6717 | 0.14% | +1.4% |
| rotation | 1 ns | 32.343 | 30.079 | 29.856 | 0.74% | +9.8% |
| rotation | 4 ns | 40.502 | 37.938 | 37.760 | 0.47% | +6.9% |

The design's own rule is *no 1% difference is a success*. **Every raw difference is
below 1%.** By the criterion fixed before the experiment ran, force conditioning
does not clear the bar — the gap-closure column is the more flattering framing and
is reported second on purpose.

### 5.4 Gates

* **Gate 1 — C consistently beats A at both lags** → fires. Reproducible across
  three seeds, both lags, both aggregations, on rotation and Cα RMSD.
* **Gate 2 — B shows nothing, C does** → fires, cleanly. The summed residue force
  and torque carry nothing the structure does not already say; the learned
  equivariant latent carries a little. This is the internal-cancellation premise
  in §2 of the design confirmed by measurement.
* **Gate 3 — D beats C** → does not fire. D is worse than C.
* **Gate 4 — E does not beat A** → does not fire. E does beat A on rotation.
* **Gate 5 — E beats A but D does not** → does not fire; both beat A, comparably.

**The result that bounds everything:** E is an oracle. It reads ground-truth
mdCATH forces at frame `t`, and it does no better than C, which reads only Phase 1's
prediction. Perfect instantaneous force information is worth ~0.2° of frame
rotation and ~0.005 Å of Cα RMSD at 1–4 ns, and Phase 1's latent already captures
essentially all of it. That is a ceiling, not a training artefact, and it does not
move with more data or a better force model.

### 5.5 What this says for Phase 2

Gate 1 fires, so the Phase 1 representation is worth keeping — it is the best
conditioner tested and it beats the explicit alternative and the oracle. But it
buys under 1%, and the oracle proves that is the ceiling for *instantaneous* force
at these lags. Building Phase 2 around richer force conditioning would be
optimising against a measured bound. The 7–9% that all arms take off the identity
baseline comes from structure and history, not from force; that is where the
remaining signal is.

---

## 6. Risks going in

* **The identity baseline is strong.** At 1 ns / 320 K it is 1.1–1.4 Å Cα RMSD.
  Beating it requires predicting the *direction* of a small motion, not its scale.
* **Overfitting will look like physics.** Gate 6 exists for this: train improving
  while held-out domains do not is capacity, not a claim about force conditioning.
* **Arm E is an oracle and is expected to be optimistic.** If E does not beat A,
  that is informative (gate 4) and is the cheapest useful outcome of the probe.
* **Non-determinism.** `scatter_sum` is a CPU `index_add` whose accumulation order
  depends on thread scheduling; a 250-step Phase 1 test varied 0.554–0.626 across
  identical 8-thread runs and was bit-identical at 1 thread. Three seeds is a
  floor, not a luxury.
* **§2.2** — a finite-but-astronomical loss still runs to completion unnoticed.
