# Phase 1.5: the transition probe — design and contracts

Phase 1.5 is a **small vertical slice**, not Phase 2. It exists to answer one
question with the smallest apparatus that can answer it honestly:

> Does a model conditioned on Phase 1's force-supervised representation predict
> 1 ns and 4 ns future structure better than a transition baseline that sees only
> the current structure and its history?

Everything here is subordinate to that question. Where a richer model would have
answered it no better, the richer model was not built.

---

## 1. What this is not

Not implemented, deliberately, and not present even as a placeholder:

* stochastic flow matching, noise schedules, or any generative transition
* Chapman–Kolmogorov consistency
* long autoregressive rollout
* an all-atom side-chain decoder
* retraining or fine-tuning Phase 1

The probe predicts **one deterministic rigid update per residue**. Calling that a
flow model would be a claim the code does not support.

---

## 2. The measured facts that shaped the design

Every non-obvious decision below came from a measurement on real mdCATH, not from
a preference. They are collected here because a design document that only states
conclusions cannot be checked.

| measurement | value | what it decided |
|---|---|---|
| unaligned mean `\|CA(t+4ns) − CA(t)\|` | 8.97 Å | targets must be Kabsch-canonicalised |
| the same after proper Kabsch | 2.55 Å | ~70% of the raw displacement is tumbling |
| identity-baseline Cα RMSD, 1 ns @ 320 K | 1.1–1.4 Å | every metric needs its baseline quoted |
| the same at 4 ns | 1.5–1.8 Å | 1 ns and 4 ns are *not* independent difficulties |
| the same at 450 K | 3.0–5.6 Å | temperature dominates lag; report split by both |
| residue frame rotation, p95 @ 4 ns | 125° | rotation head must be 6D, not axis-angle |
| clash rate of real structures at 3.6 Å Cα | 0.000 | clash penalty ships at weight 0 |
| Phase 1 forward | ~14 ms/protein (H100) | five arms recompute the same features |
| e3nn `\|D₁(R) − R\|` in float64 | 4e-7 | use `R` directly for ℓ=1 |
| e3nn Wigner-D on GPU | raises | build ℓ≥2 blocks on CPU, transfer 70 kB not 47 MB |
| single-pair overfit, 600 steps @ 3e-3 | loss 1.26 → 0.003 | the probe is expressive enough |
| the same at 1e-3 | 0.019 | 3e-3 is faster **and** bounced at step 50 |
| mean `\|CA(t+1ns) − CA(t)\|`, 320 K | 1.36 Å | history input must be scale-free |
| the same at 450 K | 5.61 Å | 4× spread across temperature alone |
| per-residue max over 630 pairs | 44.4 Å | no corrupt frames; this is real motion |
| median grad norm, raw history seed, 3 blocks | 1.7e6 (peak 1.4e11) | the seed had to be normalised |
| the same with the history input removed | 0.01 (peak 0.74) | located the cause in that input |
| full manifest, 726 train domains | 289,120 pairs | 1.1 epochs at 40k steps, batch 8 |
| probe throughput, batch 8, 3 blocks | ~6 steps/s (1 H100) | cost estimate in the report |

---

## 3. Data: raw-trajectory lag pairs

`force_md.data.adapters.lag_pairs`.

One item is a transition `(q_{t-1}, q_t) → q_{t+lag}`.

**Lags come from raw frame indices.** `lag_frames = lag_ps / ps_per_frame`, and
the division must be exact — 1500 ps is refused, not rounded, because rounding
would relabel the physical time of every pair. At mdCATH's 1 ns/frame the default
lags `(1000, 4000) ps` are 1 and 4 frames.

**Phase 1's index must not be reused.** Phase 1 samples 40 frames evenly over a
~450-frame trajectory, so adjacent entries are 12–13 frames — 12–13 ns — apart.
Building "1 ns" pairs from them would be wrong by an order of magnitude and would
look entirely reasonable. `test_phase1_subsample_cannot_supply_a_one_nanosecond_lag`
is the guard.

**A pair never crosses a trajectory.** History, current and future all come from
one `(domain, temperature, replica)`.

**The split is Phase 1's, by domain.** Restored from the run snapshot *and*
recomputed from the recipe, with the two required to agree
(`restore_phase1_split`). Verified: 726 train / 181 val, zero overlap.

**Both quarantines are reused unchanged.** A coordinate-quarantined frame
disqualifies every pair that would touch it as history, current or future.
Force-quarantined trajectories are excluded by default so that arms A–E, including
the ground-truth oracle, run on exactly the same pairs; a production-only manifest
is available via `require_current_force_labels=False`.

**Manifests are content-hashed** and record the config, seed, quarantine file
hashes and per-lag counts. Loading a manifest whose rows do not match its hash
raises.

---

## 4. Target: what the model is asked to predict

`force_md.transition.targets`.

The future structure is first canonicalised: take the residues with a valid frame
in **both** structures, fit the proper rotation (`det = +1`, no reflection) taking
the future Cα onto the current, and apply it to the whole future structure. What
remains is conformational change.

Conventions, fixed once and asserted by test:

```
R_i has the local axes as COLUMNS, so R_i maps local -> global
delta_r_local_i = R_cur_i^T (CA_fut_aligned_i − CA_cur_i)
R_rel_i         = R_cur_i^T R_fut_aligned_i          (right multiplication)

delta_r_global_i = R_cur_i delta_r_local_i
R_fut_aligned_i  = R_cur_i R_rel_i
```

Both targets are **invariant** under a global rigid motion of the pair, and
invariant under a rigid motion of the future alone — the second is what proves the
tumbling has been removed rather than merely reduced.

**The rotation target is a matrix, not a chart.** Charts are needed only where a
network must *emit* a rotation.

**Kabsch is used for the target and the metrics, and nowhere else.** The alignment
is computed from the future, so aligning an input would leak the label.

---

## 5. Metrics

`force_md.transition.metrics`. Seven views, because a model can be right in one
and wrong in the others:

1. Cα RMSD (and `ca_rmsd_aligned`, after re-fitting the prediction to the target)
2. residue translation RMSE
3. residue-frame rotation geodesic error, degrees
4. pair-distance MAE
5. contact precision / recall / F1 at a configurable cutoff
6. clash rate on the reconstructed backbone
7. backbone `phi` / `psi` MAE

**Every one is reported beside the identity baseline** (`*_identity`) and as a
ratio (`*_relative`). A relative at or above 1.0 means the model has not improved
on predicting that nothing moved.

**Aggregation is reported twice**: `micro` weights residues, `domain_macro`
weights domains. They answer different questions, and a gain in only one of them
is not a gain.

Thresholds are sourced, not invented. The clash criterion is 3.6 Å between
non-neighbouring Cα, which is *below* this dataset's audited bonded Cα–Cα range of
3.67–4.01 Å: a violation means two residues are closer than two bonded ones. No
van der Waals table was introduced, because the repository has no sourced one.

---

## 6. Frozen Phase 1 features

`force_md.transition.phase1_features`.

Phase 1 is loaded from `runs/phase1_full/last.pt`, put in `eval()`, frozen, and
run under `no_grad`. The checkpoint is never written to — asserted by comparing
its SHA-256 and mtime across a load-and-run.

The recorded latent contract, the rebuilt model's contract and any runtime
expectation are cross-checked; a mismatch raises. A checkpoint that loads but
produces features of the wrong width, row order or frame is more dangerous than
one that fails to load.

**Label leakage is prevented by the type system.** Ground-truth forces are not a
nullable field of the production bundle — they live in a different class,
`OracleFeatureBundle`, produced by a different method. `FeatureBundle` has no
`forces` attribute at all, because a nullable field invites
`if bundle.forces is not None` and one day it will not be None. A production
conditioner handed an oracle bundle raises `TypeError`.

"Frozen" is verified by running a downstream head, backpropagating, and requiring
every Phase 1 parameter to still have `grad is None`.

---

## 7. The five conditioners

`force_md.transition.conditioners`. All emit `[N_res, d_cond]` **invariant**
scalars through one shared adapter shape.

| arm | conditioner | sees |
|---|---|---|
| A | `ZeroConditioner` | nothing — structure-only control |
| B | `ResidueForceTorqueConditioner` | predicted residue net force and torque |
| C | `PhysicsLatentConditioner` | the 152-d residue latent |
| D | `ForcePatternShapeConditioner` | latent + atom-force pattern + moments + shape |
| E | `OracleAtomicForceConditioner` | ground-truth atom forces — diagnostic only |

**Global-frame irreps are rotated into the residue frame, never flattened.**
`z_local = D(R^T) z` makes every component an SE(3) invariant with nothing
discarded. ℓ=0 passes through, ℓ=1 uses `R` itself (exact, and e3nn's own ℓ=1
Wigner-D differs from `R` by 4e-7 through its angle round-trip), and only ℓ≥2
goes through e3nn — on CPU, as 5×5 blocks, because e3nn's generators are cached
there and the full block-diagonal would move 47 MB per batch instead of 70 kB.

**Why arm D exists.** The residue net force is the right observable for
translation and a poor conformational descriptor, because Newton's third law
cancels the internal forces: Phase 1 measured mean atomic `|f| = 38.4` against a
mean residue `|F| = 51.8`, where uncorrelated atoms would give ~108. The first
moment `M_i = Σ y_ia ⊗ f_local_ia` keeps what the sum cancels, and splits into
isotropic compression, the torque, and symmetric traceless stress. A two-atom
example makes it concrete: pull two atoms of one residue in exactly opposite
directions and the net force and torque are both zero while the isotropic and
traceless components are not.

Atom-set pooling forms a **nonlinear per-atom message first and sums second**.
Summing raw forces and then applying an MLP would discard the pattern before the
network saw it, making arm D an expensive way to recompute arm B.

**Fairness.** One adapter shape, one `d_cond`, one hidden width for every arm.
Raw input widths differ — that is the experiment — so parameter counts differ and
are recorded per arm rather than hidden.

**Arm C is not new information.** `z_phys = g(q_t, sequence, temperature)` is a
function of inputs arm A already has. If C beats A, that is evidence about
force-supervised pretraining as an inductive bias, **not** about extra
observations, and the report must not say otherwise.

---

## 8. The probe

`force_md.transition.probe`. A residue-level graph model over Phase 1's own
`BackboneInteractionBlock`, with the same sequence (±1, ±2) and Cα-kNN(16) edge
semantics.

Inputs: residue identity and PLM, temperature, a continuous lag encoding, the
conditioner block, and history `(t−1, t)` — the history displacement *direction*
as an equivariant ℓ=1 seed, its magnitude as bounded invariant scalars, and the
history frame turn as an invariant 6D.

**Why the seed is a direction and not the displacement itself.** Every
`BackboneInteractionBlock` carries a body-order-3 term, `h ← h + square_mix(h ⊗ h)`,
so feature magnitude is squared once per block and `num_blocks` blocks compound it
to roughly `|h|^(2^num_blocks)`. Seeding ℓ=1 with a raw Ångström displacement —
1.36 Å mean at 320 K, 5.61 Å at 450 K, 44 Å at the per-residue maximum — turns
that into a divergence rather than a non-linearity. Measured over 60 real steps
at `num_blocks=3`: median gradient norm **1.7e6**, peak **1.4e11**, training Cα
RMSD reaching 1.1e8 Å. The same run with the history input removed gave median
**0.01** and peak **0.74**, which is what located the cause. Both the depth and
the magnitude are needed to trigger it — `num_blocks=2` survived (peak 1.2e5),
`num_blocks=4` went non-finite — and it is *not* a temperature effect: 320 K alone
diverged as hard as the mixed set.

The unit direction is exactly equivariant and cannot compound, and
`displacement_features` puts the magnitude back as a Gaussian basis over
`[0, 20] Å` plus `tanh(|dr|/20)`, every channel in `[0, 1]`. Phase 1 never hits
this because it interleaves its two backbone blocks with pooling and injection
layers instead of stacking them; the probe stacks `num_blocks` of them directly.
`tests/test_transition_probe.py` pins the bound at `num_blocks ∈ {2, 3, 4}`.

Outputs, both expressed in the current residue frame, both therefore invariant:

```
delta_r_local = R_cur^T v                       v from a 1x1o head
R_rel         = R_cur^T GramSchmidt(R_cur[:, :2] + delta)
```

**The rotation head predicts a correction to the current frame's own axes**, so
zero-initialised it reproduces `R_cur` exactly and `R_rel = I`. Together with the
zero-initialised translation head, **every arm starts at exactly the identity
baseline** — verified numerically, all five arms at Cα RMSD 1.9069 Å and 20.80°
on the same batch. No arm gets a head start, and "has it learned anything" is
readable without interpreting a loss curve.

**Rotation uses the 6D chart** (Zhou et al. 2019): continuous, surjective onto
SO(3), no antipodal identification. Axis-angle was rejected on the measurement
that 4 ns frame rotations reach a p95 of 125°, close enough to its 180°
discontinuity to matter.

---

## 9. Losses

`force_md.transition.losses`.

```
L = λ_pos   · Huber(|Δr_pred − Δr_target| / 2.5 Å)
  + λ_rot   · ‖R_pred − R_target‖²_F / (8 sin²(0.59/2))
  + λ_pair  · pair-distance MAE within 12 Å
  + λ_clash · Σ relu(3.6 Å − d)²                      [weight 0 by default]
```

**Position and rotation are divided by their own measured scales before being
added.** Ångström and radians are not commensurable; adding them raw sets the
trade-off to whatever the units happen to be. The scales (2.5 Å, 0.59 rad) are the
measured identity-baseline magnitudes, so both terms enter at O(1).

**The rotation loss is chordal, and reported as geodesic.** The geodesic angle is
the right metric and a poor loss: its gradient carries `1/sin θ` and is singular
at `θ = 0`, exactly where a converging model sits. `‖R_a − R_b‖²_F = 8 sin²(θ/2)`
is smooth everywhere with the same minimiser. `rotation_loss="geodesic"` remains
available and is documented as the less stable option.

**The clash penalty ships at weight 0** because the measured clash rate of both
the current and the future structure is 0.000 — there is nothing for it to fix
until a model starts producing collapses.

---

## 10. Protocol

`force_md.training.transition_module`, `scripts/run_phase1_5_ablation.py`.

Every arm of one ablation must see the identical experiment: same manifest, same
seed, same batch order, same optimiser and schedule, same step budget, same
backbone, same `d_cond`. The runner **asserts** this rather than assuming it — the
manifest hash, Phase 1 checkpoint hash, seed and step budget of each arm are
compared against the first arm's, and a mismatch aborts.

Learning rate defaults to **1e-3**, not the 3e-3 that a single-pair sweep found
faster. Phase 1's own config records "3e-3 diverged. Twice." over a long run, and
a short sweep measures which rate descends fastest, not which survives.

Non-finite steps cost one batch, not the run: backward always executes (DDP
expects one backward per forward), only the optimiser step is withheld, and 20
consecutive skips abort as divergence rather than bad data.

**A note on determinism.** `scatter_sum` is a CPU `index_add`, whose accumulation
order depends on thread scheduling; a Phase 1 test that ran 250 optimisation steps
was measured to vary from 0.554 to 0.626 across identical runs at 8 threads and to
be bit-identical at 1 thread. Phase 1.5 results carry the same property. This is
one more reason the decision gates below require three seeds.

---

## 11. Decision gates

Applied to the **full** 3-seed experiment with domain-level confidence intervals.
A single-seed ordering is an observation, not a result, and no 1% difference is a
success.

> **Outcome (2026-08-24).** Gates 1 and 2 fired; 3, 4 and 5 did not. Every effect
> came in under the 1% bar, and the oracle arm did not beat the learned latent —
> so the bar was missed, not merely approached. See
> [phase1_5_report.md §5](phase1_5_report.md).

1. **C consistently beats A at both lags** → keep the current Phase 1
   representation and extend to full Phase 2.
2. **B shows nothing, C does** → the learned equivariant latent matters, the
   summed residue force does not.
3. **D beats C** → adopt the explicit force-moment/shape conditioner in the
   Phase 2 interface.
4. **E does not beat A either** → conditioning on instantaneous force has little
   value at 1–4 ns; redesign Phase 2 around history and stochastic memory.
5. **E beats A but D does not** → force information is useful; the bottleneck is
   Phase 1's prediction or its representation.
6. **Train improves, held-out domains do not** → capacity/overfitting. Not a
   physics claim.

---

## 12. What Phase 2 inherits

The lag-pair dataset, the canonicalised target, the metric suite with its identity
baselines, the frozen-feature interface with its production/oracle separation, the
conditioner interface, and the ablation runner. None of it is throwaway
scaffolding; the probe itself is the only part expected to be replaced.
