# Phase 1.6 — force-informed pair interaction, feasibility ablation

**Status: Stages A and B complete. Stage C not started (see §6.4). Stage D
pending on Stage C. `P5` prepared, not implemented, not run.**
Sections marked `pending` contain no numbers because none have been measured.

**Headline:** the pair *architecture* is worth more than the pair *physics*. A
geometry-only control with the same edges, the same pooling and the same
parameter count is the best arm on Cα RMSD at both lags; Phase 1's pair message
beats it on rotation and loses on RMSD. The oracle again fails to beat the
structure-only baseline, reproducing the Phase 1.5 ceiling on a different
manifest.

Phase 1.5 answered its question and closed a door: conditioning a transition
probe on Phase 1's *node* latent beats a structure-only control reproducibly and
by under 1%, and an oracle reading ground-truth forces did no better, so that is
a ceiling on instantaneous force at 1–4 ns rather than a training artefact.

Phase 1.6 asks a different question with the same machinery:

> Phase 1 computes an interaction for every residue pair and then sums it away.
> Is the interaction worth more than the sum?

---

## 1. What is new, and what is reused

### 1.1 The tensor this is about

Inside every `EquivariantMessageBlock` ([nn/blocks.py:195](../src/force_md/nn/blocks.py#L195)):

```python
messages = self.tp(node_features[edges.src], edge_sh, weights)   # [E, 880]
pooled   = scatter_sum(messages, edges.dst, n) / avg_neighbors**0.5
h        = h + self.post_message(pooled)
```

Only `pooled` survived. `messages` is the pair latent: one 880-dimensional
irreps tensor (`88x0e + 104x1o + 96x2e`) per directed residue edge, taken from
the **last backbone block of the last cycle**, where the graph is sequence ±1/±2
plus CA-kNN(16).

The brief that commissioned this work stated Phase 1 already exposes
"pair/triplet/higher-order interaction latent". It does not — `Phase1Output` had
fifteen fields and none was edge-level. **There is also no triplet tensor and none
was invented**: Phase 1's body-order-3 term is a node-level `o3.TensorSquare`, so
that information already lives in `physics_latent`, which is exactly what arm
`S2` conditions on. Phase 1.6 therefore compares *edge-level* physics against
*node-level* physics, and says nothing about triplets it does not have.
Full audit: [phase1_6_audit.md](phase1_6_audit.md).

### 1.2 Reused from Phase 1.5, unchanged

The split (`restore_phase1_split` → 726/181 domains), the lag pairing
(`exact_lag_frames`, 1 ns and 4 ns = 1 and 4 frames), the manifest and its hash,
Kabsch alignment and the residue-frame targets, the metric definitions, the
`TransitionProbe` backbone, `d_cond`, the optimiser and schedule, the identity
baseline, the results row builder, and the runner's fairness assertion. `P2`'s
force moments are a **call to the same `force_moments()`** arm D used, not a
reimplementation.

### 1.3 Added

| Piece | File | What it does |
|---|---|---|
| Pair message interface | [nn/blocks.py](../src/force_md/nn/blocks.py), [nn/hierarchical_encoder.py](../src/force_md/nn/hierarchical_encoder.py), [models/local_physics.py](../src/force_md/models/local_physics.py) | `forward(..., return_pair_messages=True)` populates `Phase1Output.pair`. Additive and defaulted off. |
| Pair adapter and arms P0–P4 | [transition/pair_physics.py](../src/force_md/transition/pair_physics.py) | edge source → message MLP → degree-stable pooling → the shared adapter |
| P4 auxiliary loss | [transition/future_physics.py](../src/force_md/transition/future_physics.py) | future physics latent as a detached Huber target |
| Canonical arm registry | [transition/arms.py](../src/force_md/transition/arms.py) | `P1_pair_physics_frozen` ↔ `pair_physics`, stages, oracle flags |
| Staged runner | [scripts/run_phase1_6_ablation.py](../scripts/run_phase1_6_ablation.py) | canonical names, resource records, `--reuse-from` |
| Statistics and report | [scripts/analyze_phase1_6.py](../scripts/analyze_phase1_6.py) | paired on sample id, domain cluster bootstrap, SVG figures |

**No Phase 1 state-dict key changed and no existing checkpoint was touched.**
`runs/phase1_full/last.pt` loads exactly as before; the default forward path is
bitwise what it was (`test_pair_messages_are_off_by_default`).

---

## 2. The arms

| Canonical | Implementation | Conditions on | Oracle |
|---|---|---|---|
| `S0_structure_history` | `structure_only` | structure, history, PLM, temperature, lag | no |
| `S1_residue_wrench` | `force_torque` | S0 + predicted residue net force and torque | no |
| `S2_node_physics_152d` | `physics_latent` | S0 + the 152-d node latent, local frame | no |
| `P0_pair_geometry_control` | `pair_geometry` | the same edges, geometry only, capacity-matched | no |
| `P1_pair_physics_frozen` | `pair_physics` | Phase 1's pair messages + pair geometry | no |
| `P2_pair_physics_moments` | `pair_physics_moments` | P1 + predicted-force moments `F, τ, S` | no |
| `P3_pair_physics_uncertainty` | `pair_physics_uncertainty` | P2 + gate `σ(MLP[log σ̂, T, Δ])` | no |
| `P4_future_physics_consistency` | `pair_physics_future` | P2's inputs + an auxiliary future-latent loss | no |
| `O_current_gt_force_oracle` | `oracle_force` | **ground-truth atom forces at t** | **yes** |
| `P5_pair_physics_partial_e2e` | — | gated; config only, see §7 | no |
| *(legacy)* `force_pattern_shape` | Phase 1.5 arm D | kept, reported as legacy, no canonical role | no |

### 2.1 The control is the experiment

`P1` beating `S0` would prove nothing: it has more parameters and an extra
pooling stage. So `P0` is the **same architecture with the physics removed** —
same edge set, same edge geometry, same message MLP, same pooling, same adapter,
same `d_cond` — differing only inside one module, and matched in parameter count
by construction:

```
h = (2P + P·d − 2G) / (G + 1 + d)          matched_hidden_width()
```

solves the geometry source's hidden width so its parameter count equals the
physics source's. Measured at the production widths:

| arm | total | conditioner | difference |
|---|---|---|---|
| `P0_pair_geometry_control` | 386,426 | 72,530 | — |
| `P1_pair_physics_frozen` | 386,452 | 72,556 | **26 parameters (0.007%)** |

Because the formula solves rather than guesses, this holds at any width, and the
test `test_geometry_control_is_capacity_matched_to_the_physics_arm` fails if a
future edit breaks it.

### 2.2 Everything is invariant

The pair message is a global-frame irreps tensor. It is rotated into the
**receiving** residue's frame by `IrrepsLocalFrame` before it meets an MLP; the
edge direction enters as `R_i^T r̂_ij` and the relative orientation as `R_i^T R_j`.
Verified in float64 to `1e-8` under a random rotation plus translation
(`test_conditioner_is_invariant_under_a_global_rigid_motion`).

---

## 3. Leakage guards

Every one is a runtime assertion, not only a test.

| # | Guard | Mechanism | Test |
|---|---|---|---|
| 1 | production arms cannot read GT force | `assert_production_safe()` raises `TypeError` on an `OracleFeatureBundle` | `test_production_pair_arms_refuse_an_oracle_bundle` |
| 2 | only `O` reads current GT force | `OracleFeatureBundle` is a **separate class**, not a nullable field | `test_only_the_oracle_arm_reads_ground_truth_force` |
| 3 | no arm reads future force | `future_state_batch()` builds the future state with `forces=None` | `test_future_state_batch_has_no_forces` |
| 4 | `P4`'s target is detached | `future_physics_target()` detaches at creation | `test_p4_target_is_detached_...` |
| 5 | future perturbation cannot move the conditioner | — | `test_changing_the_future_does_not_change_the_conditioning` |
| 6 | an oracle checkpoint cannot be exported | `export_production_checkpoint()` refuses `oracle=True` | `test_oracle_checkpoint_cannot_be_exported_as_production` |
| 7 | no future contact in current edges | edges are built from the current bundle only | (5) covers it |
| 8 | frozen Phase 1 takes no gradient | `eval()`, `requires_grad_(False)`, `no_grad` forward | `test_frozen_phase1_receives_no_gradient_from_a_pair_arm` |

Guard 8 is checked the strong way: a real loss is backpropagated through a pair
arm and **every** Phase 1 parameter must still have `grad is None`, while the
conditioner's must not.

Two failure modes specific to Phase 1.6 get their own guards, because both would
surface as a *null result* rather than as an error:

* a pair arm handed a node-only bundle raises rather than conditioning on nothing
  (`FeatureBundle.require_pair()`);
* a `P4` head with a zero weight, or a non-zero weight with no head, is refused at
  trainer construction — otherwise `P4` would silently be `P2` under another name.

---

## 4. Feature cache

`FEATURE_FORMAT_VERSION` 1 → 2, and the shard header now carries
`pair_features`. A shard written without pair messages is **rejected**, not read
with a missing column. `split_bundle()` refuses to shard a bundle carrying pair
messages rather than renumbering an edge index across a graph boundary, where a
mistake would silently condition one protein on another.

---

## 5. Stage A — smoke

8 domains, 2 val, 28 train pairs, 100 steps, both lags, all nine arms on GPU 4.
Every arm ran forward and backward, checkpointed, reloaded and evaluated; the
feature cache, the CLI and the guards were exercised.

**No number from Stage A is a result** — at 28 training pairs every arm is worse
than the identity baseline, which is what 100 steps on 28 pairs looks like. What
it establishes is that the ten arms share one output contract
(`test_every_arm_emits_the_same_output_shapes`) and that the pipeline runs.

Resource record from that run:

| arm | params | conditioner | peak GPU | wall |
|---|---|---|---|---|
| `S0_structure_history` | 313,896 | 0 | 1408 MiB | 16 s |
| `S2_node_physics_152d` | 342,040 | 28,144 | 1409 MiB | 15 s |
| `P0_pair_geometry_control` | 386,426 | 72,530 | 1409 MiB | 24 s |
| `P1_pair_physics_frozen` | 386,452 | 72,556 | 1409 MiB | 24 s |
| `P2_pair_physics_moments` | 389,312 | 75,416 | 1410 MiB | 24 s |
| `P3_pair_physics_uncertainty` | 390,305 | 76,409 | 1409 MiB | 24 s |
| `P4_future_physics_consistency` | 407,020 | 75,416 | 1721 MiB | 30 s |
| `O_current_gt_force_oracle` | 370,804 | 56,908 | 1409 MiB | 16 s |

`P4` costs ~22% more memory and ~25% more time than `P2`: it runs frozen Phase 1
a second time, on the future structure, to build its target.

---

## 6. Stage B — bounded screening

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_phase1_6_ablation.py \
  --config configs/phase1_6_bounded.yaml --out-dir runs/phase1_6_bounded_seed0
python scripts/analyze_phase1_6.py --runs runs/phase1_6_bounded_seed0 \
  --out docs/phase1_6_results_bounded.md --plots docs/figures
```

120 train / 30 val domains, 14,080 train and 3,520 val pairs, 6,000 steps, one
seed, seven arms (`S0 S1 S2 P0 P1 P2 O`), both lags. GPU 4, ~2 h 40 m total.
Every arm evaluated the **identical 3,520 sample ids**, asserted by the analysis.

Full tables, per-temperature stratification and every hash:
[phase1_6_results_bounded.md](phase1_6_results_bounded.md). Figures in
[figures/](figures/).

### 6.0 A measurement error that had to be fixed first

The first analysis pass reported 150 pairs per lag instead of 1,760 and an
identity baseline of 3.619 Å instead of 3.058 Å. The evaluation records carried
no unique sample id, so the paired analysis keyed on
`(domain, temperature, lag)` — which is **not unique**: one domain contributes
several replicas and several frames. A dictionary keyed on it silently keeps the
last row of each group, and every downstream number was computed on a sixteenth
of the data with nothing to indicate it.

Fixed by recording `LagPair.pair_id` in every metric record, and by making
`key_of()` **raise** on a record without one rather than falling back. The seven
saved checkpoints were then re-evaluated over the whole validation set with
`scripts/reevaluate_phase1_6.py`; **no model was retrained**, and the re-evaluated
micro means reproduce the runner's own table exactly.

This is the same class of error as Phase 1.5 §5.0 — internally consistent, and
wrong. It is recorded here for the same reason.

### 6.1 Reproduction check against Phase 1.5

The budget is deliberately identical to Phase 1.5's `short` run: manifest
`24a1abad503a`, Phase 1 checkpoint `94b0d7c02e55`, seed 0, 6,000 steps all match
`runs/phase1_5_short_seed0`. So `S0/S1/S2/O` were reusable under the plan's rule —
and re-running them instead turns them into a reproduction check.

That earlier run's stored results came from the buggy `eval_batches` path Phase
1.5 §5.0 documents (4 domains, 240 pairs per lag), so the comparison is made on
**its** sample set and with `aggregate_metric_records`, the same function that
produced it:

| arm | Phase 1.5 short | Phase 1.6, same 4 domains | Δ |
|---|---|---|---|
| identity baseline (1 ns) | 3.0918 | **3.0918** | bit-identical |
| `structure_only` 1 ns | 2.9159 | 2.9138 | −0.07% |
| `structure_only` 4 ns | 3.9393 | 3.9400 | +0.02% |
| `force_torque` 1 ns | 2.9196 | 2.9190 | −0.02% |
| `physics_latent` 1 ns | 2.9143 | 2.9117 | −0.09% |
| `oracle_force` 1 ns | 2.9184 | 2.9135 | −0.17% |

Every arm reproduces to within 0.17%, inside the `scatter_sum` non-determinism
this repository has measured before. The Phase 1.6 harness is the Phase 1.5
harness.

### 6.2 The result

Validation, both lags, 1,760 pairs per lag over 30 held-out domains. `±` is a
95% cluster bootstrap over domains. **Note the averaging**: this table weights
each pair equally, while the runner's console table weights by residue, so the
absolute numbers differ from the training log while the ordering does not.

| arm | 1 ns Cα RMSD | 4 ns Cα RMSD | 1 ns rotation | 4 ns rotation |
|---|---|---|---|---|
| identity ("nothing moves") | 3.4178 | 4.6513 | 37.799 | 47.547 |
| `S0_structure_history` | 3.2079 | 4.2951 | 36.094 | 45.578 |
| `S1_residue_wrench` | 3.2084 | 4.3011 | 36.110 | 45.640 |
| `S2_node_physics_152d` | 3.2089 | 4.3029 | 35.976 | 45.543 |
| `P0_pair_geometry_control` | **3.1913** | **4.2777** | 35.873 | 45.444 |
| `P1_pair_physics_frozen` | 3.2021 | 4.2874 | **35.776** | **45.342** |
| `P2_pair_physics_moments` | 3.2041 | 4.2798 | 35.980 | 45.484 |
| `O_current_gt_force_oracle` | 3.2101 | 4.3028 | 36.041 | 45.591 |

Paired deltas against the control, cluster-bootstrapped (positive = the second
arm is better):

| comparison | 1 ns Cα RMSD | 4 ns Cα RMSD | 1 ns rotation | 4 ns rotation |
|---|---|---|---|---|
| `P1` − `P0` | **−0.34%** worse | **−0.23%** worse | **+0.27%** better | **+0.23%** better |
| `P2` − `P0` | **−0.40%** worse | −0.05% n.s. | **−0.30%** worse | −0.09% n.s. |
| `P1` − `S2` | **+0.21%** better | **+0.36%** better | **+0.56%** better | **+0.44%** better |
| `P2` − `P1` | −0.07% n.s. | **+0.18%** better | **−0.57%** worse | **−0.31%** worse |
| `O` − `S0` | −0.07% n.s. | **−0.18%** worse | **+0.15%** better | −0.03% n.s. |

"n.s." = the 95% interval contains zero.

### 6.3 What this says

**The pair architecture is worth more than the pair physics.** `P0` — the same
edges, the same degree-stable pooling, the same adapter, 26 fewer parameters, and
**no physics whatsoever** — is the best arm on Cα RMSD at both lags, taking 6.6%
and 8.0% off the identity baseline against `S0`'s 6.1% and 7.7%. Whatever the
edge-message stage is buying, it is buying it from geometry.

**Phase 1's pair message is not nothing, but it is not what was hypothesised.**
`P1` beats `P0` on rotation at both lags (+0.27%, +0.23%, intervals excluding
zero) and loses to it on Cα RMSD at both lags. A split of signs across the two
primary metrics is not a gate firing; it is the same pattern Phase 1.5 saw, where
rotation moved first and by more than placement.

**Edge-level physics does beat node-level physics.** `P1` beats `S2` on all four
comparisons, 0.21–0.56%. That answers research question 2 in the affirmative —
the interaction carries more than the sum it was collapsed into — but inside a
regime where neither of them beats plain geometry.

**The force moments do not help, again.** `P2` is worse than `P1` on rotation at
both lags and separates on RMSD only at 4 ns. Phase 1.5 measured the same thing
for arm D; this is the second independent measurement that explicit `F, τ, S`
of the predicted forces add nothing the latent does not already carry.

**The oracle reproduces the Phase 1.5 ceiling on a different manifest.** Reading
ground-truth atom forces at `t` does **not** beat structure-and-history on Cα
RMSD — nominally worse at 1 ns (n.s.) and significantly worse at 4 ns — and buys
0.15% of rotation at 1 ns only. Recoverability is `not identifiable` at both
lags, exactly as the degenerate-denominator rule anticipated. Phase 1.5's
conclusion was not an artefact of that manifest.

**Cost.** `P1`/`P2` take 2,025 s against `P0`'s 1,124 s — **1.8× the wall time**
— because Phase 1 must emit an 880-dimensional tensor per residue edge. Peak
memory is indistinguishable (2,982 MiB vs 2,980 MiB).

**Temperature.** `P0` is the best arm at every one of the five temperatures at
1 ns. At 4 ns `P2` is best at 450 K (7.3536 against `P0`'s 7.3747) and `P0`
elsewhere. One seed; no weight is put on it.

### 6.4 The Stage C gate does not fire cleanly

The pre-registered gate was: *`P1` or `P2` improves on `P0` in the primary
metric, and not by parameter count*. Capacity is excluded by construction
(26 parameters, 0.007%), but the metric disagrees with itself — `P1` improves on
rotation and regresses on Cα RMSD, at both lags, with intervals that exclude zero
in both directions. `P2` never improves on `P0`.

So **Stage C was not started**, which is also what the plan requires absent an
explicit request. Three things would resolve the split, in order of cost:

1. **Re-run the screening at a second seed.** The whole result rests on
   differences of 0.2–0.4% from one seed. Phase 1.5 needed three seeds to
   separate 0.18%, and its own note says three is "a floor, not a luxury".
   ~2.7 GPU-hours.
2. **Ask whether the rotation gain survives at all.** If `P1`'s rotation
   advantage over `P0` reproduces across seeds while the RMSD regression also
   reproduces, that is a real trade-off and a finding — pair physics buys
   orientation at the cost of placement — not a null.
3. **Only then** consider Stage C's 3-seed confirmation.

---

## 7. Stages C, D and the gated arm

### Stage C — 3-seed confirmation: `not started`

The gate did not fire cleanly (§6.4) and the plan forbids starting a long run
without an explicit request, so no Stage C numbers exist and none are estimated.
The command exists so that request is one line:

```bash
bash scripts/launch_phase1_6_full.sh      # seeds 0,1,2 on GPUs 4,5,6
```

Arms confirmed: `S0`, `S2`, `P0`, the better of `P1`/`P2`, and `O`. Cost is
comparable to Phase 1.5's full experiment (~34 GPU-hours for 3 × 5 × 40,000
steps), plus the pair arms' 1.8× overhead — call it ~45 GPU-hours.

The cheaper next step recommended in §6.4 is a second screening seed:

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_phase1_6_ablation.py \
  --config configs/phase1_6_bounded.yaml --seed 1 \
  --out-dir runs/phase1_6_bounded_seed1
python scripts/analyze_phase1_6.py \
  --runs runs/phase1_6_bounded_seed0 runs/phase1_6_bounded_seed1 \
  --out docs/phase1_6_results_bounded.md --plots docs/figures
```

### Stage D — mechanism: `not run`

`P3` and `P4` are implemented, registered and pass Stage A on real data with the
full leakage guard suite. They are **not screened**: the plan makes them
conditional on Stage C confirming a pair-physics signal, and Stage C has not run.
No `P3` or `P4` number appears anywhere in this report.

What Stage A established about them, and nothing more: both build, train,
checkpoint and reload; `P4`'s auxiliary loss is reported and its target is
detached (`test_p4_target_is_detached_...`); `P4` costs ~22% more memory and ~25%
more time than `P2` because it runs frozen Phase 1 a second time on the future
structure.

### `P5_pair_physics_partial_e2e` — prepared, not implemented, not run

[configs/phase1_6_p5.yaml](../configs/phase1_6_p5.yaml) records what the arm
would be. The runner **refuses** any arm whose stage is `gated`.

Its precondition ("P1 or P2 clearly better than P0") has not fired, so the code
path is deliberately absent rather than stubbed. Three things would have to exist
first, and building the first without the other two would produce a number worse
than no number:

1. selective unfreezing of Phase 1's last backbone block only;
2. the Phase 1 force auxiliary loss that keeps force-reconstruction semantics
   from collapsing while the transition loss pulls on those weights — without it
   the arm improves the transition metric by destroying the representation the
   whole experiment is about, and that looks like a win;
3. a geometry-only partial-finetuning control, without which a `P5` number
   supports no conclusion about physics.

---

## 8. The decision matrix, and which row fired

Fixed before the numbers; the right-hand column is the Stage B reading.

| Observation | Reading | Stage B |
|---|---|---|
| Oracle does not beat `S0` | instantaneous force carries little extra predictive signal at 1–4 ns, or the probe cannot use it | **fires** on Cα RMSD at both lags. Phase 1.5's ceiling reproduced on a new manifest. |
| Oracle beats `S0`, `P1`/`P2` do not | GT force carries hidden-solvent / omitted-DOF information not recoverable from geometry and history | **does not fire** — the oracle itself does not beat `S0` |
| `P1`/`P2` beat `P0` | supports force-supervised **pair** representation as the mechanism, not capacity | **splits**: `P1` yes on rotation, no on Cα RMSD, both lags. `P2` never. |
| `P2` beats `P1` | torque and stress-like pattern still matter after net-force cancellation | **does not fire** — `P2` is worse on rotation at both lags |
| `P3` helps at 4 ns or high T | supports the uncertainty-aware bath/gating hypothesis | **not run** (§7) |
| `P4` improves endpoint *and* physics metrics | future interaction-state consistency helps the transition representation | **not run** (§7) |

One row fires that was not in the matrix, because nobody expected to need it:
**the pair architecture beats every physics arm on Cα RMSD.** That is a finding
about the model, not about force — `P0` was built to be a null, and it won.

Rules the report holds itself to:

* **RMSD alone decides nothing.** Rotation, pair distance and contact metrics are
  reported beside it, and Phase 1.5's experience is that rotation moves first.
* Bounded, single-seed results are a **direction**, never a claim of superiority.
* Confidence intervals are cluster bootstraps over **domains**. A frame-level
  t-test on 3,520 pairs would call a 0.05% difference significant and is not used.
* Recoverability `(M(S0) − M(best)) / (M(S0) − M(O))` prints
  `not identifiable` when the oracle gap is at or below zero — which, given
  Phase 1.5, is the expected case.

---

## 9. What this says for Phase 2

Phase 1.5 said: do not build Phase 2 around richer force conditioning, because
the oracle bounds it. Phase 1.6 adds two things to that.

* **The bound holds at the edge level too.** Exposing Phase 1's per-pair
  interaction — the most informative force-supervised tensor the model has, 880
  dimensions per edge instead of 152 per node — does not beat a geometry-only
  control on placement. It buys ~0.25% of rotation, at 1.8× the compute.
* **But the pair *stage* is worth keeping.** `P0` beats `S0` by 0.5–0.4% on Cα
  RMSD, more than any physics arm beats anything. Explicit residue-pair message
  passing over the current structure, with degree-stable pooling, is a cheap
  architectural win that has nothing to do with force.

Read together with Phase 1.5: the 6–8% that every arm takes off the identity
baseline comes from structure and history; a further ~0.5% is available from
modelling residue **pairs** explicitly; force contributes ~0.2% of rotation and
nothing measurable on placement, whether it is read from a node latent, an edge
latent, explicit moments, or the ground truth itself.

The recommendation from [phase1_5_report.md](phase1_5_report.md) §5.5 stands and
narrows: put Phase 2's effort into longer history, stochastic memory and
non-Markovian structure — and, on this evidence, into the pair graph as an
architectural component rather than as a physics channel.

## 10. Tests

580 pass (was 533; +47 new, ~4 min). The new file is
[tests/test_pair_physics.py](../tests/test_pair_physics.py). The existing
arm-parametrised tests in `test_conditioners.py` and `test_transition_probe.py`
now iterate over **all ten** arms rather than five, so the shared-contract
properties — one output shape, one backbone size, one head size, invariance,
atom-order independence, zero conditioning on masked residues, gradient
isolation — are enforced on the new arms by the tests that already existed.
