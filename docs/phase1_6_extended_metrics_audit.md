# Phase 1.6 extended metrics — Step 0 audit

Written **before** any code changed, against the working tree at commit
`9370c194238c` (dirty). Everything below was read out of the source or measured
from the saved artefacts; nothing is assumed from the task brief.

Companion to `docs/phase1_6_audit.md`, which audited the *arms*. This one audits
the **prediction/target contract and the metrics**, because the extended suite is
only meaningful if it inherits the existing definitions exactly rather than
inventing parallel ones.

---

## 1. Reproducibility metadata — brief vs. repository

Every hash the brief predicted is confirmed. Measured with `sha256sum`.

| item | brief | measured | verdict |
|---|---|---|---|
| dataset manifest | `24a1abad503a` | `24a1abad503af9d98fdb96ba217a687feb7721ccfcafa0b83e721555d78ee495` | ✅ |
| Phase 1 checkpoint | `94b0d7c02e55` | `94b0d7c02e5570b3d08864d9cd28e403b9c5ab3210d18c63b537112c387dca9c` | ✅ |
| config | `aeab18db40c6` | `aeab18db40c6c0bf32b6a476ecdc551ad69f4f402a7f8eb365259b2ace26459a` | ✅ |
| git commit | `9370c194238c` | `9370c194238cc1ac145de1baf8d77c10cc0dea58` | ✅ |
| dirty working tree | `true` | `true` (17 modified, 12 untracked) | ✅ |

Transition checkpoints (`runs/phase1_6_bounded_seed0/<arm>/last.pt`):

| arm | sha256 (12) |
|---|---|
| `S0_structure_history` | `45d6731cc6c2` |
| `S1_residue_wrench` | `8f176b7393c0` |
| `S2_node_physics_152d` | `7b19014e39c4` |
| `O_current_gt_force_oracle` | `83b576756150` |
| `P0_pair_geometry_control` | `060ad6adf53f` |
| `P1_pair_physics_frozen` | `98adbf48368a` |
| `P2_pair_physics_moments` | `3df6944c8da0` |

Environment: Python 3.11.15, PyTorch 2.13.0+cu130, e3nn 0.6.0, NumPy 2.4.6,
8× NVIDIA H100 80GB. Model dtype float32; Kabsch SVD and all new pair/torsion
arithmetic in float64. `mdtraj`, `Biopython`, `pandas`, `pyarrow` and
`matplotlib` are **not installed** in the `md` env — this constrains three
metrics, recorded in §6.

Validation set as predicted: 30 held-out domains, 1,760 pairs per lag at
1 ns and 4 ns, 3,520 unique `pair_id`s, 704 pairs per temperature group
(320/348/379/413/450 K). Verified by loading `val_records.json`.

---

## 2. Prediction / target contract (§3.1 of the brief)

Read from `src/force_md/transition/targets.py` and
`src/force_md/geometry/{frames,alignment,so3}.py`.

### 2.1 What a prediction is

`TransitionPrediction` carries `translation_local [N_res,3]` and
`rotation [N_res,3,3]`, both **in the current residue's local frame**:

```
delta_r_local_i = R_cur_i^T (CA_fut_aligned_i - CA_cur_i)
R_rel_i         = R_cur_i^T R_fut_aligned_i
```

`apply_prediction()` maps them back to the current structure's global frame:
`ca = CA_cur + R_cur · delta_r_local`, `R = R_cur · R_rel`.

| question | answer |
|---|---|
| translation origin | **Cα.** `BackboneFrameBatch` documents `r_i = ca_positions[i]`; `ResidueFrames.origin` is the Cα. Not a residue centroid. |
| rotation convention | **Local axes as columns**, so `R_i` maps local → global. `relative_rotation(A,B) = Aᵀ B`; a global rotation acts on the **left**: `R → Q R` (`RigidAlignment.apply_frames`). This is exactly the brief's §4.2 convention — no adaptation needed. |
| target frame reconstruction | `build_residue_frames(N, CA, C)` on the **Kabsch-aligned** future N/CA/C. |
| local atom coordinates in the prediction | **No.** The model emits a rigid frame update only. `reconstruct_backbone()` carries the residue's *current* internal geometry (`local_n`, `local_c`) onto the predicted frame. Consequences in §5. |
| reconstructable backbone atoms | **N, Cα, C only.** There is no O and no Cβ in the reconstruction; `target.local_n / local_c` are the only internal coordinates stored. |
| chain break / missing atoms | `TransitionTarget.valid = current_frames.valid & future_frames.valid & residues.mask & alignment.valid[graph]`. Sequence adjacency comes from `sequence_neighbours()`, which requires same graph, same chain **and** `resid_original` differing by exactly 1 — so a numbering gap is a chain break and torsions are not computed across it. |
| padding | There is no padding: the batch is ragged/flattened with `residue_batch_index`. "Padding" in the brief maps onto `~valid`. |
| metadata | `LagPair` carries `domain`, `temperature`, `replica`, `current_frame`, `lag_frames`, `lag_ps`, and `pair_id = "{domain}/{T}/{replica}/t{cur}/f{fut}/lag{ps}ps"`. `future_frame = current_frame + lag_frames`. All available at evaluation time via `LagPairBatch.pairs`. |

### 2.2 Where Kabsch is applied

**Once, inside `build_transition_target`,** fitting the *future* Cα onto the
*current* Cα over residues valid in both structures, weighted by that mask,
`det = +1` enforced, SVD in float64. The whole future structure (N, Cα, C) is
then moved by that rigid motion. Nothing else is aligned — the model's input is
never aligned, because the alignment is computed *from the future* and feeding it
forward would leak the label.

So the prediction and the target already live in a common frame: the current
structure's global frame. **No second superposition is required for the metrics
to be rigid-motion invariant**, and the existing `ca_rmsd` does not do one.

---

## 3. Existing metric audit (§3.2 of the brief)

Read from `src/force_md/transition/metrics.py`.

| item | finding |
|---|---|
| `ca_rmsd` | `sqrt(mean_i ‖pred_ca_i − targ_ca_i‖²)` over that graph's valid residues, **no second Kabsch**. This is the number in `results.csv` and in `docs/phase1_6_results_bounded.md`. |
| `ca_rmsd_aligned` | A *separate* key that re-fits Kabsch of `pred_ca` onto `targ_ca` per graph (needs `n ≥ 3`). It exists in `val_records.json` but was **never promoted to the report**. |
| `translation_rmse` | `sqrt(mean ‖Δ̂_local − Δ_local‖²)`. Since the residue origin is Cα and `R_cur` is orthogonal, this is **algebraically identical to `ca_rmsd`**. Confirmed numerically on the saved records: over all 24,640 rows (3,520 pairs × 7 arms) the maximum absolute difference is `1.9e-6 Å` and the maximum relative difference `2.1e-7` — float32 round-off, not a difference in definition. → `translation_rmse_is_ca_rmsd = true`, and it is **not** interpreted as an independent metric. |
| rotation geodesic | `rotation_geodesic_angle(pred_R, targ_R)` = angle of `pred_Rᵀ targ_R`, computed with `atan2(‖vee((R−Rᵀ)/2)‖, (tr(R)−1)/2)` — **not** `arccos`, which is ill-conditioned exactly where a good model sits. Reported as the **mean** over residues, in **degrees**. No median was stored. |
| reflection correction | Present and tested (`test_kabsch_never_returns_a_reflection`). `det(Q) = +1` enforced by the sign correction on the SVD. |
| residue vs sample weighting | **This is the one real inconsistency.** `transition_metrics()` and `aggregate_metric_records()` weight each graph by its valid-residue count (`*_micro`) or average within domain first (`*_domain_macro`). The runner's console line and `results.csv` therefore print `ca_rmsd_micro = 2.8766` for S0 @ 1 ns. `scripts/analyze_phase1_6.py`, which produced the published report, instead takes a plain **`np.mean` over the per-pair records** → `3.2079`. Both are legitimate; they are different quantities and the report never says which. The extended suite makes the sample-equal mean the stated primary (brief §12.1) and reports the residue-weighted one alongside, labelled. |
| `pair_id` on every record | Yes, since the `reevaluate_phase1_6.py` pass. All 3,520 rows carry one and all 3,520 are unique, in all 7 arms. |
| duplicate-key behaviour | `analyze_phase1_6.key_of()` raises `KeyError` when `pair_id` is absent, but a **duplicate** `pair_id` silently overwrites in `by_lag()`'s dict. The extended loader raises instead (brief §16.19). |
| contact definition in use | `contact_cutoff = 8.0 Å` on **Cα**, `contact_sequence_separation = 3` on `resid_original` within a chain. The brief's primary is separation **≥ 6**; both are computed and reported separately, the sep-3 one only as the legacy reproduction. |
| empty-contact samples | Existing code sets `contact_f1 = 0.0` when precision and recall are both finite but sum to zero, and NaN when a denominator is empty. The extended suite reports `NA` and counts those samples, per brief §7.2. |
| clash | `clash_rate` = fraction of Cα–Cα pairs with `|resid_i − resid_j| ≥ 2` closer than **3.6 Å**. The threshold is sourced (below the audited 3.67–4.01 Å bonded Cα–Cα distance in this dataset), not invented. It is a **Cα-only** metric and is *not* an all-atom clash rate. |
| NaN handling | `_weighted_mean` skips NaN. NaN, never 0, is used for "not defined here". That convention is kept. |

---

## 4. Rotation alignment contract — the decision

The brief §4.2 says to apply the Kabsch rotation `Q` to the predicted frames
before comparing. In this repository the relevant `Q` is already **inside the
target**: the single global fit that canonicalised the future. Relative to that
common frame there is no further rotation, i.e. the brief's `Q` is the identity
for the existing definition.

Two variants are therefore computed and named, rather than one being silently
substituted for the other:

| key | definition | role |
|---|---|---|
| `ca_rmsd` | canonical frame, no re-superposition | **co-primary**; reproduces the existing seed-0 number |
| `rotation_geodesic_mean_deg` | `angle(R̂ᵀ R)` in the canonical frame | **key secondary**; reproduces the existing seed-0 number |
| `ca_rmsd_superposed` | extra per-sample Kabsch of `pred_ca` onto `targ_ca` (`= ca_rmsd_aligned`) | diagnostic |
| `rotation_geodesic_superposed_mean_deg` | same `Q` applied as `R̂ → Q R̂`, then `angle((QR̂)ᵀ R)` | diagnostic; makes the two views mutually consistent, which the existing pair `ca_rmsd_aligned` + `rotation_geodesic_deg` was **not** |

Both are tested for invariance under a global rigid motion of the whole pair, and
`rotation_geodesic_superposed_*` is tested to be invariant when a common global
rotation is applied to the prediction alone (the existing non-superposed one is
not, and must not be — it would then be measuring a different thing).

---

## 5. What the reconstruction contract makes measurable — and what it does not

`reconstruct_backbone()` places predicted N and C by carrying the residue's
**current** `local_n` / `local_c` onto the predicted frame, and does the same for
the target. Both sides therefore carry identical intra-residue internal geometry.
The consequence is exact and was verified numerically:

| quantity | spans | measurable? |
|---|---|---|
| N–Cα bond | inside a residue | **No.** Identical in prediction and target by construction; error is exactly 0. |
| Cα–C bond | inside a residue | **No.** Same. |
| C–N(next) peptide bond | between residues | **Yes.** |
| N–Cα–C angle | inside a residue | **No.** Same. |
| Cα–C–N(next) angle | between residues | **Yes.** |
| C–N(next)–Cα(next) angle | between residues | **Yes.** |
| φ = (C₍ᵢ₋₁₎, Nᵢ, Cαᵢ, Cᵢ) | between residues | **Yes.** |
| ψ = (Nᵢ, Cαᵢ, Cᵢ, N₍ᵢ₊₁₎) | between residues | **Yes.** |
| ω = (Cαᵢ, Cᵢ, N₍ᵢ₊₁₎, Cα₍ᵢ₊₁₎) | between residues | **Yes.** New — `backbone_torsions()` computed φ and ψ only. |
| Cα(i)–Cα(i+1) distance | between residues | **Yes.** |
| Cβ chirality | needs Cβ | **No.** There is no Cβ in the reconstruction, and carrying the current local Cβ onto the predicted frame would make chirality invariant by construction too — a metric that cannot fail is not a measurement. Recorded as `not_applicable`. |

The three intra-residue quantities are reported as **null with a reason**, not as
0.0, so no table can read them as a passed check. A test asserts they are
identically zero, which is the evidence for the claim.

---

## 6. Metrics constrained by the environment

| metric | status | reason |
|---|---|---|
| canonical bond-length / bond-angle violation rate | `not_applicable` | No canonical range exists in this repository (`residue_constants.py` has no bond table) and no chemistry library is installed. Brief §9.1 forbids inventing a threshold after seeing results, so only target-relative MAE/RMSE is reported. |
| all-atom / heavy-atom clash | `not_applicable` | The model predicts frames, not atoms; there are no side-chain coordinates to clash. The existing **Cα-only** clash metric is reused and is labelled `ca_only_min_distance_3.6A_sep>=2` in every record and table. |
| Cβ chirality | `not_applicable` | §5 above. |
| DSSP secondary-structure stratification | `not_applicable` | Neither `mdtraj` nor `Biopython` is installed in the `md` env, and this task must not add a dependency to an evaluation of frozen checkpoints. |
| RMSF-based rigid/flexible stratification | `not_applicable` | Would need a reference trajectory. The manifest samples 4 frames per trajectory; an RMSF over 4 frames is not an RMSF. Recorded rather than approximated. |

Stratifications that **are** available and implemented: lag, temperature, domain,
protein-length bin, and sequence-separation class (the dRMSD subsets).

---

## 7. Current spatial-edge subset — how it is built without touching the future

`TransitionProbe._graph()` builds its residue graph as
`build_sequence_edges(max_offset=2)` merged with
`build_knn_edges(ca_positions, k=residue_knn=16, cutoff=backbone_cutoff=13.0)`,
on the **current** Cα positions, rebuilt every forward pass.

`drmsd_current_spatial_edges` reuses exactly `build_knn_edges` on the current
Cα, with the same `k` and `cutoff` read from the arm's own saved probe config —
not from a constant here. The edge set is deduplicated to `i < j` and restricted
to pairs whose endpoints are both valid. Predicted and target distances are then
evaluated on that one shared edge set. No future coordinate participates in
selection; a test asserts the selection is byte-identical when the future
structure is replaced by noise.

---

## 8. Blockers

None. The re-evaluation needs no dataset change, no retraining and no change to
any saved weight.

---

## 9. Decisions taken during implementation

Recorded here because each one is a choice a reader could reasonably have made
differently, and none of them was made after seeing a result.

### 9.1 F1 where a denominator is empty

`metrics.py` returns NaN whenever precision *or* recall is undefined. That is the
wrong answer for the identity baseline's **event** metrics: it predicts no formed
contact at all, so its precision is `0/0`, and reporting NaN would remove it from
the formed-contact table entirely — the one arm whose behaviour that table exists
to expose.

The extended suite therefore uses `F1 = 2·TP / (2·TP + FP + FN)`, which is
algebraically the harmonic mean wherever the harmonic mean is defined and stays
defined when only one of the two is. Consequences, all intended:

| case | precision | recall | F1 | jaccard |
|---|---|---|---|---|
| no event in target, none predicted | NA | NA | **NA** | NA |
| events happened, none predicted (identity) | NA | 0.0 | **0.0** | 0.0 |
| no event happened, some predicted | 0.0 | NA | **0.0** | 0.0 |

`contact_f1_legacy_sep3` uses a **separate** function that reproduces
`metrics.py`'s edge cases exactly, so the reproduction check compares like with
like.

### 9.2 `drmsd_current_spatial_edges` uses the kNN edges only

`TransitionProbe._graph` merges sequence edges (±1, ±2) with Cα-kNN edges. The
subset here takes the **kNN** half only, since the sequence half is already the
`drmsd_local` subset and including it would make this metric a blend of two
things. `k` and the cutoff are read from each checkpoint's own probe config, not
from a constant.

### 9.3 Cross-chain pairs count as long-range

`_separation` gives residues on different chains a sentinel above every real
sequence separation, so a through-space relation between two chains lands in
`drmsd_long_range` and in the contact map. The alternative — excluding them — would
drop exactly the pairs a fold is made of. The mdCATH domains here are
single-chain, so this affects nothing in practice and is stated for the general
case.

### 9.4 Two aggregations, both printed

The report's primary is the **sample-equal** mean (every evaluation pair counts
once). The **residue-weighted** mean that `results.csv` prints is printed beside
it under its own name, because the two differ by ~10% and the Stage B report did
not say which it was quoting (§3 above).

### 9.5 Bootstrap

10,000 domain-cluster resamples, seed `20260827`, both recorded in
`paired_deltas.csv`. The implementation resamples per-domain sums and counts
rather than rebuilding the concatenated value vector; that is algebraically the
same estimator and is pinned to the Stage B implementation by
`test_fast_cluster_bootstrap_matches_the_stage_b_implementation`.
