# Open questions

Decisions that are deliberately unresolved, with the reasoning and what would
close them. Each one is a place where guessing would have produced
plausible-looking but unjustified code.

---

## 1. `L_local_geometry` is undefined in the design document

**Status:** open — substituted, not implemented as specified.

The Phase 1 loss in the design document lists

```
L_phase1 = ... + lambda_geom * L_local_geometry
```

but never defines `L_local_geometry`. Inventing a physics-shaped term to fill the
slot is exactly the "plausible ad-hoc formula" the same document forbids, so it
was not invented.

**What is implemented instead.** `physics/losses.py` carries an *energy gauge*
term with weight `1e-4`:

```
lambda_gauge * mean(U^2)
```

This is genuinely needed and is not a stand-in for a geometry law. mdCATH ships
no energy label, so `U` is supervised only through its gradient; that leaves the
additive offset and overall scale of `U` unconstrained and free to drift. The
gauge term bounds them. It is not a fit to any measured energy, and its docstring
says so.

**To close this:** state the intended definition. Plausible readings, any of
which is straightforward to add:

- a bond-length / bond-angle deviation penalty on predicted or rolled-out
  geometry (would only become meaningful in Phase 2, where geometry is generated);
- a penalty on the predicted force component along constrained bonds (mdCATH
  used rigid X–H constraints, so force along those bonds is not free);
- a smoothness prior on predicted forces between neighbouring frames.

Until then `lambda_geom` does not exist in `LossWeights`; adding it later is
additive and breaks nothing.

---

## 2. Nanoseconds per frame — RESOLVED

**Status:** resolved 2026-08-16 — **1 ns per frame**, confirmed by the user from
the mdCATH paper.

The shards themselves contain no per-frame timestamp, and `MDensemble` (the
sibling project on the same data) had independently declined to hard-code a value
after finding external sources inconsistent. The value is stated in the
publication, not inferred from the files.

**Recorded as** `MDCATH_PS_PER_FRAME = 1000.0` in `force_md.data.units`, and set
explicitly per run in `configs/phase1_small.yaml`. Phase 1 does not use it —
force supervision is per frame. Phase 2's 1/2/4/8/16 ns multi-lag training does:
those lags are 1/2/4/8/16 frames.

## 3. Upstream `forces == coords` defect, cause unknown

**Status:** characterised, mitigated, not explained.

5 of 1000 downloaded domains (`1aq3A00`, `1b4rA00`, `1cjyA01`, `1cukA01`,
`1gk9B03`) ship a `forces` array byte-identical to `coords`, always in exactly
the same pattern: replicas 0–3 corrupt at all five temperatures, replica 4 intact.
All 1000 files match the HuggingFace repo's recorded sizes, so this is upstream in
the published dataset, not a download artifact.

**What is implemented:** `scripts/audit_mdcath_forces.py` writes a per-trajectory
quarantine list; the adapter masks those trajectories' force labels via
`force_valid` rather than dropping the shards, because their coordinates are fine
and remain usable for Phase 2 transitions.

**Not closed:** why the pattern is exactly replicas 0–3. Worth reporting upstream.

---

## 3b. Upstream saturated-coordinate defect

**Status:** characterised, mitigated, cause understood, not reported upstream.

87 frames across 15 domains have **every atom** at `2.14748e7 Å`. That number is
`INT_MAX / 1000 nm`: the XTC format stores coordinates as `int32` in units of
1/1000 nm, and the writer saturated instead of erroring. The frame carries no
position information at all — it is not a protein that drifted.

Affected trajectories (one per domain, always exactly one): `1gkgA02/320/4`,
`1y66A00/450/0`, `1zxqA02/348/2`, `2akcA00/320/1`, `2jbrA03/413/3`,
`2jysA00/320/4`, `2wdcA03/348/3`, `2y94A03/348/3`, `3b0dC00/320/3`,
`3gk0A00/348/0`, `3j7aN00/413/1`, `4ekuA01/320/2`, `4o30B00/379/3`,
`4qpiC00/348/1`, `4tmpA00/348/4`.

**Why `check_pbc` cannot catch it.** That check compares consecutive CA–CA
distances. With every atom at the same saturated value the CA–CA distances are
exactly zero, so a fully corrupt frame looks *more* intact than a real one. This
needed its own audit, in the same way the `forces == coords` defect did.

**What is implemented:** `scripts/audit_mdcath_coords.py` scans every frame
(11,603,224 of them) and writes `mdcath_coord_quarantine.json`; the adapter's
`coord_quarantine_path` drops those frames while the index is built, and samples
`frames_per_trajectory` evenly over the survivors so an affected trajectory is not
silently thinned. `MdCathDataset.__getitem__` keeps a magnitude backstop that
raises with the domain/temperature/replica/frame named, so an unaudited shard
fails legibly instead of producing a NaN several hours into a run.

The surviving frames of all 15 trajectories were checked and are physically sane
(CA–CA 2.77–4.16 Å), so only the corrupt frames are dropped, not the trajectories.

**Why this mattered more than 87/11,603,224 suggests.** The affected trajectories
are short (40–475 frames), and the run samples 40 frames per trajectory — so the
corrupt frames were sampled at close to certainty. The first full launch died of
this at step ~300.

---

## 5. Non-finite training steps — RESOLVED

**Status:** resolved. Cause found, quarantined, verified by frequency.

**The cause is one frame.** `2qenA03/348/3` frame 447 has all 1194 atoms at
exactly `(0, 0, 0)` — a final frame allocated but never written. Forces in it are
normal. Every non-finite step across four runs came from this one frame.

**Verified two independent ways.** The corpus-wide extent audit finds exactly one
collapsed frame; the runs show exactly one skipped step per epoch:

| epoch | steps | skip |
|---|---|---|
| 0 | 1–7,562 | 4,987 |
| 1 | 7,563–15,124 | 10,939 |
| 2 | 15,125–22,686 | 20,285 |
| 3 | 22,687–30,248 | 27,854 |

One frame drawn once per epoch. The count and the rate agree.

**Why it took four runs.** Three things hid it, and each defeated a different
check:

1. *A magnitude audit cannot see it.* All coordinates are exactly zero, so
   `|coord| = 0` passes any threshold however tight. The audit was written
   around the saturated-coordinate defect and inherited its invariant. The right
   invariant is **spatial extent** — a protein occupies space — and the audit now
   checks that.
2. *It is sampled with certainty.* `np.linspace(0, n-1, k)` always includes the
   last frame, so evenly spaced sampling draws a broken final frame every epoch.
   A rare defect and a sampler that seeks it out are not independent.
3. *DDP hid which rank was at fault.* Coincident atoms make the backward produce
   NaN, and the gradient all-reduce copies that NaN to every rank. All four ranks
   then report an identical failure, and on three of them the loss is perfectly
   finite — so the evidence reads as "finite forward, NaN backward", pointing at
   the model rather than at one rank's data. Only rank 2's dump showed a NaN loss.

**Contained by** the collective non-finite skip in `Phase1Trainer.train_step`,
which cost one batch in 7,562 while this was unresolved, and would have cost the
whole run without it. The dump hook (`nonfinite_dump_dir`) is what finally
produced the evidence: weights, batch and gradients from the failing step, on
every rank.

**Regression test:** `test_collapsed_frame_is_quarantined`.

---

## 5b. Superseded — the reasoning that did not work

Kept because the dead ends are worth not repeating. Rejected hypotheses, each
tested against the whole corpus rather than a sample:

| hypothesis | test | result |
|---|---|---|
| learning rate 3e-3 diverging | 3-arm sweep, then two full runs | wrong — run 1 died of the frame, not the lr |
| saturated coordinates | 11,603,224 frames | 87 found and quarantined, but not these |
| non-finite/absurd forces | 11,603,224 frames | 0 found, max 277.6 kcal/mol/A |
| coincident atoms in `2akcA00` | all 991 sampled frames | min connected distance 1.107 A |
| `(chain, resid)` collision merging chains | 1000 domains | 0; every domain is single-chain |
| unguarded division in the model | `frames`/`radial`/`edges`/e3nn | all clamp before dividing |
| a NaN born in the backward pass | replay with the failing weights | forward *and* backward clean on rank 0 |

The last row is the instructive one. Replaying rank 0's dump reproduced nothing
on either device, which looked like evidence that the data was innocent. It was
evidence that **rank 0's data** was innocent. Checking one rank of a four-rank
all-reduce and generalising from it was the error that cost the most time here.

---

## 6. The pipeline computes in absolute box coordinates

**Status:** open — measured, fix identified, not yet applied.

mdCATH coordinates are absolute positions in the simulation box and reach ~460 Å.
Everything the model consumes is a *difference* of two of them — edge vectors,
lever arms `x_a - r_i`, backbone frame axes — and those differences are a few
Ångström. Nothing in the pipeline centres the coordinates first.

This is visible, not theoretical: switching matmuls to TF32 moves the loss on the
batch above from **2.43 to 12.68**. TF32 has a 10-bit mantissa, so at 460 Å its
resolution is ~0.45 Å — comparable to the lever arms being computed. Plain float32
has 24 bits and far more headroom, but the same cancellation is structurally
present, and it is a plausible contributor to item 5.

**The fix** is to subtract the per-graph centroid before the model sees the batch.
The model is translation invariant by construction, so this is physically a no-op;
it simply stops spending ~8 mantissa bits on an offset that is never used.

---

## 4. Conservative force is differentiated w.r.t. atom positions only

**Status:** resolved in the model, worth re-checking against the real adapter.

`-dU/dx` is taken with respect to `atoms.positions`. When backbone N/CA/C
positions are stored as separate tensors (as the synthetic fixture does), part of
the geometry dependence would bypass that gradient.

**What is implemented:** `LocalPhysicsModel` relinks backbone positions to be
gathers of the atom position tensor before differentiating, so the gradient is
complete. The adapter should produce batches that already satisfy this.
