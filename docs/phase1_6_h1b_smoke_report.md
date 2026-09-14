# Phase 1.6 H1b — backbone frame constraint refiner (smoke)

**Nothing in this document is a result.** Every number below comes from a
30-step smoke run or a 200-step single-batch overfit probe. Both are bounded
runs whose purpose is to establish that the stage is wired correctly and that
the refiner *can* move; neither says anything about generalisation. Full H1b
training was not run — it is on the explicit do-not-auto-run list.

Artefacts: `runs/phase1_6_h1b_smoke/`, `runs/phase1_6_h1b_overfit/`,
`runs/phase1_6_h1b_resume/`. Each carries a
`reproducibility_manifest.json` with `is_smoke: true`.

---

## 1. Why H1b runs before H1a

The original plan put H1a (side-chain torsion decoder) first. H0 measured the
two error sources and the ordering was changed on that evidence.

Replacing one half of the construction with ground truth, lag 1 ns, mean over
all seven arms, every comparison significant in 7/7 arms:

| metric | frame oracle − H0 | conformer oracle − H0 | ratio |
|---|---:|---:|---:|
| `heavy_atom_rmsd` | **+2.449 Å (+63.9%)** | +0.187 Å (+4.9%) | **13.1×** |
| `backbone_heavy_rmsd` | +2.881 Å (+89.9%) | +0.014 Å (+0.4%) | 206× |
| `sidechain_heavy_rmsd` | **+2.351 Å (+54.8%)** | +0.322 Å (+7.5%) | **7.3×** |

The last row decides it: even on side-chain RMSD — the metric H1a exists to
improve — getting the frame right is worth 7× more than getting the torsion
right. H1a attacks the smaller term.

### What this does *not* change

H1a is unaffected by running second, and this was checked rather than assumed:

- **χ is a frame invariant.** Moving a residue's frame changes no χ value, so
  H1a's targets, its measured floor (0.574 Å residue-local), and its inputs are
  all the same whichever stage runs first.
- **The two operations commute.** A Δχ rotation turns a residue's atoms about an
  axis built from that residue's own atoms, so a rigid frame correction carries
  the axis with the atoms. `test_a_chi_rotation_commutes_with_a_rigid_frame_correction`
  holds it to 1e-12. Had it failed, the two stages would have needed joint
  training.
- **The clash budget is already split.** Of the inter-residue serious overlaps
  H1b optimises, 66.6% are backbone–backbone, which no χ rotation can touch. The
  remainder is the backbone–side-chain and side-chain–side-chain overlap the
  brief had already assigned to H1a, and the metrics report those classes
  separately.

### The floor H1b cannot reach alone

Serious Bondi overlaps per structure, lag 1 ns, from the full H0 run:

| construction | inter bb–bb | inter bb–sc | inter sc–sc | intra (all) |
|---|---:|---:|---:|---:|
| identity (nothing moves) | 82.4 | 4.4 | 1.8 | 66.6 |
| **H0 transported (deployable)** | **272.3** | 105.9 | 30.7 | 66.6 |
| frame oracle | 108.3 | 38.7 | 48.4 | 66.6 |

Even a **perfect** frame prediction leaves 108.3 inter-residue bb–bb overlaps
against the identity baseline's 82.4, because the side chains are still the
stale ones. So the overlap term has a floor H1b cannot pass, and its weight must
not be tuned toward zero — that would trade away frame accuracy chasing a
residual that belongs to H1a. It is set to 0.1 and judged on the bb–bb column
only.

(The intra-residue column is identical to the last digit across all four
constructions, which is the `construction_invariant` claim confirmed on real
data: a rigid transport cannot change a residue's internal geometry, so H1b is
neither credited nor blamed for those 66.6.)

---

## 2. What H1b is

A per-residue residual SE(3) update composed onto the frozen coarse frames.

- **Rotation through the Lie algebra.** The head emits a rotation vector and
  `so3_exp_map` turns it into a rotation, so `det = +1` holds by construction
  with no orthogonalisation step, and it is well behaved at the small angles
  this head is capped to — which a quaternion normalisation is not, near zero
  norm.
- **Translation in the coarse predicted frame,** rotated into the global one by
  `compose`. Both choices make the composition equivariant: rotating the input
  rotates the output and leaves the predicted correction itself unchanged.
- **Zero-init**, so an untrained refiner is exactly the coarse prediction and
  every later number is a departure from it rather than from a random pose.
- 24,710 parameters against the frozen arm's 386,426.

### Features: 57 invariant scalars, and no probe surgery

The refiner does **not** read the coarse probe's internal node embeddings.
Doing so would have required changing a frozen code path, and it would have let
H1b borrow the coarse model's representation capacity, which a later
capacity-matched comparison depends on it not doing. Instead the features are
built from quantities that are already public: the residue's own coarse
prediction and internal geometry, the same for its two sequence neighbours, the
neighbour Cα positions seen in this residue's own current frame, the peptide
bond and both angles measured at time *t* on each side, and the lag.

Every block is a rotation- and translation-invariant scalar. Since the only
thing carrying a global frame is the correction, and that is expressed in the
coarse frame, the refined frames are equivariant **by construction**, not by
training. `test_refined_frames_rotate_with_the_input_end_to_end` checks it
through the real head with non-zero weights.

The window is the residue plus its two neighbours because every quantity Stage M
found broken — peptide C–N bond, the two angles across it, Cα–Cα spacing — is
inter-residue, and a residue that can only see itself cannot fix any of them.

### The objective

```
L = 1.0 · transition_loss(refined, target)      # the coarse arm's own loss
  + 1.0 · peptide_geometry(refined, current)    # bond + two angles, vs time t
  + 0.1 · soft_overlap(refined backbone)        # Bondi, hinged at 0.4 Å
```

The primary term is the **same** loss with the **same** weights the coarse arm
was trained with, read from the config's `train:` block rather than restated.
H1b is only worth having if it buys geometry without giving back transition
accuracy, so the original objective stays in at full weight and the frozen arm's
own value is logged beside it every step.

The geometry term regularises toward the **current structure's own** peptide
geometry. There is no ideal-geometry table in this repository and inventing one
was forbidden; the current frame is a real MD snapshot, so its geometry is
physical by construction. Reading it is not leakage — it is the model's input,
and `test_the_peptide_geometry_reference_is_measured_at_t` perturbs the future
and requires the reference not to move.

This is **soft regularisation, not a hard constraint**. Nothing here guarantees
a physical bond. Every line says `constraint-regularized`, never "guaranteed".

---

## 3. What was run

| run | budget | purpose |
|---|---|---|
| `phase1_6_h1b_smoke` | 30 steps, streamed batches | does the stage run end to end |
| `phase1_6_h1b_overfit` | 200 steps, **1 fixed batch** | can the refiner move at all |
| `phase1_6_h1b_resume` | resume at 200 → 210 | checkpoint save/resume |

Refined arm: `P0_pair_geometry_control`, chosen before any H1b number existed
because it had the best backbone RMSD of the seven in H0 (3.1927 Å vs the oracle
arm's 3.2116), so a geometry gain on top of it cannot be explained as slack the
coarse arm had left.

The coarse checkpoint's sha256 is taken before and after every run and compared;
all three report `coarse_weights_unchanged: true`. The optimiser is given the
refiner's parameters only and the arm runs in `eval()` under `no_grad`.

The rebuilt data manifest hash is checked against the checkpoint's
`provenance.json` and the run aborts on a mismatch, so H1b cannot train against
a different split from its coarse arm.

## 4. Overfit probe — 1 batch, 200 steps

**A capacity probe, not a result.** It answers "can the head represent a useful
correction at all", nothing more.

| term | step 1 | step 200 | |
|---|---:|---:|---|
| primary (transition loss) | 4.24308 | **4.00912** | frozen coarse arm: 4.24308 |
| geometry — bond | 0.18645 | 0.06922 | −63% |
| geometry — angle CA–C–N | 0.51826 | 0.11950 | −77% |
| geometry — angle C–N–CA | 0.58666 | 0.12692 | −78% |
| overlap | 0.00204 | 0.00101 | −50% |
| translation RMSE | 6.71052 Å | 6.74518 Å | +0.035 Å (worse) |
| rotation error | 50.290° | 47.719° | −2.571° (better) |
| mean correction | 0 Å / 0° | 0.4389 Å / 8.590° | |

Two things this establishes and one it flags.

**The refiner can improve geometry and the primary loss at the same time.** On
this batch the transition loss falls below the frozen arm's, driven by rotation
(−2.57°) against a small translation cost (+0.035 Å). That is not evidence it
generalises — it is one memorised batch — but it rules out the possibility that
the objective is internally contradictory.

**It does not collapse to identity.** The identity baseline has near-perfect
peptide geometry because it *is* a real MD frame, so a refiner rewarded for
geometry can score well by simply undoing the coarse prediction. The correction
magnitude is logged every step for exactly this reason, and it grows to 0.44 Å /
8.6° while the primary loss *improves* — the opposite signature from reversion,
which would show a growing correction and a degrading primary loss.

**⚠ The caps saturate.** Maximum correction reached exactly 1.00000 Å and
15.00000° against caps of 1.0 Å and 15°. At least one residue is pinned at the
bound after 200 steps on one batch. If full training saturates them too, the cap
is binding rather than protective and the value would have to be revisited — and
that decision must be made with a stated reason, before looking at the resulting
metric, not after.

## 5. Three defects found and fixed

All were found by the probe, not by review, and all are now regression tests.

### The correction cap, in three iterations

**(a) Per-component `tanh` bounded each axis, not the magnitude.** The vector
norm could therefore reach `√3 ×` the configured cap, and did: the first overfit
probe reported a maximum of 1.72272 Å and 25.96452° against caps of 1.0 Å and
15° — exactly `√3` over, in both. The caps are set against measured quantities
(mean |Δr| 2.5 Å, mean frame rotation 33.8°), so a factor of 1.73 is the
difference between "enough to fix a bond" and "enough to redo a fifth of the
transition".
→ `test_the_correction_caps_bound_the_magnitude_not_the_components`

**(b) The first fix introduced a dead fixed point.** Writing the norm cap as
`tanh(n) / clamp(n, 1e-12)` evaluates to `0 / 1e-12 = 0` at the origin, killing
both the scale *and* its gradient. A zero-init head then sits at exactly zero
correction forever. It trains without error and the loss still falls — the
coarse arm is doing the work — and the only visible symptom was `|dt| 0.0000 Å`
on every logged step.
→ `test_the_zero_init_refiner_still_has_a_gradient`

**(c) `tanh` on the norm respected the bound but was needlessly restrictive.**
`|v_out| = tanh(|v|) · cap` reaches only 76% of the cap at `|v| = 1` and
approaches it asymptotically, exactly where `tanh′ → 0` and the gradient dies —
so the head could not freely use the range it had been given. The cap is now a
hard clip, `|v_out| = min(|v|, cap)`: the identity inside the ball, a radial
projection onto the sphere outside. Below the cap the correction is untouched
and full-gradient; above it the tangential gradient survives, so the *direction*
still trains while the magnitude is held. The head's raw output is now the
correction in physical units directly.
→ `test_the_cap_is_the_identity_below_the_cap`

Measured on the same probe, same caps, the clip is better on every term:

| | per-component `tanh` | `tanh` on norm | **hard clip** |
|---|---:|---:|---:|
| primary loss | 4.08191 | 4.08191 | **4.00912** |
| geometry | 0.24606 | 0.32624 | 0.31564 |
| rotation error | — | 48.535° | **47.719°** |
| max \|Δt\| | 1.72272 Å ✗ | 1.00000 Å | 1.00000 Å |
| max \|ΔR\| | 25.96452° ✗ | 14.99999° | 15.00000° |

(The per-component column's geometry looks lowest only because it was allowed a
√3-larger correction; its caps are violated, so it is not a valid comparison.)

### Silent NaN in the physical-unit log

Two logged metrics were keyed on names `transition_loss` does not emit, so they
recorded `NaN` without complaint. They now use the real keys
(`translation_rmse_angstrom`, `rotation_error_deg`) and are the only place the
accuracy/geometry trade-off is visible in physical units.

## 6. Test coverage added

19 tests across two new files, all passing, alongside the 63 existing
heavy-atom tests. The full suite passes.

**Equivariance (15)** — feature invariance under a global rigid motion, `compose`
equivariance, `det = +1` preservation, zero-init identity, gradient at zero-init,
all three cap regressions, backmapping round-trip, overlap-penalty invariance,
Δχ/rigid-motion commutation, and the end-to-end refined-frame equivariance
through the real head.

**Leakage (4)** — behavioural, not structural. The same batch is built against
two genuinely different futures and the deployable outputs must be
*bit-identical*; the oracle modes must, conversely, change. A structural check
("this module does not import that one") would pass for a leak routed through a
shared object; this cannot. Covers the refiner features, the peptide-geometry
reference, the deployable placement, and the identity baseline's independence
from both the future and the model.

## 7. What full H1b would need

**Superseded — full H1b has since been run at the user's request. See
[`phase1_6_h1b_results.md`](phase1_6_h1b_results.md).** All four items below were
addressed: the evaluator exists and its coarse control reproduces Stage M
bit-for-bit, the cap decision was made and recorded before training (kept, and
subsequently vindicated at 14.6%/6.3% saturation), and both the validity
comparison and the primary-metric cost are reported. This section is kept as the
record of what was required beforehand.

It needed, in order:

1. A held-out evaluation script reporting the **full PSF-based H0 metrics** on
   refined frames — the training penalty is a deliberately strict subset (it
   drops the 1-4 pairs the metric keeps) and must never be quoted as the result.
2. A decision on the caps, given §4's saturation, made and recorded before
   seeing the resulting metrics.
3. The direct Stage M comparison: peptide C–N bond, backbone angles, Cα
   neighbour distance and clash rate against both the coarse arm and the
   identity baseline, since identity beat every trained arm on all ten validity
   cells and that is the bar.
4. Confirmation that the primary transition metrics — Cα RMSD, dRMSD, rotation —
   have not regressed, reported as a cost if they have.
