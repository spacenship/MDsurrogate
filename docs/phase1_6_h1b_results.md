# Phase 1.6 H1b — backbone frame constraint refiner, results

Records: `runs/phase1_6_h1b_eval_seed0/records.jsonl`  
Refiner: `/data1/miplab/wjyang/MDsurrogate/runs/phase1_6_h1b_full_seed0/refiner_best.pt` at step 6000, on `P0_pair_geometry_control`  
Bootstrap: 10,000 resamples over domains, seed 20260827  
Neither the coarse checkpoint nor the refiner checkpoint was modified.

- 10560 records, 3520 per stage, 3520 pairs, 30 domains.

## 1. Does the coarse control reproduce Stage M?

- `P0_pair_geometry_control`, 3520 pairs matched against `records.jsonl`, over 10 metrics.
- **Reproduces exactly** (max absolute difference 0). The H1b harness and the Stage M harness agree bit for bit, so a refined number below can be quoted beside a Stage M number.

## 2. Physical validity — the ten Stage M cells

**Physical validity, lag 1 ns** — lower is better throughout. The `x identity` columns are the Stage M framing: how many times worse than doing nothing.

| metric | identity | coarse arm | refined | coarse ×id | refined ×id |
|---|---:|---:|---:|---:|---:|
| peptide C-N bond RMSE (A) | 0.07881 | 0.32733 | **0.15219** | 4.2× | 1.9× |
| backbone angle MAE (deg) | 4.79864 | 17.58407 | **7.54366** | 3.7× | 1.6× |
| consecutive Ca distance MAE (A) | 0.07384 | 0.38965 | **0.13330** | 5.3× | 1.8× |
| Ca clash rate | 0.00000 | 0.00137 | **0.00065** | 452.6× | 215.4× |
| backbone torsion MAE (deg) | 19.18086 | 21.01333 | **20.40907** | 1.1× | 1.1× |

**Physical validity, lag 4 ns** — lower is better throughout. The `x identity` columns are the Stage M framing: how many times worse than doing nothing.

| metric | identity | coarse arm | refined | coarse ×id | refined ×id |
|---|---:|---:|---:|---:|---:|
| peptide C-N bond RMSE (A) | 0.07882 | 0.44144 | **0.23362** | 5.6× | 3.0× |
| backbone angle MAE (deg) | 4.83274 | 24.17441 | **11.89605** | 5.0× | 2.5× |
| consecutive Ca distance MAE (A) | 0.07378 | 0.54827 | **0.23265** | 7.4× | 3.2× |
| Ca clash rate | 0.00000 | 0.00479 | **0.00326** | 2040.4× | 1389.8× |
| backbone torsion MAE (deg) | 22.25252 | 27.02938 | **25.64525** | 1.2× | 1.2× |

## 3. The cost: primary transition metrics

**Primary transition metrics, lag 1 ns** — what the geometry must not be bought with.

| metric | identity | coarse arm | refined | refined − coarse |
|---|---:|---:|---:|---:|
| Ca RMSD (A) | 3.41782 | 3.19129 | **3.22454** | -0.03326 ✗ |
| dRMSD, |i-j| >= 6 (A) | 2.60301 | 2.45810 | **2.46943** | -0.01134 ✗ |
| frame rotation error (deg) | 37.79857 | 35.87317 | **35.82195** | 0.05122 ✓ |
| contact F1 | 0.70349 | 0.70215 | **0.70698** | 0.00483 ✓ |
| formed-contact F1 | 0.00000 | 0.16757 | **0.16919** | 0.00162 ✓ |

`refined − coarse` is sign-normalised so **positive means refined is better**, for lower- and higher-is-better metrics alike.

**Primary transition metrics, lag 4 ns** — what the geometry must not be bought with.

| metric | identity | coarse arm | refined | refined − coarse |
|---|---:|---:|---:|---:|
| Ca RMSD (A) | 4.65133 | 4.27766 | **4.31830** | -0.04064 ✗ |
| dRMSD, |i-j| >= 6 (A) | 3.60527 | 3.40855 | **3.41450** | -0.00595 ✗ |
| frame rotation error (deg) | 47.54745 | 45.44413 | **45.40969** | 0.03444 ✓ |
| contact F1 | 0.62375 | 0.62552 | **0.62977** | 0.00425 ✓ |
| formed-contact F1 | 0.00000 | 0.15145 | **0.15761** | 0.00616 ✓ |

`refined − coarse` is sign-normalised so **positive means refined is better**, for lower- and higher-is-better metrics alike.

## 4. Paired deltas

Paired Δ over shared samples, lag 1 ns, domain-cluster bootstrap. **Positive = candidate better.**

| comparison | metric | Δ [95% CI] | relative | significant |
|---|---|---|---|---|
| refined − coarse | peptide C-N bond RMSE (A) | 0.17514 [0.16031, 0.18996] | +53.5% | yes |
| refined − coarse | backbone angle MAE (deg) | 10.04041 [7.57630, 12.77270] | +57.1% | yes |
| refined − coarse | consecutive Ca distance MAE (A) | 0.25634 [0.20023, 0.31742] | +65.8% | yes |
| refined − coarse | Ca clash rate | 0.00072 [0.00034, 0.00116] | +52.4% | yes |
| refined − coarse | backbone torsion MAE (deg) | 0.60426 [0.40582, 0.84562] | +2.9% | yes |
| refined − coarse | Ca RMSD (A) | -0.03326 [-0.04067, -0.02661] | -1.0% | yes |
| refined − coarse | dRMSD, |i-j| >= 6 (A) | -0.01134 [-0.01454, -0.00778] | -0.5% | yes |
| refined − coarse | frame rotation error (deg) | 0.05122 [0.03249, 0.06926] | +0.1% | yes |
| refined − coarse | contact F1 | 0.00483 [0.00295, 0.00719] | +0.7% | yes |
| refined − coarse | formed-contact F1 | 0.00133 [-0.00270, 0.00522] | +0.8% | no |
| *H1b's own effect: refined vs the frozen arm* | | | | |
| refined − identity | peptide C-N bond RMSE (A) | -0.07338 [-0.08871, -0.05913] | -93.1% | yes |
| refined − identity | backbone angle MAE (deg) | -2.74502 [-3.87597, -1.74654] | -57.2% | yes |
| refined − identity | consecutive Ca distance MAE (A) | -0.05946 [-0.08433, -0.03761] | -80.5% | yes |
| refined − identity | Ca clash rate | -0.00065 [-0.00110, -0.00028] | -21443.3% | yes |
| refined − identity | backbone torsion MAE (deg) | -1.22821 [-2.13400, -0.42351] | -6.4% | yes |
| refined − identity | Ca RMSD (A) | 0.19328 [0.15238, 0.23962] | +5.7% | yes |
| refined − identity | dRMSD, |i-j| >= 6 (A) | 0.13357 [0.10909, 0.16279] | +5.1% | yes |
| refined − identity | frame rotation error (deg) | 1.97663 [1.87160, 2.08703] | +5.2% | yes |
| refined − identity | contact F1 | 0.00429 [-0.00006, 0.00788] | +0.6% | no |
| refined − identity | formed-contact F1 | 0.16967 [0.15630, 0.18235] | +nan% | yes |
| *against the bar Stage M set: refined vs nothing moves* | | | | |
| coarse − identity | peptide C-N bond RMSE (A) | -0.24852 [-0.27722, -0.22063] | -315.4% | yes |
| coarse − identity | backbone angle MAE (deg) | -12.78543 [-16.62246, -9.34580] | -266.4% | yes |
| coarse − identity | consecutive Ca distance MAE (A) | -0.31580 [-0.40098, -0.23864] | -427.7% | yes |
| coarse − identity | Ca clash rate | -0.00137 [-0.00226, -0.00062] | -45162.1% | yes |
| coarse − identity | backbone torsion MAE (deg) | -1.83247 [-2.91108, -0.87604] | -9.6% | yes |
| coarse − identity | Ca RMSD (A) | 0.22653 [0.17990, 0.27959] | +6.6% | yes |
| coarse − identity | dRMSD, |i-j| >= 6 (A) | 0.14491 [0.12045, 0.17374] | +5.6% | yes |
| coarse − identity | frame rotation error (deg) | 1.92540 [1.80980, 2.04786] | +5.1% | yes |
| coarse − identity | contact F1 | -0.00054 [-0.00610, 0.00359] | -0.1% | no |
| coarse − identity | formed-contact F1 | 0.16834 [0.15277, 0.18311] | +nan% | yes |
| *the Stage M gap this stage set out to close* | | | | |

Paired Δ over shared samples, lag 4 ns, domain-cluster bootstrap. **Positive = candidate better.**

| comparison | metric | Δ [95% CI] | relative | significant |
|---|---|---|---|---|
| refined − coarse | peptide C-N bond RMSE (A) | 0.20782 [0.18755, 0.22943] | +47.1% | yes |
| refined − coarse | backbone angle MAE (deg) | 12.27836 [9.97450, 14.73398] | +50.8% | yes |
| refined − coarse | consecutive Ca distance MAE (A) | 0.31562 [0.26118, 0.37299] | +57.6% | yes |
| refined − coarse | Ca clash rate | 0.00153 [0.00085, 0.00230] | +31.9% | yes |
| refined − coarse | backbone torsion MAE (deg) | 1.38413 [1.05263, 1.76084] | +5.1% | yes |
| refined − coarse | Ca RMSD (A) | -0.04064 [-0.04783, -0.03425] | -0.9% | yes |
| refined − coarse | dRMSD, |i-j| >= 6 (A) | -0.00595 [-0.01168, 0.00031] | -0.2% | no |
| refined − coarse | frame rotation error (deg) | 0.03444 [0.00910, 0.05976] | +0.1% | yes |
| refined − coarse | contact F1 | 0.00425 [0.00309, 0.00550] | +0.7% | yes |
| refined − coarse | formed-contact F1 | 0.00616 [0.00262, 0.00974] | +4.1% | yes |
| *H1b's own effect: refined vs the frozen arm* | | | | |
| refined − identity | peptide C-N bond RMSE (A) | -0.15480 [-0.19569, -0.11801] | -196.4% | yes |
| refined − identity | backbone angle MAE (deg) | -7.06332 [-9.55851, -4.84624] | -146.2% | yes |
| refined − identity | consecutive Ca distance MAE (A) | -0.15888 [-0.21584, -0.10839] | -215.3% | yes |
| refined − identity | Ca clash rate | -0.00326 [-0.00516, -0.00162] | -138880.4% | yes |
| refined − identity | backbone torsion MAE (deg) | -3.39272 [-4.93865, -2.02779] | -15.2% | yes |
| refined − identity | Ca RMSD (A) | 0.33303 [0.26568, 0.40992] | +7.2% | yes |
| refined − identity | dRMSD, |i-j| >= 6 (A) | 0.19077 [0.15811, 0.22621] | +5.3% | yes |
| refined − identity | frame rotation error (deg) | 2.13776 [2.01069, 2.27443] | +4.5% | yes |
| refined − identity | contact F1 | 0.00602 [0.00326, 0.00884] | +1.0% | yes |
| refined − identity | formed-contact F1 | 0.15770 [0.14156, 0.17307] | +nan% | yes |
| *against the bar Stage M set: refined vs nothing moves* | | | | |
| coarse − identity | peptide C-N bond RMSE (A) | -0.36263 [-0.42362, -0.30696] | -460.1% | yes |
| coarse − identity | backbone angle MAE (deg) | -19.34168 [-24.20824, -14.88204] | -400.2% | yes |
| coarse − identity | consecutive Ca distance MAE (A) | -0.47450 [-0.58712, -0.37127] | -643.1% | yes |
| coarse − identity | Ca clash rate | -0.00479 [-0.00744, -0.00248] | -203936.7% | yes |
| coarse − identity | backbone torsion MAE (deg) | -4.77685 [-6.65079, -3.11099] | -21.5% | yes |
| coarse − identity | Ca RMSD (A) | 0.37367 [0.30056, 0.45703] | +8.0% | yes |
| coarse − identity | dRMSD, |i-j| >= 6 (A) | 0.19672 [0.16336, 0.23256] | +5.5% | yes |
| coarse − identity | frame rotation error (deg) | 2.10332 [1.97759, 2.24532] | +4.4% | yes |
| coarse − identity | contact F1 | 0.00177 [-0.00146, 0.00494] | +0.3% | no |
| coarse − identity | formed-contact F1 | 0.15154 [0.13492, 0.16706] | +nan% | yes |
| *the Stage M gap this stage set out to close* | | | | |

## 5. How much did the refiner move, and do the caps bind?

| quantity | mean [95% CI] |
|---|---|
| translation, mean over residues (Å) | 0.3753 [0.3236, 0.4306] |
| translation, per-structure max (Å) | 0.8739 [0.8418, 0.9046] |
| rotation, mean over residues (°) | 4.2964 [3.7786, 4.8710] |
| rotation, per-structure max (°) | 12.2291 [11.7858, 12.6785] |
| residues at the translation cap | 0.1462 [0.1021, 0.1955] |
| residues at the rotation cap | 0.0629 [0.0407, 0.0882] |

At most 14.6% of residues sit at a cap, so the bounds are protective rather than binding: the refiner is choosing these magnitudes, not being clipped to them.

## 6. What H1b did

**Lag 1 ns.**

- Validity improved on 5/5 cells: peptide C-N bond RMSE (A) (+53.5%, significant), backbone angle MAE (deg) (+57.1%, significant), consecutive Ca distance MAE (A) (+65.8%, significant), Ca clash rate (+52.4%, significant), backbone torsion MAE (deg) (+2.9%, significant)
- **Paid for in:** Ca RMSD (A) (-1.0%), dRMSD, |i-j| >= 6 (A) (-0.5%)
- Cells now at or better than the identity baseline: **0/5** — the bar Stage M set is still not cleared on any cell

**Lag 4 ns.**

- Validity improved on 5/5 cells: peptide C-N bond RMSE (A) (+47.1%, significant), backbone angle MAE (deg) (+50.8%, significant), consecutive Ca distance MAE (A) (+57.6%, significant), Ca clash rate (+31.9%, significant), backbone torsion MAE (deg) (+5.1%, significant)
- **Paid for in:** Ca RMSD (A) (-0.9%), dRMSD, |i-j| >= 6 (A) (-0.2%)
- Cells now at or better than the identity baseline: **0/5** — the bar Stage M set is still not cleared on any cell

