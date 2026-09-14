# Phase 1.6 H1b full report

## Decision scope

This is the registered first full H1b decision report. The hard stop is after held-out evaluation. H1a, H2, ESM3 screening, and Phase 2 were not started.

The full H1b training artifact is `steps=6000` with `is_smoke=False`. It was reused after an independent provenance audit because its registered full-run manifest and frozen-P0 hash are intact; no checkpoint was overwritten.

Primary aggregation is the sample-equal mean. The 95% intervals resample domains as clusters, with `iterations=10,000` and fixed seed `20260827`. Paired effects are computed on shared `(pair_id, domain_id)` rows and are always reported as candidate minus P0.

## Provenance and exact configuration

- H0 settings: `configs/phase1_6_h0.yaml` (evaluation manifest: `198d369cb38d5dcfd9f1eea1b34f0170f0000bf10f8900b05c93ac147184facb`).
- H1b training config: `/data1/miplab/wjyang/MDsurrogate/configs/phase1_6_h1b_full.yaml`; SHA256 `1775865559ee25de661fc1a10922b4ac7dc7a2f059d63588d5ae75cab120b9db`.
- Canonical coarse arm: `P0_pair_geometry_control`; implementation `pair_geometry`.
- P0 checkpoint SHA256 before/after: `060ad6adf53f092d6c8689a9ec706e5a8966082f0ccc239979f9893de1873d4d` / `060ad6adf53f092d6c8689a9ec706e5a8966082f0ccc239979f9893de1873d4d`; unchanged=`True`.
- Refiner checkpoint: `/data1/miplab/wjyang/MDsurrogate/runs/phase1_6_h1b_full_seed0/refiner_best.pt`; SHA256 `64221734ff8701951c06f2a4859c07140f655f712e804b241eabf656cf22fe96`; step `6000`.
- Train manifest hash: `24a1abad503af9d98fdb96ba217a687feb7721ccfcafa0b83e721555d78ee495`; held-out manifest hash: `88a0d8a31d2f60abcb8d0844bd6b4b1897ca2bc89d4b42232801d879f0bc8f26`.
- Validation rows: `3520` per stage, `3520` pairs, `30` domains; lags `1 ns` and `4 ns`.
- P0 evaluation: frozen `eval()` model under `no_grad()`; refiner is also `eval()` and P0 parameters have `requires_grad=False`.
- Alignment: the existing single proper-Kabsch convention in the H0/Stage-M evaluator; no additional primary alignment was applied here.
- Heavy mapping/masks: PSF atom names and atomic numbers are verified per domain; current conformers are transported through predicted frames for P0/H1b, identity uses current atoms, and target uses future heavy atoms only for scoring. PSF 1–2 and 1–3 exclusions are applied, 1–4 pairs remain in the primary overlap total and are tallied; Bondi radii come from the audited VDW table.
- Units and thresholds: coordinates/distances in Å, frame rotation in degrees, lag in ns; heavy contact cutoff 5.0 Å with within-chain residue separation ≥2; serious Bondi depth is the preregistered `0.4 Å` threshold.
- Future coordinates are used only as loss/evaluation targets. Identity, P0, and H1b predictions use current-frame inputs; future coordinates are not deployable inputs.

## Train and validation curves

Complete machine-readable curves: `docs/phase1_6_h1b_full_report_curves.csv`. Training history has `6000` rows and validation history has `24` rows.

| step | total | primary | coarse primary | geometry | overlap | translation cap | rotation cap |
|---|---|---|---|---|---|---|---|
| 250 | 2.6400 | 2.4226 | 2.4149 | 0.2173 | 0.0009 | 0.0897 | 0.0321 |
| 500 | 2.6225 | 2.4233 | 2.4149 | 0.1990 | 0.0009 | 0.0939 | 0.0568 |
| 750 | 2.6150 | 2.4231 | 2.4149 | 0.1918 | 0.0010 | 0.0926 | 0.0515 |
| 1000 | 2.6063 | 2.4217 | 2.4149 | 0.1845 | 0.0010 | 0.0940 | 0.0407 |
| 1250 | 2.6025 | 2.4206 | 2.4149 | 0.1819 | 0.0010 | 0.0989 | 0.0370 |
| 1500 | 2.5987 | 2.4202 | 2.4149 | 0.1784 | 0.0010 | 0.0995 | 0.0403 |
| 1750 | 2.6004 | 2.4210 | 2.4149 | 0.1793 | 0.0010 | 0.0979 | 0.0434 |
| 2000 | 2.5957 | 2.4193 | 2.4149 | 0.1763 | 0.0010 | 0.0969 | 0.0387 |
| 2250 | 2.5949 | 2.4194 | 2.4149 | 0.1754 | 0.0010 | 0.0969 | 0.0390 |
| 2500 | 2.5949 | 2.4200 | 2.4149 | 0.1748 | 0.0010 | 0.0994 | 0.0404 |
| 2750 | 2.5930 | 2.4199 | 2.4149 | 0.1729 | 0.0010 | 0.0980 | 0.0479 |
| 3000 | 2.5913 | 2.4197 | 2.4149 | 0.1714 | 0.0010 | 0.1059 | 0.0437 |
| 3250 | 2.5904 | 2.4194 | 2.4149 | 0.1709 | 0.0010 | 0.0969 | 0.0376 |
| 3500 | 2.5904 | 2.4193 | 2.4149 | 0.1710 | 0.0010 | 0.0985 | 0.0394 |
| 3750 | 2.5876 | 2.4199 | 2.4149 | 0.1676 | 0.0010 | 0.1010 | 0.0494 |
| 4000 | 2.5908 | 2.4185 | 2.4149 | 0.1722 | 0.0010 | 0.0992 | 0.0450 |
| 4250 | 2.5931 | 2.4209 | 2.4149 | 0.1721 | 0.0010 | 0.0948 | 0.0466 |
| 4500 | 2.5867 | 2.4182 | 2.4149 | 0.1684 | 0.0010 | 0.0992 | 0.0404 |
| 4750 | 2.5885 | 2.4214 | 2.4149 | 0.1670 | 0.0010 | 0.1001 | 0.0427 |
| 5000 | 2.5863 | 2.4180 | 2.4149 | 0.1682 | 0.0010 | 0.0960 | 0.0422 |
| 5250 | 2.5855 | 2.4187 | 2.4149 | 0.1667 | 0.0010 | 0.0983 | 0.0504 |
| 5500 | 2.5834 | 2.4183 | 2.4149 | 0.1650 | 0.0010 | 0.0970 | 0.0479 |
| 5750 | 2.5840 | 2.4179 | 2.4149 | 0.1660 | 0.0010 | 0.0976 | 0.0409 |
| 6000 | 2.5824 | 2.4175 | 2.4149 | 0.1648 | 0.0010 | 0.1004 | 0.0402 |

## Absolute held-out metrics

The cell is `sample-equal mean [95% domain-cluster CI]`.

### Lag 1 ns

| metric | identity_current_atoms | P0 coarse | P0 + H1b |
|---|---|---|---|
| Cα RMSD (Å) | 3.4178 [2.9189, 3.9661] | 3.1913 [2.7355, 3.6898] | 3.2245 [2.7622, 3.7304] |
| long-range dRMSD (Å) | 2.6030 [2.1878, 3.0638] | 2.4581 [2.0660, 2.8953] | 2.4694 [2.0781, 2.9068] |
| all-pair dRMSD (Å) | 2.4560 [2.0886, 2.8673] | 2.3324 [1.9779, 2.7267] | 2.3369 [1.9853, 2.7291] |
| residue-frame rotation geodesic (deg) | 37.7986 [32.7604, 43.3371] | 35.8732 [30.8700, 41.3555] | 35.8219 [30.8126, 41.3100] |
| peptide C–N bond RMSE (Å) | 0.0788 [0.0781, 0.0796] | 0.3273 [0.2993, 0.3563] | 0.1522 [0.1377, 0.1677] |
| CA–C–N angle error (deg) | 4.3032 [4.2783, 4.3288] | 16.7528 [13.4449, 20.4398] | 7.2071 [6.1571, 8.4031] |
| C–N–CA angle error (deg) | 5.2941 [5.2390, 5.3487] | 18.4154 [14.8131, 22.4338] | 7.8802 [6.9123, 8.9733] |
| both peptide-angle error (deg) | 4.7986 [4.7615, 4.8361] | 17.5841 [14.1304, 21.4284] | 7.5437 [6.5374, 8.6856] |
| consecutive Cα distance error (Å) | 0.0738 [0.0735, 0.0741] | 0.3896 [0.3124, 0.4751] | 0.1333 [0.1114, 0.1583] |
| Cα static contact F1 | 0.7035 [0.6529, 0.7512] | 0.7022 [0.6475, 0.7531] | 0.7070 [0.6538, 0.7565] |
| Cα formed-contact F1 | 0.0000 [0.0000, 0.0000] | 0.1676 [0.1517, 0.1828] | 0.1692 [0.1556, 0.1821] |
| Cα broken-contact F1 | 0.0000 [0.0000, 0.0000] | 0.4415 [0.4264, 0.4565] | 0.3656 [0.3566, 0.3750] |
| heavy-atom RMSD (Å) | 4.0362 [3.5109, 4.6123] | 3.8174 [3.3389, 4.3420] | 3.8579 [3.3713, 4.3924] |
| backbone-heavy RMSD (Å) | 3.4053 [2.9122, 3.9474] | 3.1927 [2.7416, 3.6865] | 3.2119 [2.7575, 3.7099] |
| side-chain-heavy RMSD (Å) | 4.4933 [3.9485, 5.0918] | 4.2736 [3.7755, 4.8200] | 4.3307 [3.8197, 4.8894] |
| Bondi serious overlap, inter-residue total | 0.0003 [0.0002, 0.0003] | 0.0019 [0.0012, 0.0027] | 0.0010 [0.0006, 0.0013] |
| Bondi serious overlap, inter bb–bb | 0.0010 [0.0008, 0.0012] | 0.0048 [0.0032, 0.0066] | 0.0027 [0.0018, 0.0037] |
| Bondi serious overlap, inter bb–sc | 0.0000 [0.0000, 0.0000] | 0.0011 [0.0007, 0.0017] | 0.0003 [0.0002, 0.0005] |
| Bondi serious overlap, inter sc–sc | 0.0000 [0.0000, 0.0000] | 0.0006 [0.0004, 0.0008] | 0.0005 [0.0003, 0.0007] |
| Bondi serious overlap, intra bb–bb | 0.0523 [0.0448, 0.0600] | 0.0523 [0.0448, 0.0600] | 0.0523 [0.0448, 0.0600] |
| Bondi serious overlap, intra bb–sc | 0.0237 [0.0227, 0.0247] | 0.0237 [0.0227, 0.0247] | 0.0237 [0.0227, 0.0247] |
| Bondi serious overlap, intra sc–sc | 0.1445 [0.1259, 0.1615] | 0.1445 [0.1259, 0.1615] | 0.1445 [0.1259, 0.1615] |
| atom contact static precision | 0.6066 [0.5845, 0.6282] | 0.5678 [0.5313, 0.6025] | 0.5808 [0.5480, 0.6122] |
| atom contact static recall | 0.6054 [0.5828, 0.6272] | 0.6324 [0.6198, 0.6447] | 0.6266 [0.6115, 0.6414] |
| atom contact static F1 | 0.6055 [0.5830, 0.6273] | 0.5895 [0.5615, 0.6162] | 0.5971 [0.5708, 0.6223] |
| atom contact static bb–bb F1 | 0.7906 [0.7759, 0.8049] | 0.7622 [0.7357, 0.7868] | 0.7761 [0.7529, 0.7979] |
| atom contact static bb–sc F1 | 0.5217 [0.4937, 0.5485] | 0.5081 [0.4777, 0.5372] | 0.5148 [0.4853, 0.5430] |
| atom contact static sc–sc F1 | 0.4307 [0.3971, 0.4628] | 0.4186 [0.3860, 0.4500] | 0.4239 [0.3907, 0.4556] |
| atom contact formed precision | — | 0.2865 [0.2650, 0.3072] | 0.3002 [0.2804, 0.3195] |
| atom contact formed recall | 0.0000 [0.0000, 0.0000] | 0.1940 [0.1798, 0.2099] | 0.1687 [0.1589, 0.1795] |
| atom contact formed F1 | 0.0000 [0.0000, 0.0000] | 0.2121 [0.2066, 0.2176] | 0.2020 [0.1971, 0.2070] |
| atom contact broken precision | — | 0.6871 [0.6689, 0.7055] | 0.6910 [0.6758, 0.7064] |
| atom contact broken recall | 0.0000 [0.0000, 0.0000] | 0.3202 [0.3149, 0.3253] | 0.2921 [0.2880, 0.2962] |
| atom contact broken F1 | 0.0000 [0.0000, 0.0000] | 0.4335 [0.4288, 0.4385] | 0.4084 [0.4032, 0.4137] |

### Lag 4 ns

| metric | identity_current_atoms | P0 coarse | P0 + H1b |
|---|---|---|---|
| Cα RMSD (Å) | 4.6513 [3.9716, 5.3811] | 4.2777 [3.6694, 4.9251] | 4.3183 [3.7047, 4.9724] |
| long-range dRMSD (Å) | 3.6053 [3.0149, 4.2485] | 3.4085 [2.8352, 4.0349] | 3.4145 [2.8451, 4.0369] |
| all-pair dRMSD (Å) | 3.3963 [2.8656, 3.9699] | 3.2346 [2.7121, 3.8021] | 3.2325 [2.7164, 3.7948] |
| residue-frame rotation geodesic (deg) | 47.5474 [41.0144, 54.5625] | 45.4441 [38.9556, 52.4372] | 45.4097 [38.9042, 52.4133] |
| peptide C–N bond RMSE (Å) | 0.0788 [0.0781, 0.0796] | 0.4414 [0.3860, 0.5023] | 0.2336 [0.1970, 0.2745] |
| CA–C–N angle error (deg) | 4.3303 [4.3058, 4.3553] | 23.0930 [18.7927, 27.7971] | 11.6695 [9.3645, 14.2739] |
| C–N–CA angle error (deg) | 5.3352 [5.2756, 5.3947] | 25.2559 [20.6135, 30.3061] | 12.1226 [9.9655, 14.5618] |
| both peptide-angle error (deg) | 4.8327 [4.7922, 4.8729] | 24.1744 [19.7068, 29.0502] | 11.8961 [9.6638, 14.4167] |
| consecutive Cα distance error (Å) | 0.0738 [0.0736, 0.0740] | 0.5483 [0.4451, 0.6609] | 0.2327 [0.1822, 0.2898] |
| Cα static contact F1 | 0.6238 [0.5574, 0.6868] | 0.6255 [0.5579, 0.6898] | 0.6298 [0.5631, 0.6931] |
| Cα formed-contact F1 | 0.0000 [0.0000, 0.0000] | 0.1514 [0.1348, 0.1670] | 0.1576 [0.1414, 0.1730] |
| Cα broken-contact F1 | 0.0000 [0.0000, 0.0000] | 0.4899 [0.4675, 0.5142] | 0.4153 [0.3969, 0.4358] |
| heavy-atom RMSD (Å) | 5.3274 [4.6155, 6.0860] | 4.9589 [4.3222, 5.6366] | 5.0099 [4.3646, 5.6977] |
| backbone-heavy RMSD (Å) | 4.6315 [3.9583, 5.3546] | 4.2801 [3.6742, 4.9243] | 4.3014 [3.6936, 4.9495] |
| side-chain-heavy RMSD (Å) | 5.8246 [5.0930, 6.6113] | 5.4510 [4.7974, 6.1553] | 5.5250 [4.8581, 6.2426] |
| Bondi serious overlap, inter-residue total | 0.0003 [0.0002, 0.0003] | 0.0036 [0.0022, 0.0051] | 0.0022 [0.0013, 0.0032] |
| Bondi serious overlap, inter bb–bb | 0.0010 [0.0008, 0.0012] | 0.0075 [0.0049, 0.0105] | 0.0051 [0.0032, 0.0073] |
| Bondi serious overlap, inter bb–sc | 0.0000 [0.0000, 0.0000] | 0.0027 [0.0016, 0.0039] | 0.0013 [0.0007, 0.0020] |
| Bondi serious overlap, inter sc–sc | 0.0000 [0.0000, 0.0000] | 0.0016 [0.0009, 0.0023] | 0.0012 [0.0007, 0.0017] |
| Bondi serious overlap, intra bb–bb | 0.0522 [0.0444, 0.0604] | 0.0522 [0.0444, 0.0604] | 0.0522 [0.0444, 0.0604] |
| Bondi serious overlap, intra bb–sc | 0.0237 [0.0228, 0.0246] | 0.0237 [0.0228, 0.0246] | 0.0237 [0.0228, 0.0246] |
| Bondi serious overlap, intra sc–sc | 0.1447 [0.1262, 0.1615] | 0.1447 [0.1262, 0.1615] | 0.1447 [0.1262, 0.1615] |
| atom contact static precision | 0.5510 [0.5233, 0.5782] | 0.4912 [0.4471, 0.5334] | 0.5051 [0.4632, 0.5451] |
| atom contact static recall | 0.5520 [0.5242, 0.5791] | 0.5975 [0.5825, 0.6124] | 0.5894 [0.5723, 0.6063] |
| atom contact static F1 | 0.5508 [0.5228, 0.5782] | 0.5224 [0.4857, 0.5572] | 0.5308 [0.4956, 0.5644] |
| atom contact static bb–bb F1 | 0.7546 [0.7338, 0.7745] | 0.7021 [0.6644, 0.7377] | 0.7167 [0.6809, 0.7499] |
| atom contact static bb–sc F1 | 0.4567 [0.4227, 0.4895] | 0.4369 [0.3999, 0.4726] | 0.4442 [0.4079, 0.4793] |
| atom contact static sc–sc F1 | 0.3620 [0.3247, 0.3981] | 0.3493 [0.3132, 0.3841] | 0.3545 [0.3176, 0.3902] |
| atom contact formed precision | — | 0.2393 [0.2156, 0.2625] | 0.2513 [0.2278, 0.2742] |
| atom contact formed recall | 0.0000 [0.0000, 0.0000] | 0.2044 [0.1875, 0.2228] | 0.1780 [0.1646, 0.1925] |
| atom contact formed F1 | 0.0000 [0.0000, 0.0000] | 0.1911 [0.1823, 0.1997] | 0.1826 [0.1747, 0.1902] |
| atom contact broken precision | — | 0.7280 [0.7062, 0.7507] | 0.7308 [0.7119, 0.7503] |
| atom contact broken recall | 0.0000 [0.0000, 0.0000] | 0.3511 [0.3450, 0.3577] | 0.3261 [0.3181, 0.3347] |
| atom contact broken F1 | 0.0000 [0.0000, 0.0000] | 0.4703 [0.4624, 0.4791] | 0.4480 [0.4378, 0.4589] |

## Paired P0 + H1b effects

Raw effect is `P0+H1b − P0`; for lower-is-better metrics a positive value is worse, while for higher-is-better contact metrics a negative value is worse.

### Lag 1 ns

| metric | Δ (P0+H1b − P0) | 95% domain-cluster CI | pairs | domains | verdict |
|---|---|---|---|---|---|
| Cα RMSD (Å) | 0.0333 | [0.0266, 0.0407] | 1760 | 30 | REGRESSION |
| long-range dRMSD (Å) | 0.0113 | [0.0078, 0.0145] | 1760 | 30 | REGRESSION |
| all-pair dRMSD (Å) | 0.0045 | [-0.0007, 0.0091] | 1760 | 30 | not separated |
| residue-frame rotation geodesic (deg) | -0.0512 | [-0.0693, -0.0325] | 1760 | 30 | improves |
| peptide C–N bond RMSE (Å) | -0.1751 | [-0.1900, -0.1603] | 1760 | 30 | improves |
| CA–C–N angle error (deg) | -9.5457 | [-12.0726, -7.2660] | 1760 | 30 | improves |
| C–N–CA angle error (deg) | -10.5351 | [-13.4831, -7.8826] | 1760 | 30 | improves |
| both peptide-angle error (deg) | -10.0404 | [-12.7727, -7.5763] | 1760 | 30 | improves |
| consecutive Cα distance error (Å) | -0.2563 | [-0.3174, -0.2002] | 1760 | 30 | improves |
| Cα static contact F1 | 0.0048 | [0.0030, 0.0072] | 1758 | 30 | improves |
| Cα formed-contact F1 | 0.0013 | [-0.0027, 0.0052] | 1753 | 30 | not separated |
| Cα broken-contact F1 | -0.0762 | [-0.0851, -0.0677] | 1746 | 30 | REGRESSION |
| heavy-atom RMSD (Å) | 0.0406 | [0.0316, 0.0506] | 1760 | 30 | REGRESSION |
| backbone-heavy RMSD (Å) | 0.0192 | [0.0149, 0.0242] | 1760 | 30 | REGRESSION |
| side-chain-heavy RMSD (Å) | 0.0571 | [0.0439, 0.0717] | 1760 | 30 | REGRESSION |
| Bondi serious overlap, inter-residue total | -0.0009 | [-0.0013, -0.0006] | 1760 | 30 | improves |
| Bondi serious overlap, inter bb–bb | -0.0021 | [-0.0029, -0.0014] | 1760 | 30 | improves |
| Bondi serious overlap, inter bb–sc | -0.0008 | [-0.0012, -0.0005] | 1760 | 30 | improves |
| Bondi serious overlap, inter sc–sc | -0.0001 | [-0.0002, -0.0001] | 1760 | 30 | improves |
| Bondi serious overlap, intra bb–bb | 0.0000 | [0.0000, 0.0000] | 1760 | 30 | not separated |
| Bondi serious overlap, intra bb–sc | 0.0000 | [0.0000, 0.0000] | 1760 | 30 | not separated |
| Bondi serious overlap, intra sc–sc | 0.0000 | [0.0000, 0.0000] | 1760 | 30 | not separated |
| atom contact static precision | 0.0131 | [0.0095, 0.0168] | 1760 | 30 | improves |
| atom contact static recall | -0.0058 | [-0.0100, -0.0018] | 1760 | 30 | REGRESSION |
| atom contact static F1 | 0.0076 | [0.0060, 0.0095] | 1760 | 30 | improves |
| atom contact static bb–bb F1 | 0.0139 | [0.0109, 0.0172] | 1760 | 30 | improves |
| atom contact static bb–sc F1 | 0.0067 | [0.0056, 0.0079] | 1760 | 30 | improves |
| atom contact static sc–sc F1 | 0.0052 | [0.0045, 0.0060] | 1760 | 30 | improves |
| atom contact formed precision | 0.0137 | [0.0116, 0.0157] | 1760 | 30 | improves |
| atom contact formed recall | -0.0253 | [-0.0306, -0.0203] | 1760 | 30 | REGRESSION |
| atom contact formed F1 | -0.0101 | [-0.0114, -0.0088] | 1760 | 30 | REGRESSION |
| atom contact broken precision | 0.0039 | [0.0006, 0.0071] | 1760 | 30 | improves |
| atom contact broken recall | -0.0281 | [-0.0320, -0.0240] | 1760 | 30 | REGRESSION |
| atom contact broken F1 | -0.0251 | [-0.0283, -0.0216] | 1760 | 30 | REGRESSION |

### Lag 4 ns

| metric | Δ (P0+H1b − P0) | 95% domain-cluster CI | pairs | domains | verdict |
|---|---|---|---|---|---|
| Cα RMSD (Å) | 0.0406 | [0.0343, 0.0478] | 1760 | 30 | REGRESSION |
| long-range dRMSD (Å) | 0.0060 | [-0.0003, 0.0117] | 1760 | 30 | not separated |
| all-pair dRMSD (Å) | -0.0021 | [-0.0102, 0.0052] | 1760 | 30 | not separated |
| residue-frame rotation geodesic (deg) | -0.0344 | [-0.0598, -0.0091] | 1760 | 30 | improves |
| peptide C–N bond RMSE (Å) | -0.2078 | [-0.2294, -0.1876] | 1760 | 30 | improves |
| CA–C–N angle error (deg) | -11.4234 | [-13.6161, -9.3488] | 1760 | 30 | improves |
| C–N–CA angle error (deg) | -13.1333 | [-15.8243, -10.6060] | 1760 | 30 | improves |
| both peptide-angle error (deg) | -12.2784 | [-14.7340, -9.9745] | 1760 | 30 | improves |
| consecutive Cα distance error (Å) | -0.3156 | [-0.3730, -0.2612] | 1760 | 30 | improves |
| Cα static contact F1 | 0.0043 | [0.0031, 0.0055] | 1759 | 30 | improves |
| Cα formed-contact F1 | 0.0062 | [0.0026, 0.0097] | 1759 | 30 | improves |
| Cα broken-contact F1 | -0.0746 | [-0.0817, -0.0678] | 1750 | 30 | REGRESSION |
| heavy-atom RMSD (Å) | 0.0511 | [0.0425, 0.0606] | 1760 | 30 | REGRESSION |
| backbone-heavy RMSD (Å) | 0.0214 | [0.0171, 0.0262] | 1760 | 30 | REGRESSION |
| side-chain-heavy RMSD (Å) | 0.0740 | [0.0607, 0.0886] | 1760 | 30 | REGRESSION |
| Bondi serious overlap, inter-residue total | -0.0014 | [-0.0019, -0.0009] | 1760 | 30 | improves |
| Bondi serious overlap, inter bb–bb | -0.0024 | [-0.0032, -0.0017] | 1760 | 30 | improves |
| Bondi serious overlap, inter bb–sc | -0.0014 | [-0.0020, -0.0009] | 1760 | 30 | improves |
| Bondi serious overlap, inter sc–sc | -0.0004 | [-0.0006, -0.0002] | 1760 | 30 | improves |
| Bondi serious overlap, intra bb–bb | 0.0000 | [0.0000, 0.0000] | 1760 | 30 | not separated |
| Bondi serious overlap, intra bb–sc | -0.0000 | [-0.0000, 0.0000] | 1760 | 30 | not separated |
| Bondi serious overlap, intra sc–sc | 0.0000 | [0.0000, 0.0000] | 1760 | 30 | not separated |
| atom contact static precision | 0.0139 | [0.0115, 0.0164] | 1760 | 30 | improves |
| atom contact static recall | -0.0082 | [-0.0123, -0.0042] | 1760 | 30 | REGRESSION |
| atom contact static F1 | 0.0084 | [0.0070, 0.0099] | 1760 | 30 | improves |
| atom contact static bb–bb F1 | 0.0146 | [0.0123, 0.0169] | 1760 | 30 | improves |
| atom contact static bb–sc F1 | 0.0073 | [0.0063, 0.0083] | 1760 | 30 | improves |
| atom contact static sc–sc F1 | 0.0052 | [0.0041, 0.0063] | 1760 | 30 | improves |
| atom contact formed precision | 0.0121 | [0.0108, 0.0133] | 1760 | 30 | improves |
| atom contact formed recall | -0.0264 | [-0.0305, -0.0225] | 1760 | 30 | REGRESSION |
| atom contact formed F1 | -0.0086 | [-0.0099, -0.0072] | 1760 | 30 | REGRESSION |
| atom contact broken precision | 0.0028 | [-0.0006, 0.0061] | 1760 | 30 | not separated |
| atom contact broken recall | -0.0251 | [-0.0298, -0.0200] | 1760 | 30 | REGRESSION |
| atom contact broken F1 | -0.0223 | [-0.0263, -0.0180] | 1760 | 30 | REGRESSION |

## Heavy-atom PSF metrics and domain breakdown

Heavy-atom RMSDs, Bondi serious overlaps, and static/formed/broken atom contacts use the H0 PSF atom mapping, masks, Bondi radii, units, and thresholds. The complete per-domain/per-lag/per-stage table is [`phase1_6_h1b_full_report_domain_metrics.csv`](phase1_6_h1b_full_report_domain_metrics.csv).

The Bondi table separates inter-residue bb–bb, bb–sc, sc–sc and the corresponding intra-residue classes. Contact precision, recall, and F1 are kept separate for static, formed, and broken events.

## Cap policy and saturation

Caps were preregistered and unchanged: translation norm ≤ `1 Å`; rotation residual geodesic norm ≤ `15°`. The implementation applies a radial projection to vector magnitude, not component-wise clipping; this is recorded in the evaluation manifest. Bound check passed=`True` (clipped maxima `1.0000 Å`, `15.0000°`).

| scope | n | raw translation mean / median / p90 / p95 / p99 / max (Å) | clipped translation mean / max (Å) | raw rotation mean / median / p90 / p95 / p99 / max (deg) | clipped rotation mean / max (deg) | translation saturation | rotation saturation |
|---|---|---|---|---|---|---|---|
| all validation residues | 401480 | 0.4148/0.1982/1.0613/1.6421/2.8696/11.6130 | 0.3283/1.0000 | 4.3363/2.5802/8.6730/13.8201/31.9564/180.0063 | 3.8240/15.0000 | 0.1077 | 0.0440 |
| lag 1 ns | 200740 | 0.3325/0.1832/0.8040/1.2235/2.0338/6.8459 | 0.2931/1.0000 | 3.4939/2.4210/6.8770/9.9061/19.1671/113.4116 | 3.3536/15.0000 | 0.0729 | 0.0192 |
| lag 4 ns | 200740 | 0.4970/0.2168/1.3690/2.0510/3.3189/11.6130 | 0.3635/1.0000 | 5.1787/2.7662/11.1171/18.5872/40.3203/180.0063 | 4.2943/15.0000 | 0.1425 | 0.0688 |

Complete saturation and norm breakdown by lag and domain: [`phase1_6_h1b_full_report_cap_by_domain.csv`](phase1_6_h1b_full_report_cap_by_domain.csv).

Raw statistics are computed over residue-level cap records; clipped statistics are computed from the composed correction. Saturation is `raw vector norm >= cap*(1−1e−6)`.

## Geometry-versus-transition trade-off

The paired tables above are the decision record: peptide C–N/angle/Cα-neighbor errors and Bondi bb–bb overlap are geometry outcomes, while Cα RMSD, long-range dRMSD, and frame rotation are transition outcomes. No metric was reweighted or hidden after validation.

Metric-space identity-reversion diagnostic: fraction of shared validation pairs for which the refined scalar metric is closer to the identity scalar than the coarse scalar. This is a diagnostic, not a coordinate-level claim.

| lag | metric | fraction closer to identity | pairs |
|---|---|---|---|
| 1 ns | Cα RMSD (Å) | 0.8205 | 1760 |
| 1 ns | long-range dRMSD (Å) | 0.7778 | 1760 |
| 1 ns | all-pair dRMSD (Å) | 0.7489 | 1760 |
| 1 ns | residue-frame rotation geodesic (deg) | 0.3864 | 1760 |
| 4 ns | Cα RMSD (Å) | 0.8534 | 1760 |
| 4 ns | long-range dRMSD (Å) | 0.7710 | 1760 |
| 4 ns | all-pair dRMSD (Å) | 0.7386 | 1760 |
| 4 ns | residue-frame rotation geodesic (deg) | 0.3801 | 1760 |

## Required flags

Statistically separated primary regressions:

- Lag 1 ns — Cα RMSD (Å): Δ `+0.033255`, 95% CI `[+0.026607, +0.040668]`.
- Lag 1 ns — long-range dRMSD (Å): Δ `+0.011335`, 95% CI `[+0.007779, +0.014538]`.
- Lag 4 ns — Cα RMSD (Å): Δ `+0.040638`, 95% CI `[+0.034254, +0.047832]`.

Explicit geometry flags:

- Lag 1 ns — peptide C–N bond: `improves` (Δ `-0.175143`, 95% CI `[-0.189960, -0.160311]`).
- Lag 1 ns — CA–C–N angle: `improves` (Δ `-9.545667`, 95% CI `[-12.072628, -7.266030]`).
- Lag 1 ns — C–N–CA angle: `improves` (Δ `-10.535150`, 95% CI `[-13.483091, -7.882626]`).
- Lag 1 ns — consecutive Cα distance: `improves` (Δ `-0.256344`, 95% CI `[-0.317415, -0.200227]`).
- Lag 1 ns — inter-residue bb–bb Bondi overlap: `improves` (Δ `-0.002134`, 95% CI `[-0.002924, -0.001423]`).
- Lag 4 ns — peptide C–N bond: `improves` (Δ `-0.207821`, 95% CI `[-0.229426, -0.187552]`).
- Lag 4 ns — CA–C–N angle: `improves` (Δ `-11.423441`, 95% CI `[-13.616117, -9.348786]`).
- Lag 4 ns — C–N–CA angle: `improves` (Δ `-13.133275`, 95% CI `[-15.824277, -10.605973]`).
- Lag 4 ns — consecutive Cα distance: `improves` (Δ `-0.315619`, 95% CI `[-0.372994, -0.261180]`).
- Lag 4 ns — inter-residue bb–bb Bondi overlap: `improves` (Δ `-0.002431`, 95% CI `[-0.003217, -0.001721]`).

Explicit cap-saturation flag:

- Translation saturation is recurring/systematic across lag and domain: overall `0.1077`, with nonzero saturation in `30/30` domains at 1 ns and `30/30` at 4 ns.
- Rotation saturation is lower but also recurring/systematic: overall `0.0440`, with nonzero saturation in `30/30` domains at 1 ns and `30/30` at 4 ns.

Peptide geometry and bb–bb clashes must be read directly from their rows above; the report does not substitute a composite score.
The explicit flags above preserve the geometry gains and transition costs without reweighting them.

## Stop condition

Held-out evaluation and this report are complete. No H1a, H2, ESM3 screening, or Phase 2 action was taken after this point.
