# Stage 3 atom physics implementation report

> Historical report. Current implementation and training commands:
> [heavy-flow v2](docs/heavy_flow_v2_architecture.md). V2 trains pair/axial projections through the atomic force loss and requires new Stage 3 training.

Status: implementation complete; scientific Stage 3 training is not claimed complete.

## Implemented boundary

The direct path is:

`sequence + current/history geometry -> C_atom -> PhysicsPredictor -> PhysicsState -> ForceHead -> atom force distribution`

`PhysicsPredictor.forward` accepts no force target, future coordinate, oracle
latent, or teacher input. `edge_scalar` is a learned net-atomic-force
interaction representation; it is not a unique pairwise force decomposition.

The initial configuration is recorded in
`configs/heavy_flow/stage3.yaml`: three physics blocks, `lmax=2`, polar vector
mean, isotropic scalar log-variance, and pair latent enabled.

## Data and loss

`HeavyFlowPhysicsFrameDataset` uses the mdCATH reader's direct represented
heavy-atom forces and preserves reader atom order. Its split manifest holds out
unseen domains in full and keeps same-domain validation in non-overlapping
trajectory time blocks. The force normalizer is a single train-split RMS and is
stored in the checkpoint.

The likelihood is masked per-atom heteroscedastic Gaussian NLL, averaged over
atoms within each protein before averaging proteins. Log-variance bounds and
float32 variance arithmetic make the loss finite under AMP. Metrics include
RMSE, correlation, R2, cosine, NLL, standardized residuals, 68/95% coverage,
variance calibration, residue net force, and torque. Torque uses the current
represented-heavy-atom residue centroid as its documented origin.

## Validation performed

* Existing repository test suite: passed with no reported failures.
* `tests/heavy_flow`: Stage 1/2/3 regression suite passed.
* Stage 3 tests cover target-free signatures, packed/dense shapes, rotation
  equivariance, atom permutation equivariance, finite all-mask gradients,
  normalizer roundtrip, and leakage-safe temporal splitting.
* One real mdCATH frame (`1a0rP01/320/0/frame 0`) passed the adapter audit:
  1,017 represented heavy atoms, 1,036 mapped bond edges, and
  `force_scope=direct_heavy_atom`.
* Synthetic bounded overfit decreased NLL from `3.633` to `2.646` over 24
  steps. This is only an executable smoke check; its force-mean RMSE did not
  improve, so the Stage 4 proper-score/generalization gate remains pending.

## Artifacts

The bounded smoke checkpoint and its companion metadata are under
`outputs/heavy_flow/stage3/`:

* `tiny_overfit_checkpoint.pt`
* `normalizer.json`
* `split_manifest.json`
* `per_domain_metrics.json`

The real-data result still requires the requested 20–30 train-domain / 8–10
validation-domain target audit, explicit train/validation force-norm
comparison, tiny overfit on the real split, and same-domain/unseen-domain
proper-score comparison before any full transition training.
