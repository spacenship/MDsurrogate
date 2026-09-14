# Phase 1.6 — Step 0 repository audit

Written before any code changed, against commit `9370c19` ("Phase 1 and 1.5"),
working tree clean. Everything below was read out of the source, not assumed.

Its purpose is to record **where the task brief and the repository disagree**, so
the implementation follows the code rather than the brief's description of it.

---

## 1. Where the brief is wrong about this repository

| Brief says | Repository actually has |
|---|---|
| Phase 1 outputs "pair/triplet/higher-order interaction latent" | **No.** `Phase1Output` (`src/force_md/physics/outputs.py`) has 15 fields; none is edge-level. The only handoff is `physics_latent` `[N_res, 152]`. |
| Phase 1.5 has a "full-physics arm" reusable as `S2` | Correct — `physics_latent` (arm C). |
| `P2` needs new force-moment code | **Already exists.** `force_moments()` (`transition/moments.py:148`) returns exactly `F_i`, `τ_i` and the symmetric-traceless `S_i` the brief specifies, in the residue-local frame. Reused, not rewritten. |
| `history_length=2` default | Correct (`TransitionProbeConfig.history_length = 2`). |
| Future structure is a label the model cannot see | Correct and **structurally enforced**: `LagPairExample.future` is a `FrameGeometry`, not a state, and `TransitionProbe.forward` has no parameter that could accept it. |

### 1.1 What "pair latent" can honestly mean here

Per-edge messages **do** exist, as an intermediate that is currently discarded.
`EquivariantMessageBlock.forward` (`nn/blocks.py:195`) computes

```python
messages = self.tp(node_features[edges.src], edge_sh, weights)   # [E, 880]
pooled   = scatter_sum(messages, edges.dst, n) / avg_neighbors**0.5
```

and only `pooled` survives. At Phase 1's widths the message irreps are
`88x0e + 104x1o + 96x2e` = **880 per edge** (derived from the `uvu` instruction
set in `_build_uvu_tensor_product`, verified numerically — see
`tests/test_pair_physics.py::test_pair_message_irreps_match_contract`).

Two edge levels exist. The **backbone level** is the residue-pair graph
(sequence ±1/±2 plus CA-kNN(16), `backbone_cutoff=13.0`) and is the one Phase 1.6
uses. The atom level is not a residue-pair object.

**There is no triplet or higher-order edge tensor, and none is invented.** Phase 1
reaches correlation order 3 through a node-level `o3.TensorSquare` inside each
block (`blocks.py:200`), so the 3-body information is already inside
`physics_latent` — which is exactly what arm `S2` conditions on. Phase 1.6
therefore compares *edge-level* against *node-level* physics, which is research
question 2, and says nothing about triplets it does not have.

---

## 2. Arm mapping: canonical name → existing implementation

| Canonical | Existing arm | Class | Status |
|---|---|---|---|
| `S0_structure_history` | `structure_only` | `ZeroConditioner` | reused unchanged |
| `S1_residue_wrench` | `force_torque` | `ResidueForceTorqueConditioner` | reused unchanged |
| `S2_node_physics_152d` | `physics_latent` | `PhysicsLatentConditioner` | reused unchanged |
| `O_current_gt_force_oracle` | `oracle_force` | `OracleAtomicForceConditioner` | reused unchanged |
| — | `force_pattern_shape` | `ForcePatternShapeConditioner` | legacy Phase 1.5 arm D; no canonical P name. Kept, reported as legacy. |
| `P0_pair_geometry_control` | `pair_geometry` | new | new |
| `P1_pair_physics_frozen` | `pair_physics` | new | new |
| `P2_pair_physics_moments` | `pair_physics_moments` | new | new |
| `P3_pair_physics_uncertainty` | `pair_physics_uncertainty` | new | new |
| `P4_future_physics_consistency` | `pair_physics_future` | new | new |
| `P5_pair_physics_partial_e2e` | `pair_physics_partial_e2e` | config + smoke only | not run |

---

## 3. Verified tensor sources

| Feature | Source | Shape |
|---|---|---|
| node physics latent | `Phase1Output.physics_latent` | `[N_res, 152]`, global frame |
| atom force mean | `Phase1Output.atom_force_mean` | `[N_atom, 3]`, global frame, **predicted** |
| atom force logvar | `Phase1Output.atom_force_logvar` | `[N_atom, 3]`, residue-local frame |
| residue force / torque | `residue_force_mean`, `residue_torque_mean` | `[N_res, 3]` global, torque about CA |
| pair message | *new* — last backbone block of the last cycle | `[E_res, 880]`, global frame |
| pair edge index / type / geometry | *new* — the same `EdgeSet` the block consumed | `[E_res]`, `[E_res]`, `[E_res]`, `[E_res, 3]` |
| GT atom force (oracle only) | `OracleFeatureBundle.atom_force` | `[N_atom, 3]` |

`FeatureBundle` has **no** `forces` field by construction, and
`assert_production_safe()` rejects an `OracleFeatureBundle` at every production
conditioner's entry point.

---

## 4. Data, split, pairing, metrics — reused unchanged

* Split: `restore_phase1_split()` reads `runs/phase1_full/config_snapshot.json`
  → 726 train / 181 val domains, by domain, before any frame is enumerated.
  Phase 1.6 changes nothing here and creates no new split.
* Lags: `exact_lag_frames()` asserts the lag is an integral number of frames;
  1 ns and 4 ns are 1 and 4 frames at `ps_per_frame=1000`.
* Manifest: `LagPairManifest.content_hash()`; `assert_disjoint()` between splits.
* Targets: `build_transition_target()` — Kabsch alignment, residue-frame
  translation and relative rotation. Unchanged.
* Metrics: `transition/metrics.py`, micro and domain-macro. Unchanged.
* Loss: `transition_loss()` — unchanged as the primary objective.

## 5. Feature cache

`Phase1FeatureCache` exists but is **not wired into training** — it is a
precompute utility. Its shard header pins `format_version`, `checkpoint_sha256`
and `config_hash`, and refuses a mismatched shard.

Phase 1.6 bumps `FEATURE_FORMAT_VERSION` 1 → 2 and adds `pair_features` to the
header, so a shard written without pair features cannot be read by a pair arm.
`split_bundle()` refuses to shard a bundle carrying pair features rather than
silently dropping edges across a graph boundary.

## 6. Existing results reuse

`runs/phase1_5_full_seed{0,1,2}/` hold the completed 3-seed Phase 1.5 experiment.
They were produced against `runs/phase1_full/last.pt`. As of 2026-08-26 a
reproduction run `runs/phase1_full_rerun/last.pt` also exists with a **different**
SHA-256; every Phase 1.6 config pins `runs/phase1_full/last.pt` so the existing
`S0/S1/S2/O` numbers stay comparable. Reuse is only legitimate when
`manifest_hash`, `phase1_sha256`, `seed` and `max_steps` all match — the runner
asserts this and aborts otherwise.

## 7. Blockers found: none

The one design risk was `P4`, whose target needs frozen Phase 1 applied to the
**future** structure. `FrameGeometry.positions` carries the full represented-atom
coordinates of that frame (`lag_pairs.py:767`), so the future state can be built
by substituting positions into the current batch — no dataset change, no extra
h5 read. The substituted batch is built with `forces=None`, so a future force
cannot be read even by accident.
