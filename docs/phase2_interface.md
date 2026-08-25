# Phase 2 handoff

Phase 1 is not a prototype to be replaced. Phase 2 loads these modules unchanged
and adds temporal machinery on top. This document is the contract.

---

## 1. What Phase 2 may depend on

`LocalPhysicsModel.latent_contract()` returns this at runtime, so a Phase 2 run
can assert it instead of assuming it:

```python
{
  "physics_latent_irreps": "64x0e+16x1o+8x2e",
  "physics_latent_dim": 152,
  "row_order": "aligned with batch.residues (residue node index)",
  "frame": "global",
  "target_scope": "heavy_atom",
  "predicts_hidden_force": True,
  "num_cycles": 2,
  "lmax": 2,
}
```

**Frozen guarantees:**

1. `physics_latent` is `[N_res, 152]`, row `i` is residue node `i` of
   `batch.residues`, in the **global frame**, carrying irreps
   `64x0e + 16x1o + 8x2e`.
2. Under a proper rotation `Q`, `physics_latent -> physics_latent @ D(Q).T`,
   where `D` is the Wigner-D of those irreps. Verified at `8e-10` relative.
3. Every field of `Phase1Output` keeps its name, shape and frame.
4. A saved checkpoint reloads into the same class and reproduces the same
   outputs to `1e-12` (`test_checkpoint_round_trip_preserves_the_output`).

Adding fields is fine. Renaming or repurposing one breaks a Phase 2 checkpoint.

**Widening is configuration, not a new class.** `EncoderConfig.num_cycles` and
the `IrrepsConfig` multiplicities are the knobs. `lmax` stays 2. Changing widths
of course changes `physics_latent_dim`, so Phase 2 should read it from
`latent_contract()` rather than hard-coding 152.

---

## 2. Classes Phase 2 reuses unchanged

| module | role in Phase 2 |
|---|---|
| `data.contracts.HierarchicalProteinBatch` | the state `q_t`; also the state at every rollout step |
| `data.collate.collate_batches` | batching, including batching over time windows |
| `geometry.frames` | `R_i` is the object the `SO(3)` transition flows on |
| `geometry.link_backbone_to_atom_positions` | keeps one geometry tensor of record |
| `graph.build_hierarchical_graph` | rebuilt each rollout step |
| `nn.HierarchicalPhysicsEncoder` | the shared encoder, frozen then fine-tuned |
| `physics.ResidueSumProjector` | auxiliary force/torque targets |
| `physics.*Head`, `physics.losses` | auxiliary physics supervision |
| `models.LocalPhysicsModel` | loaded whole from the Phase 1 checkpoint |
| `training.metrics` | zero-baseline-relative reporting |

---

## 3. Recommended Phase 2 training order

1. Load the Phase 1 checkpoint (`Phase1Trainer.load_checkpoint`).
2. Freeze the encoder; train the transition head alone.
3. Joint fine-tune at a small learning rate.
4. **Keep the Phase 1 force/torque/energy auxiliary losses on** so the physics
   representation does not degrade while the transition objective dominates.
5. Add multi-lag transitions, then short-horizon consistency, then
   Chapman–Kolmogorov consistency — in that order, one at a time.

---

## 4. Training target vs. inference input

This distinction matters more in Phase 2 than in Phase 1, because a rollout has
no labels at all.

| | Phase 1 training | inference / Phase 2 rollout |
|---|---|---|
| positions, chemistry, sequence, temperature | input | input |
| **ground-truth forces** | **label only** | **not available, not needed** |
| predicted forces / torques / energy | outputs | conditioning for the transition |

`LocalPhysicsModel.forward` never reads `batch.atoms.forces`.
`test_forward_ignores_ground_truth_forces` runs the model with the labels
deleted and with the labels randomised and requires bitwise identical output. A
model that peeked at the label would train beautifully and be useless in rollout.

---

## 5. Rollout state — what must be carried

A rollout step must produce enough state to build the **next** encoder input.
Backbone frames alone are not enough:

* `AtomEmbedding` consumes `y_ia = R_i^T (x_ia - r_i)` for every represented
  atom, so heavy-atom local coordinates are a required input, not a decoration;
* `atom__bonded__atom` and `atom__spatial__atom` are built from atom positions;
* the force/energy heads are per-atom.

So Phase 2 must generate, per step, **both** the future backbone frames
`(R_i, r_i)` **and** the future heavy-atom local states `y_ia`. Global positions
follow as `x_ia = R_i y_ia + r_i`. Generating backbone only and discarding atom
child state produces an incomplete rollout that cannot be fed back in.

Convenience: `geometry.link_backbone_to_atom_positions` rebuilds backbone
positions as gathers of the atom tensor, which keeps one tensor of record and
makes `-dU/dx` complete. Apply it after reconstructing global coordinates.

---

## 6. Physical time — unresolved

mdCATH stores **no per-frame timestamp**. `frame_index` counts frames and no
nanosecond value is invented anywhere in this codebase.

Phase 1 does not need it. Phase 2's 1/2/4/8/16 ns multi-lag training **does**.
Before that work starts, confirm the ns-per-frame for this specific download and
thread it through as `ps_per_frame`. See `docs/open_questions.md` §2.

---

## 7. What Phase 1 deliberately does not implement

None of these exist even as placeholders, because a plausible-looking wrong
formula is worse than an absent one:

* ns-scale future-structure flow matching
* recent 2–4 frame temporal memory
* `SO(3)` rotation transition solver
* stochastic multi-lag decoder
* Chapman–Kolmogorov composition loss
* HFM-inspired mean-flow consistency
* long rollout and stationary-distribution evaluation
* constraint-aware force projection (needs the constraint Jacobian)

The `physics_latent` contract and the state contract are designed so these attach
without modifying Phase 1.

---

## 8. Scientific premises Phase 2 must preserve

Carried over from Phase 1 and asserted in code and docstrings:

* An instantaneous force does not determine coordinates 1–16 ns later. Measured:
  the frame-to-frame force correlation in mdCATH is **−0.024** — essentially
  zero. Force is auxiliary supervision about the local energy landscape, not a
  short-cut to the transition.
* The coarse residue system is **open and generally non-Markovian**: momentum and
  solvent are integrated out. Do not call it Hamiltonian dynamics.
* The protein's net force is **not zero** (measured 42.7 kcal/mol/Å on a typical
  frame) precisely because solvent acts on it.
* Pair contributions are a latent interaction decomposition, not the unique
  physical decomposition of the net MD force.
* mdCATH forces are labels from one force field and protocol
  (`top_all22star_prot`), not ground truth about nature.
* Reflection is not a symmetry. Preserve chirality; do not "improve"
  equivariance by making the model E(3)-equivariant.

---

## 9. Phase 1 status at handoff

| criterion | status |
|---|---|
| three node types, vertical + horizontal relations | done |
| `A→R→B→R→A` cycle, run twice | done |
| frozen ESM-2 pipeline + offline test path | done (1000 domains, 673 MB cache) |
| e3nn `l_max=2` production blocks, no toy head | done |
| small config differs only in depth/width | done |
| force, torque, uncertainty, energy, latent, optional residual | done |
| projection vs. hidden residual interpretation separated | done |
| SE(3) invariance/equivariance verified numerically | done, `~1e-10` |
| variable-size proteins, masks, batching | done |
| synthetic overfit | done: relative RMSE 0.94 → 0.19, angle 43° → 1.5° |
| mdCATH mini-subset train/val smoke | done, see below |
| checkpoint reload preserves the output contract | done |
| no speculative transition/HFM/CK formulas | done |

**Mini-subset result** (40 domains, 32 train / 8 val split by domain, 500 steps —
a smoke test, not a performance claim):

```
atom force   RMSE 26.39 vs 46.69 zero baseline   relative 0.566   angle 37.8 deg
residue force                                     relative 0.762   angle 48.9 deg
torque                                            relative 0.998   <- not learned
```

The atom-level force is genuinely being learned on held-out proteins. Residue
force is weak and **torque is not learned at all** at 500 steps. No mdCATH
performance is claimed.
