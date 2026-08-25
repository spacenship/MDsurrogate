# Phase 1: hierarchy, contracts and conventions

Reference for the hierarchical local-physics model. Everything here is asserted
by tests; where a number came from measuring real mdCATH shards rather than from
a default, it is marked **audited**.

Run `python examples/hierarchy_sanity.py` to print all of this for a live batch.

---

## 1. The state

```
q_t = { B_i , R_i , A_ia }
```

Three node types, two of which are per-residue but stay distinct because they
carry different physics and are ablated separately.

| node | one per | carries |
|---|---|---|
| `B_i` backbone frame | residue | geometry, the rigid frame, and (Phase 2) temporal state |
| `R_i` residue semantic | residue | residue identity, frozen PLM embedding, chain/mask |
| `A_ia` atom child | atom | element, atom name, local coordinates, force label |

### Representation: flattened ragged

Nodes are concatenated along one axis with an explicit `batch_index`; there is no
padded-dense code path and the two layouts are never mixed.

1. mdCATH domains span 50–479 residues (**audited**, 1000 downloaded domains),
   so padding would waste most of the tensor.
2. `e3nn` consumes `[N, irreps_dim]` 2-D input; a dense layout is flattened at
   every block anyway.
3. A padded row silently participates in equivariant pooling and radial cutoffs
   unless every op remembers the mask. A row that does not exist cannot be
   forgotten.

---

## 2. Tensor contracts

`N_a` atoms, `N_r` residues, `B` graphs.

### `ProteinAtomBatch`

| field | shape | dtype | notes |
|---|---|---|---|
| `positions` | `[N_a, 3]` | float | global frame, Å |
| `atom_to_residue` | `[N_a]` | int64 | into `[0, N_r)`, **non-decreasing** |
| `batch_index` | `[N_a]` | int64 | into `[0, B)`, non-decreasing |
| `atomic_number` | `[N_a]` | int64 | `z` as stored |
| `atom_name_id` | `[N_a]` | int64 | into `ATOM_NAMES` |
| `is_backbone` | `[N_a]` | bool | N/CA/C/O; **excludes** caps |
| `is_cap` | `[N_a]` | bool | CHARMM terminal patch atoms |
| `forces` | `[N_a, 3]` | float | optional label, kcal/mol/Å |
| `force_valid` | `[N_a]` | bool | optional; False = do not train on |
| `is_heavy` | `[N_a]` | bool | derived, `z != 1` |

### `ResidueSemanticBatch`

| field | shape | dtype | notes |
|---|---|---|---|
| `residue_type` | `[N_r]` | int64 | into `RESIDUE_TYPES` (21 incl. UNK) |
| `plm_embedding` | `[N_r, 1280]` | float | frozen ESM-2 650M |
| `resid_original` | `[N_r]` | int64 | source numbering, **not** a 0-based index |
| `chain_index` | `[N_r]` | int64 | 0-based within its graph |
| `batch_index` | `[N_r]` | int64 | non-decreasing |
| `mask` | `[N_r]` | bool | False = excluded from losses |

### `BackboneFrameBatch`

| field | shape | dtype |
|---|---|---|
| `n_positions` / `ca_positions` / `c_positions` | `[N_r, 3]` | float |
| `residue_to_backbone` | `[N_r]` | int64, a permutation |
| `frame_valid` | `[N_r]` | bool |
| `batch_index` | `[N_r]` | int64 |

### `HierarchicalProteinBatch`

Holds the three levels plus `units`, `temperature [B]` (kelvin),
`domain_id` (length `B`), `replica_index [B]`, `frame_index [B]`.

> `frame_index` counts **frames, not nanoseconds**. mdCATH stores no per-frame
> timestamp and no value is invented; see `docs/open_questions.md`.

`validate()` is deliberately separate from `__init__` — construction is on the
hot path. It checks dtypes, index ranges, monotonicity, 1:1 backbone mapping,
that every residue owns at least one atom, and that an atom's graph equals its
parent residue's graph (batch-leak check).

---

## 3. Frames and coordinate conventions

```
r_i = CA_i
e1  = normalize(C_i - CA_i)
e2  = normalize((N_i - CA_i) - <N_i - CA_i, e1> e1)
e3  = e1 x e2
R_i = [e1 | e2 | e3]          # COLUMNS are the local axes in global coordinates
y_ia = R_i^T (x_ia - r_i)     # global point -> local
x_ia = R_i y_ia + r_i         # local -> global
```

`R_i` maps *local* vectors to *global* ones; `R_i^T` does the reverse. Verified:
`|R^T R - I| ~ 4e-16`, `det(R) = 1.000000000000`.

**Where each quantity lives.**

| quantity | frame | behaviour under proper rotation `Q` |
|---|---|---|
| encoder features (all levels) | global | `Y -> Y D(Q)^T` |
| `l=0` block of any feature | — | invariant |
| predicted forces, torques | global | `f -> Q f` |
| predicted log-variances | **residue-local** | invariant |
| `y_ia`, local relative edge vectors | residue-local | invariant |
| energy `U` | — | invariant |

A diagonal covariance is only meaningful in a fixed frame, which is why
uncertainty is expressed locally and the NLL rotates the error into that frame
before evaluating.

**Degenerate frames** (missing or collinear N/CA/C) are reported through
`valid` and replaced by the identity. Denominators are clamped *before* division,
not patched afterwards with `torch.where`: a NaN produced in an unselected branch
still poisons the backward pass.

---

## 4. Chirality

The requirement is **SE(3)** — proper rotations and translations. Reflection is
not a symmetry of a chiral molecule and is never tested as one.

`e3` is a cross product, so a mirrored structure does **not** produce the
mirrored frame. Chirality reaches the network through the residue-local
coordinates `y_ia` in `AtomEmbedding`: an encoder built only from spherical
harmonics of relative positions would be accidentally E(3)-equivariant and unable
to tell an L-protein from its D mirror image.

Measured: mirroring changes the `l=0` block of `physics_latent` by **1.90**,
while a proper rotation leaves it unchanged to `7e-15`.

---

## 5. Topology — six relations

| relation | src → dst | sub-types |
|---|---|---|
| `backbone__sequence__backbone` | B → B | offset −2,−1,+1,+2 (chain-break aware) |
| `backbone__spatial__backbone` | B → B | kNN(16) on CA |
| `backbone__owns__residue` | B → R | 1:1 |
| `residue__contains__atom` | R → A | exactly one parent per atom |
| `atom__bonded__atom` | A → A | intra / inter-residue |
| `atom__spatial__atom` | A → A | intra / inter-residue |

Relations are **never merged or de-duplicated**. A residue pair that is both a
sequence and a spatial neighbour appears in both. `merge_edge_sets` combines
relations into one scatter while offsetting sub-types, so relation identity
survives.

**No edge crosses a graph boundary.** Enforced by construction and checked by
`HierarchicalGraph.validate`.

The neighbour list is a **discrete** function of the coordinates. Gradients flow
through edge vectors and distances, never through the topology. Rebuild the graph
when coordinates move; do not claim the neighbour list is differentiable.

### Audited defaults

| parameter | value | evidence (6 real domains) |
|---|---|---|
| `atom_cutoff` | **5.0 Å** | 21.4 heavy neighbours/atom (p95 32); 4.5 Å gives 15.2, 6.0 Å gives 35.4 |
| `residue_knn` | **16** | mean CA radius 10.2 Å, p95 13.0 Å |
| `backbone_cutoff` | **13.0 Å** | covers that p95 so the envelope does not clip real neighbours |
| `bond_tolerance` | **1.3** | reproduces the PSF bond graph **exactly**: 0 missing, 0 extra on 4 domains |

---

## 6. Encoder

One cycle, run twice in the Phase 1 small config:

```
atom interaction        local chemistry, 3-body, within r_cut
atom -> residue pool    equivariant, size-normalised (mean)
residue -> backbone     pooled irreps + PLM/temperature scalars
backbone interaction    sequence ±1/±2 and CA kNN
backbone -> residue     gated global context back down
residue -> atom         gated broadcast, not a raw copy
```

### Irreps table

| tensor | irreps | dim |
|---|---|---|
| node features (all three levels) | `64x0e + 16x1o + 8x2e` | 152 |
| edge spherical harmonics | `1x0e + 1x1o + 1x2e` | 9 |
| residue conditioning scalars | `64x0e` | 64 |
| tensor-product output (uvu, pre-Linear) | sorted `0e/1o/2e` | 880 |
| `physics_latent` | `64x0e + 16x1o + 8x2e` | 152 |

### Tensor product: `uvu`, not fully connected

A fully-connected (`uvw`) product needs one weight per
(in-channel, sh-channel, out-channel) triple: **8064 weights per edge**. At 21
neighbours per atom that is 1.35 GB of weights alone for a two-protein batch —
which is exactly what made the first real training run exhaust an 80 GB card.

`uvu` keeps the input multiplicity and weights each path once: **288 weights per
edge**, a 28× reduction, reaching the same irreps. Peak memory for a
2×250-residue batch dropped from OOM-at-38 GB to **9.41 GB**.

Two implementation notes worth keeping:

* the `Linear` after the tensor product is applied **after the scatter**, not
  per edge — a linear map commutes with summation, and doing it on `N` nodes
  instead of `E` edges is where most of the activation memory is saved;
* do **not** call `.simplify()` on the output irreps after building the
  instruction list — it merges entries and silently invalidates the indices the
  instructions point at. Sort only.

### Body order

One message pass is a 2-body interaction. After aggregation the block applies an
explicit `o3.TensorSquare` and mixes the result back; the square of a sum over
neighbours contains pair-of-neighbour cross terms, so **one block reaches
correlation order 3** (centre plus two neighbours). This is the MACE idea
implemented directly — no MACE package, no pretrained potential.

### Radial basis

Bessel functions times a degree-6 polynomial envelope that reaches exactly zero
with zero derivative at `r_cut`. Without the envelope an atom crossing the cutoff
makes the energy jump and any force read off it is wrong near the boundary.

---

## 7. Force projection and its limits

```
F_i   = sum_{a in i} f_a
tau_i = sum_{a in i} (x_a - r_i) x f_a      about r_i = CA_i
tau(o') = tau(o) + (o - o') x F             origin-shift law
```

`ResidueSumProjector` is named for what it does. It is **not** a
"constraint-aware projection": mdCATH's forces come from a constrained simulation
(rigid X–H bonds), and summing them does not undo those constraints. A genuinely
constraint-aware projector needs the constraint Jacobian and is deliberately
unimplemented.

**What a heavy-atom force label already contains.** `f_a` is the *total* force on
atom `a`, including the pull of its hydrogens and of surrounding water. Summing
over heavy atoms therefore gives the net force on the heavy-atom subset with
solvent coupling included. What it omits is the force acting *on* the hydrogens.

**The omitted-atom residual is identifiable here.** mdCATH stores hydrogens
(**audited**: 558 of 1126 atoms in a typical domain), so both scopes can be built
and their difference measured:

```
residual_i = F_i(all_atom) - F_i(heavy_atom) = sum_{h in i} f_h
```

Across 12 audited domains its magnitude is **0.58–0.71×** the heavy-only residue
force (mean 0.63) — far too large to fold into uncertainty. The design document
assumed only a heavy-atom target would exist and told us to disable this head;
that assumption does not hold for this dataset, so it is **enabled**, and it is
refused at loss time if no target is supplied.

**Solvent is not identifiable and is never faked.** Water and ions are absent
from the file. No solvent atom is assigned to a residue; the unresolved solvent
contribution is represented by predicted *uncertainty*.

---

## 8. Outputs and losses

See `Phase1Output` for the full field list. Structurally:

* `atom_force_mean = atom_force_residual + atom_force_conservative`. The
  conservative part is `-dU/dx`; the residual carries the rest. The full
  effective force on a coarse-grained atom is **not** the gradient of a potential
  — momentum and solvent have been integrated out — and the code never equates
  them.
* `residue_force_mean = residue_explained_force + residue_hidden_force`.

```
L = 1.0  * NLL(atom force)          heteroscedastic, local frame
  + 1.0  * NLL(residue force)
  + 1.0  * NLL(residue torque)
  + 1.0  * MSE(omitted-atom residual)     only when identifiable
  + 0.1  * ||explained - aggregate||^2    aggregation consistency
  + 0.1  * ||-grad U - f_target||^2       auxiliary, not an identity
  + 1e-4 * mean(U^2)                      energy gauge
```

The **hidden residual is excluded from the aggregation-consistency term**. That
term ties the residue-level *explained* force to the sum of predicted atom
forces; the residual is by definition the part not carried by represented atoms,
so including it would ask the two terms to cancel and let the model hide
arbitrary mass in the residual.

mdCATH has **no energy label**. `U` is trained only through its gradient, which
leaves its offset and scale unconstrained — the gauge term bounds them. It is not
a fit to any measured energy. (The design document's `L_local_geometry` is
undefined; see `docs/open_questions.md`.)

Targets are normalised by RMS magnitude, fitted on the **training split only**.

---

## 8a. Cost of the energy branch

Training with `use_energy_branch=True` needs `create_graph=True` so the
conservative force can be backpropagated, and that second-order graph runs
through the whole e3nn encoder. Measured on CPU with the default width: **~14 s
per optimisation step**, against ~0.1 s with the branch off. On GPU the gap is far
smaller but still real.

Practical consequences:

* train with the energy branch on GPU, not CPU;
* if `lambda_energy = 0` **and** you do not want `-dU/dx` inside
  `atom_force_mean`, set `use_energy_branch=False` rather than only zeroing the
  weight — the branch still contributes to the predicted mean, so zeroing the
  loss weight alone does not remove the cost;
* `conservative_force_create_graph=False` is correct at inference and roughly
  halves the memory there.

The angular-error metric uses `atan2(|a x b|, a.b)` rather than `arccos(cos)`.
`arccos` is ill-conditioned exactly where a good model lives: at `cos = 1 - eps`
the angle is `~sqrt(2 eps)`, so float32 rounding alone reports ~0.03 degrees for a
perfect prediction.

---

## 9. Data handling

**Splits are by domain, applied before frames are enumerated.** Consecutive
mdCATH frames of one protein are highly correlated; a frame-level split measures
memorisation.

**Corrupt force labels.** 5 of 1000 downloaded domains ship a `forces` array
byte-identical to `coords`, always as replicas 0–3 at all five temperatures with
replica 4 intact. All 1000 files match the HuggingFace repo's recorded sizes, so
this is upstream in the published dataset. `scripts/audit_mdcath_forces.py`
writes a per-trajectory quarantine list; the adapter masks those labels through
`force_valid` rather than dropping the shards, because the coordinates remain
usable for Phase 2.

**PBC.** The protein is whole in every frame inspected (CA–CA 3.67–4.01 Å). The
adapter checks this per frame and raises rather than silently unwrapping.

**CHARMM naming.** Three facts that would each be a silent bug under PDB naming:
histidine is `HSD`/`HSE`/`HSP`; terminal caps (`CAY HY1 HY2 HY3 CY OY`,
`NT HNT CAT HT1 HT2 HT3`) are merged into the terminal residues, so selecting the
backbone by element would pick `CAY` up as the alpha carbon; isoleucine's delta
carbon is `CD`. Across 1000 domains and 136,825 residues the vocabulary produces
**zero** unknown residues.

**ESM-2.** `facebook/esm2_t33_650M_UR50D`, run once per domain by
`scripts/precompute_esm2.py` and cached (1000 domains, 673 MB). The training loop
never runs a PLM forward pass. Cache entries are keyed by a SHA-256 of
sequence + model + revision + layer, so an entry from a different checkpoint
cannot be silently reused. Nothing is truncated silently; the audited maximum is
479 residues against ESM-2's 1022 limit, so no domain is affected.

---

## 10. Verified numbers

From `examples/hierarchy_sanity.py`, float64:

```
|R^T R - I|                        4.441e-16
det(R)                             1.000000000000
physics_latent equivariance        8.171e-10  (relative)
atom_force_mean equivariance       8.373e-12
residue_torque_mean equivariance   1.744e-11
l=0 block invariance               7.105e-15
energy invariance                  2.220e-16
logvar invariance                  5.329e-15
reflection changes l=0 by          1.900       <- must be large
```
