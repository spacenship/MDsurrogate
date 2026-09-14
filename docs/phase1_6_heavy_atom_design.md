# Heavy-atom extension (H0–H2) — reconnaissance and design

Written **before any code changed**, against commit `9370c194238c` with the
user's dirty working tree preserved. Everything below was read out of the source
or measured from a real mdCATH shard (`data/mdcath_dataset_1ad3A02.h5`), not
assumed.

Its purpose is to record what this repository and this dataset actually support,
so H0/H1/H2 are built on measured facts rather than on the brief's assumptions
about them.

---

## 1. What already exists and will be reused

| need | what exists | status |
|---|---|---|
| residue frames, axis convention | `geometry/frames.py` — local axes as **columns**, `R_i` maps local→global | reuse unchanged |
| Kabsch alignment | `geometry/alignment.py` — proper rotation only, `det=+1`, SVD in float64, `apply_frames` = `R → QR` | reuse unchanged |
| transition target | `transition/targets.py::build_transition_target` — one global fit of future Cα onto current Cα | reuse unchanged |
| backbone reconstruction | `transition/targets.py::reconstruct_backbone` — N/Cα/C only, carries **current** local N and C | reuse; H0 generalises it to all heavy atoms |
| existing metrics | `transition/metrics.py`, `transition/extended_metrics.py` | reuse; H0 must reproduce them exactly |
| statistics | `scripts/analyze_phase1_6_extended.py` — domain cluster bootstrap, paired delta, sign normalisation, `Svg` | reuse by import |
| arm registry | `transition/arms.py::CANONICAL_ARMS` | reuse unchanged |
| checkpoint loading | `TransitionTrainer.load_checkpoint`, `scripts/evaluate_phase1_6_extended.py` | reuse the same pattern |
| provenance | `RecordContext`, `reproducibility_manifest.json`, sha256 before/after | reuse and extend |
| equivariant blocks | `e3nn` 0.6.0 via `nn/blocks.py`, `nn/irreps.py`; `scatter_sum` = `index_add` | reuse for H2 |
| tests | `tests/`, flat layout, `conftest.py` holds the dihedral convention self-test | follow |

---

## 2. What the data actually contains — measured

Measured on `1ad3A02` (195 residues, 3,066 atoms).

| item | finding |
|---|---|
| **hydrogens** | **Present.** 1,534 of 3,066 atoms are H (50.0%). Elements: C 967, H 1534, O 290, N 265, S 10. |
| **forces** | Present on **every** atom including hydrogens, `[450, 3066, 3]` per replica. |
| **units** | `data/units.py` is authoritative: length **ångström**, force **kcal/mol/Å**, energy kcal/mol, temperature K. Not inferred. |
| **coordinate array** | Protein-only (3,066), while the PSF declares the full solvated system (41,926 atoms). So protein forces already include solvent effects implicitly. |
| **residue names** | CHARMM: `HSD` for histidine, and `CHARMM_RESNAME_ALIASES` maps it. `ILE` uses `CD`, not the PDB's `CD1`. |
| **caps** | CHARMM terminal patches (`CAY HY1 HY2 HY3 CY OY` / `NT HNT CAT HT1-3`) merged into the first/last residue; flagged by `is_cap`, belong to no standard template. |
| **`represented_scope`** | `heavy_atom` **drops hydrogens at load** (`keep = nonzero(represented)`), so `batch.atoms.positions` is heavy-only. `all_atom` keeps them. Both paths already exist. |
| **hydrogen force residual** | `TrainingExample.hidden_force_target` already sums omitted-atom force **per residue**. H2 needs it **per parent heavy atom**, which is new plumbing but needs no new data. |

### 2.1 Three datasets that are present but not read

These are in every shard and no code in `src/` touches them:

| dataset | shape | content |
|---|---|---|
| `psf` | scalar string | full-system CHARMM PSF |
| `dssp` | `[450, 195]` | **per-frame, per-residue** DSSP codes — measured alphabet `{' ', E, G, H, S, T}` |
| `rmsf` | `[195]` | **per-residue RMSF over the whole 450-frame trajectory**, 0.054–0.903 Å here |

**This corrects two `not_applicable` entries in `docs/phase1_6_results_extended.md`.**
I recorded DSSP as unavailable because neither mdtraj nor Biopython is installed,
and RMSF as unavailable because the manifest samples only 4 frames per
trajectory. Both reasons were true about *computing* them and both missed that
the dataset **ships them precomputed**. DSSP and RMSF stratification are
available to H0, and the Stage M report's `not_applicable` block is wrong on
those two rows. It is left as it was written — it is a record of that run — and
corrected here and in the H0 report.

### 2.2 PSF is the authoritative topology, and it parses

`data/psf.py::parse_psf_bonds` exists, documents itself as "the authoritative
bond source", and is **referenced only in a docstring** — nothing calls it.
Verified on the real shard: 3,096 protein-internal bonds for 3,066 atoms, which
is the right order for a protein with one disulfide-bearing chain.

This single fact unblocks most of the brief's chemistry requirements:

* covalent edges for the H2 atom graph — from PSF, not from the distance
  heuristic in `graph/edges.py::build_covalent_bonds`;
* 1–2 and 1–3 clash exclusions — derived from the PSF bond graph by traversal;
* peptide C–N connectivity and **disulfides** — present as explicit bond records,
  so no distance guessing (brief §3.1 forbids it);
* hydrogen→parent mapping for H2 §6.5 — every H has exactly one heavy neighbour
  in the bond list.

---

## 3. What does **not** exist — the real blockers

Searched the whole of `src/` for `atom14`, `atom37`, `rigid_group`, `chi_angles`,
`van_der_waals`, `ideal_bond`, `BOND_LENGTH`, `rotamer`. **No hits.** Neither
OpenFold, AlphaFold, OpenMM, MDTraj, MDAnalysis nor Biopython is installed.

| missing | needed by | severity |
|---|---|---|
| fixed-width atom14/atom37 layout | H0 schema, H1 output | **low** — constructible from `SIDECHAIN_HEAVY_ATOMS`, which already lists every side-chain heavy atom per residue in CHARMM naming |
| van der Waals radii | H0/H1 heavy-atom clash | **low** — Bondi (1964) radii for C/N/O/S are a published, citable table, fixed before any result is seen |
| χ₁–χ₄ atom definitions | H1 torsion head | **medium** — a 20×4 table of 4-atom tuples. A *convention*, not a measurement, so it must be written; it is validated against the data (§4.2) |
| ideal bond lengths / angles | a from-scratch kinematic builder | **avoided by design** — see §4.2 |
| rigid-group / kinematic tree | a from-scratch kinematic builder | **avoided by design** — see §4.2 |
| rotamer library | rotamer-recovery metric | **medium** — reported as `not_supported` unless the metric is redefined as nearest-χ-bin recovery with bins fixed in config |

---

## 4. Design decisions that follow from the above

### 4.1 H0 needs no chemistry it does not have

H0's required mode is `current_local`: take the residue's current heavy atoms in
its current frame and place them on the predicted frame. That is a **rigid
transport** — it needs no bond lengths, no angles, no rigid groups. It is
`reconstruct_backbone` generalised from three atoms to all of them, and the
existing function is literally the N/Cα/C special case of it.

What H0 adds beyond rigid transport is only the *scoring* side: clash (PSF bonds
+ vdW radii) and atom contacts (a cutoff fixed in config). Both are available.

**H0 is fully feasible and will be run.**

### 4.2 H1 avoids the ideal-geometry blocker instead of inventing one

The brief's `HeavyAtomKinematicBuilder` (§5.4) builds heavy atoms from residue
type + frame + torsions using canonical bond lengths and angles. Those constants
do not exist here, and inventing them is exactly what §1 forbids.

The design used instead: **predict Δχ and rotate the residue's own current side
chain about its own χ axes.** Atoms distal to each rotatable bond are rotated
about the measured axis; every bond length and every bond angle in the residue is
carried through unchanged, because a rotation about a bond axis preserves all of
them by construction. This needs only the χ **atom definitions** — which four
atoms name each torsion, and which atoms move with it — and both are derivable
from the χ tuple plus the PSF bond graph.

Consequences, stated so no table can misread them:

* intra-residue bond lengths, bond angles, chirality and ring planarity remain
  **`is_construction_invariant=true`** under H1a, exactly as under H0. H1 does not
  get to claim them as a win.
* what H1 can genuinely move is **χ accuracy, side-chain RMSD, side-chain packing
  and steric clash**, plus — under H1b — the inter-residue backbone geometry that
  Stage M measured as 4–1600× worse than the identity baseline.
* the χ definition table is a chemistry convention I must write. It is
  **validated against the data**: χ computed from real frames must show the
  rotamer structure a real protein has (gauche⁻/gauche⁺/trans near −60/+60/180
  for χ₁ of the sp³ residues). A table with a wrong atom order fails that check.

### 4.3 H2's force representation

Forces are vectors: under `x → Qx + t`, `f → Qf`. They will not be concatenated
into invariant scalar channels. Available representations, all testable for
equivariance: residue-local components `R_iᵀ f_ia`, force norm as a scalar, and
e3nn `1o` channels for the global-frame vector.

Hydrogen handling (§6.5) is feasible in all three modes because the PSF gives an
exact hydrogen→parent map; nothing is inferred from distance.

---

## 5. Modules to add, and files to touch

New, all under `src/force_md/heavy/` so no existing module changes behaviour:

```
heavy_atom_schema.py        fixed-width layout from SIDECHAIN_HEAVY_ATOMS; masks
heavy_atom_topology.py      PSF bonds -> 1-2/1-3 exclusions, H->parent, disulfides
heavy_atom_chemistry.py     Bondi vdW radii, chi definitions, all sourced
heavy_atom_backmapping.py   current_local + the three scoring-only oracles
heavy_atom_metrics.py       heavy RMSD, atom contacts, clash, provenance records
torsion_decoder.py          chi head (H1)
heavy_atom_builder.py       chi rotation on the residue's own geometry (H1)
backbone_constraint_refiner.py  frame residual head (H1b)
atom_force_predictor.py     H2
atom_force_conditioner.py   H2 arms
```

Existing files to touch, additively only:

* `data/adapters/mdcath.py` — expose `psf`, `dssp`, `rmsf`; behind a config flag
  so the default load path is byte-identical.
* `data/residue_constants.py` — nothing removed; new tables live in
  `heavy_atom_chemistry.py`.

Nothing in `transition/` changes. Existing checkpoints are opened read-only with
sha256 verified before and after, as in Stage M.

---

## 6. Expected blockers

1. **Rotamer recovery** has no library. Will be reported as `not_supported`
   unless redefined as χ-bin recovery with bins fixed in config beforehand.
2. **Ring planarity** for HIS/PHE/TRP/TYR is construction-invariant under both
   H0 and H1a (rings are rigid under χ rotation), so it is a sanity check, not a
   score.
3. **Solvent decomposition** is impossible and will not be attempted: the shards
   carry net force on protein atoms only, and no unique protein–solvent pair
   decomposition exists. Recorded as `not_supported`, per §6.5.
4. **`all_atom` load cost.** Keeping hydrogens doubles the atom count. H2 will
   read them only to aggregate onto parents, then drop them.

---

## 7. Metric provenance vocabulary

Every record carries one of these, and the report groups by it:

| value | meaning |
|---|---|
| `model_predicted` | the model actually predicted this |
| `construction_invariant` | guaranteed by rigid transport or χ rotation; never used to choose an arm |
| `scoring_only_oracle` | uses future information; a diagnostic floor, never a deployable result |
| `not_supported` | cannot be computed here; reason recorded, never filled with 0 |
