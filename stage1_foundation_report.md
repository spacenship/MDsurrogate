# Heavy-flow Stage 1 foundation report

> Historical report. Current implementation and training commands:
> [heavy-flow v2](docs/heavy_flow_v2_architecture.md). V2 uses real past history and removes upstream prediction-lag conditioning.

Date: 2026-09-05 (Asia/Seoul)

## Status

The clean-slate foundation is implemented under `src/force_md/heavy_flow/`.
It does not import legacy H0/H1/H2 model modules, losses, or checkpoints. The
existing dirty worktree was preserved; no prior run, checkpoint, or result file
was deleted or rewritten.

The complete heavy-flow pytest suite passes, and the real local ESMC-300M
backend now passes a CUDA mdCATH forward/backward smoke and a strict embedding
cache smoke. Therefore **Stage 1 is ready for next-stage work**. The bounded
smoke assumptions and the intentionally unsupported long-training claims are
listed below. No ESM-2, ESM3, or deterministic stub fallback is used for the
real backend.

## Existing H0--H2 inventory and boundary

| legacy stage | existing implementation and interface | boundary in this stage |
|---|---|---|
| H0 | `src/force_md/heavy/backmapping.py`: `backmap_prediction(prediction, target, batch) -> HeavyAtomPlacement`; current-local atom transport and scoring oracles | not imported; no frame prediction or backmapping is in heavy-flow |
| H1a | `src/force_md/heavy/torsion_decoder.py`: `SidechainTorsionHead(in_features, config)`, `forward(features) -> dict` | not imported; no chi head or side-chain builder is in heavy-flow |
| H1b | `src/force_md/heavy/refiner.py`: `BackboneFrameConstraintRefiner(in_features, config)`, `forward(features) -> FrameCorrection` | not imported; no frame residual is in heavy-flow |
| H2 | `src/force_md/heavy/atom_force.py`: `AtomForcePredictor.forward(features) -> dict`; `AtomForceConditioner.forward(node_features, force_input)`; PSF/hydrogen helpers in the same legacy extension | not imported; new targets retain direct represented-atom forces and expose no force input/head |

The old hierarchy in `data/contracts.py`, old ESM-2 cache, old ESM3 feasibility
path, and old `force_md.nn` blocks remain outside the new encoder boundary.
The new code only reuses residue vocabulary constants and accepts a reader item
as a data boundary; it ignores the reader's legacy PLM embedding and
`hidden_force_target`.

## New files

- `src/force_md/heavy_flow/types.py`: typed dense `[B,L]`, `[B,N]`, `[B,K,N,3]`
  condition; direct `[B,N,3]` force/future targets; unit/source provenance.
- `src/force_md/heavy_flow/data.py`: explicit mdCATH bridge, PSF bond mapping,
  and padded collate. Hydrogen forces are never aggregated into residue or
  parent-heavy-atom targets.
- `src/force_md/heavy_flow/esmc_encoder.py`: sequence-only frozen ESM-C
  interface, detached FP32 content-addressed cache, trainable projection, and
  explicit test-only deterministic stub. The real path uses
  `EsmcForMaskedLM`/`EsmcTokenizer` from the installed `esm` package.
- `src/force_md/heavy_flow/atom_graph.py`: covalent, spatial radius/kNN, and
  optional explicit chain-adjacent edge types; covalent edges survive cutoff.
- `src/force_md/heavy_flow/geometry_encoder.py`: independent generic e3nn
  tensor-product message passing and history geometry encoding.
- `configs/heavy_flow/stage1.yaml`: Stage-1 defaults and unit/provenance fields.
- `tests/heavy_flow/test_stage1_foundation.py`: contract, cache, ordering,
  padding, translation/rotation/permutation, and backward tests.
- `tests/heavy_flow/test_esmc_hf_adapter.py`: current Hugging Face ESM-C API
  routing, canonical residue-to-sequence conversion, and frozen model checks.
- `tests/heavy_flow/test_chain_break_semantics.py`: explicit C(r)--N(r+1)
  chain-adjacent edge and chain-break suppression test.
- `tests/heavy_flow/test_real_mdcath_smoke.py`: one real shard forward/backward
  smoke test using the bounded explicit stub fixture.

## Contract, masks, and units

`HeavyFlowCondition` requires `sequence_tokens`, `residue_mask`, `atom_type`,
`atom_name`, `atom_to_residue`, `atom_mask`, `x_history`, `bond_index`,
`bond_type`, `temperature`, and `lag`. Bonds are local `[B,2,E]` pairs with
`-1` padding. `HeavyFlowTargets` contains only `force_current` and `x_future`,
both `[B,N,3]`, with optional atom masks. The same local atom order is used by
current/history/future/direct-force tensors. Validation rejects active atoms in
masked residues, partial bond pairs, bonds touching padding, and invalid
endpoints.

The mdCATH bridge records length `angstrom`, force `kcal/mol/angstrom`,
temperature `kelvin`, frame time, `1000 ps/frame` provenance, replica, domain,
and frame. Real mdCATH coordinates are centered by the existing safe reader;
the encoder uses only relative positions/displacements, never absolute
positions.

## ESM-C provenance

The configured family is `ESMC-300M`. With `backend="esmc"`, the adapter uses
the exact installed classes `esm.models.esmc.EsmcForMaskedLM` and
`EsmcTokenizer`, with the documented default checkpoint
`biohub/ESMC-300M`. Repository residue IDs are converted through the canonical
residue vocabulary into an amino-acid sequence before tokenization; they are
never treated as ESM tokenizer IDs. The model-reported hidden size and an
explicit revision are recorded in provenance.

The verified local Stage-1 checkpoint is:

```text
model_name: /data1/miplab/wjyang/MDsurrogate/models/ESMC-300M
resolved_revision: local-ESMC-300M
embedding_dim: 960
device: cuda:0
```

The cache key includes:

`model_name + resolved_revision + preprocessing_version + sequence_sha256`.

Cache entries are detached FP32 embeddings with exact metadata checks. The
projection is the only trainable PLM-side module. The additional local
`models/ESMC-600M` checkpoint also loads through the official API and reports
embedding dimension `1152`, but the Stage-1 config intentionally defines and
validates only the 300M family.

Tests use `backend="stub"` explicitly where deterministic synthetic fixtures
are needed and label it `ESMC-300M::explicit-test-stub`; this is not an ESM-C
result. The real mdCATH smoke uses the actual ESMC-300M checkpoint.

## Graph and geometry defaults

Graph defaults are cutoff `6.0 Å`, maximum `32` spatial neighbors, covalent
edges enabled, chain-adjacent edges disabled. Covalent records are taken from
the supplied PSF topology and made bidirectional; they are not removed when
longer than the spatial cutoff. The optional chain edge is only an explicit
same-chain C(r)--N(r+1) record and is suppressed by `chain_break`.

Scalar channels include element/atom name/residue type, backbone/sidechain/
terminal/chain-boundary flags, bond-type summary, frozen-projected sequence
features, history norms/time summary, temperature, and lag. Equivariant inputs
include bounded history displacements and frame-to-frame displacements. Edge
geometry uses relative vectors, distances, radial basis features, and spherical
harmonics through generic e3nn tensor products.

The default geometry is `4` blocks, `lmax=2`, and:

```text
96x0e + 24x1o + 8x1e + 8x2e
```

There is no force input and no force head. The default geometry with the
explicit test stub has `8,395,816` trainable parameters; the real ESM-C
backend is frozen and its original dimension is read after loading. The
independent ESMC-600M load reported `575,036,992` total model parameters.

## Commands and results

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ubuntu/miniforge3/envs/md/bin/python -m pytest -q tests/heavy_flow
12 passed
```

Real mdCATH shard: `data/mdcath_dataset_1a0rP01.h5`, domain `1a0rP01`.

```text
condition: sequence [1,129], history [1,1,1017,3], direct force [1,1017,3]
graph: nodes 1017, directed edges 3089
       covalent 2072, spatial 1017, chain-adjacent 0
       spatial candidates 35372, max retained spatial degree 1
backward: output [1017,15], loss 0.0586732589
ESMC-300M: cuda:0, embedding dimension 960, frozen=True
```

This is a bounded data/forward/backward smoke, not a training result. A real
embedding-cache check also passed: one FP32 cache entry was written with
metadata keys `model_name`, `resolved_revision`, `preprocessing_version`, and
`sequence_sha256`; strict cache-only forward and trainable projection
backward both succeeded without invoking the model.

The additional ESMC-600M checkpoint passed an independent official-API short
sequence forward with output shape `[1,22,1152]`.

The ESM-C runtime emitted non-blocking performance notices: Transformer Engine,
xFormers/flash-attn, and the fused rotary kernel are not installed, so the
official implementation used PyTorch fallback kernels. This affects runtime
performance/kernel-level numerical ordering, not the load or smoke-test
correctness result.

## Unverified assumptions and blockers

- The local checkpoint is pinned in this smoke by the explicit local revision
  label `local-ESMC-300M`; if the file is replaced, update that label and
  provenance before reusing the cache.
- The frame-only adapter uses an explicitly opted-in identity future for the
  one-frame smoke; it is not a future-dynamics target.
- `lag` is represented in frames in this foundation. The mdCATH publication's
  `1000 ps/frame` is recorded, but no per-frame timestamp is fabricated.
- The optional chain-adjacent edge is disabled by default; authoritative PSF
  bonds are the only source of covalent connectivity.
- No long training or scientific performance claim was made at Stage 1.
