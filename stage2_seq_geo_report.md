# Heavy-flow Stage 2: sequence--geometry context report

> Historical report. Current implementation and training commands:
> [heavy-flow v2](docs/heavy_flow_v2_architecture.md). V2 replaces direct ESM fusion with geometry-only residue self-attention.

Date: 2026-09-05 (Asia/Seoul)

## Readiness summary

Stage 1 was checked before Stage 2 implementation. Its core suite passed
(`12 passed`), and the existing real mdCATH/ESMC-C-300M CUDA forward/backward
smoke had already passed on `1a0rP01`. Therefore there was no Stage 1 blocker.

Stage 2 is implemented in the clean-slate `force_md.heavy_flow` package and
does not import or connect legacy H0--H2, residue-frame transition, physics
slot, or teacher modules. The full heavy-flow suite currently passes with
`19 passed` (12 Stage 1 and 7 Stage 2 tests).

The Stage 2 interface is ready for Stage 3 integration/prototyping. This is a
bounded representation and auxiliary-objective readiness result, not a claim
of trained force accuracy or scientific validation; Stage 3 still needs its
own training/evaluation gates.

## Architecture and tensor flow

```text
HeavyFlowCondition
    │  sequence-only frozen ESM-C + atom graph/history
    ▼
H_a^local  [M, D_geo], e3nn irreps
    │  scalar attention, atom mask, atom_to_residue
    ▼
G_i^geo    [B, L, D_geo] equivariant pooled feature
    ├── geometry_invariant [B, L, I] ──┐
    │                                  │ K/V + pair bias
    └── geometry_equivariant [B,L,D]   │
                                       ▼
ESM-C frozen → trainable projection → scalar cross-attention
                                       ▼
C_i^res = SeqGeoResidueContext [B, L, 384]
                                       │ indexed broadcast by rho(a)
                                       ▼
C_a^atom = 2-block e3nn refinement [M, D_geo]
```

The implementation is split across:

- `equivariant_pooling.py`: masked scalar-attention atom-to-residue pooling.
- `seq_geo_fusion.py`: invariant token construction, pair-biased cross-
  attention, and the `SeqGeoResidueContext` named boundary.
- `atom_context.py`: residue broadcast, local-feature skip, and two e3nn
  refinement blocks.
- `context_encoder.py`: the only composition point for the Stage 2 path.
- `auxiliary_heads.py`: masked residue-geometry reconstruction and bonded
  distance denoising.

## Exact formulas and mask behavior

For packed valid atoms, the attention logit is

\[
 \ell_a = f_\mathrm{attn}([\operatorname{Inv}(H_a^\mathrm{local}),
                            \operatorname{Identity}(a)]),
\qquad
 \alpha_a = \operatorname{softmax}_{a:\rho(a)=i,\,m_a=1}(\ell_a),
\]

and each equivariant channel is pooled without changing its representation:

\[
 G_i^{l,p}=\sum_{a:\rho(a)=i}\alpha_a H_a^{l,p}.
\]

Masked atoms receive zero attention weight. Active residues with no valid
atoms receive a zero feature; inactive residues are zero in the padded output.
The packed output order is active residues in batch-local sequence order.
The pooling weights are invariant scalars, so the same expression preserves
scalar channels, polar/axial vector parity, and tensor channels. It is not a
force sum.

The standard Transformer token is

\[
 G_i^\mathrm{inv} =
 [G_i^{0e},\; \operatorname{RMS}(G_i^{1o}),\;
   \operatorname{RMS}(G_i^{1e}),\; \operatorname{RMS}(G_i^{2e}),\ldots].
\]

All non-scalar irrep multiplicities contribute norms; even scalar channels are
retained and an unexpected odd scalar is made parity-safe with an absolute
value. Thus no vector x/y/z component is concatenated to a scalar token.
The full equivariant `G_i^geo` remains available separately.

The pair feature for residue pair `(i,j)` contains normalized sequence
separation, same-chain indicator, peptide adjacency, contact indicator,
chain-break adjacency indicator, and 16 distance RBFs. The learned pair bias
is `MLP(pair_features) -> attention_heads` and is added to scalar cross-
attention logits:

\[
 P_i=\operatorname{CrossAttn}(Q=E_i^\mathrm{seq},
 K=G_j^\mathrm{inv},V=G_j^\mathrm{inv};b_{ij}),
 \qquad
 C_i^\mathrm{res}=\operatorname{FusionBlocks}(E_i^\mathrm{seq},G_i^\mathrm{inv},P_i).
\]

The configured projected ESM-C tensor is the query-side sequence input. The
original `esm_frozen` tensor is returned unchanged and detached; only the
separate projected tensor participates in trainable fusion.

For each valid atom,

\[
 B_a^\mathrm{res}=C_{\rho(a)}^\mathrm{res},\qquad
 C_a^\mathrm{atom}=\operatorname{Refine}^{e3nn}
 (H_a^\mathrm{local}\oplus B_a^\mathrm{res}\oplus I_a),
\]

where the original local atom feature also enters an explicit skip path.
Element, atom name, residue type, backbone/side-chain/terminal/chain-boundary
flags, local geometry, covalent/spatial edge types, masks, and Stage 1 order
are retained. Learned atom identity channels prevent same-residue atoms from
collapsing to the same representation.

## Irreps and defaults

The Stage 1 default local feature is

```text
H_local / G_geo / C_atom: 96x0e + 24x1o + 8x1e + 8x2e  (D=232)
G_inv: 96 + 24 + 8 + 8 = 136 invariant channels
```

The Stage 2 default configuration in `configs/heavy_flow/stage2.yaml` is:

```yaml
joint_scalar_dim: 384
fusion_blocks: 4
attention_heads: 8
pair_bias: true
dropout: 0.1
atom_refine_blocks: 2
```

The real bounded mdCATH smoke used the deliberately smaller test profile
`4x0e + 1x1o + 1x1e + 1x2e` (`D=15`, `G_inv=7`), two scalar fusion blocks at
width 32, and two atom refinement blocks. This avoids confusing a bounded
correctness run with a full training configuration.

## Auxiliary objectives

Two objectives are implemented:

1. **Masked residue geometry reconstruction.** Selected residue joint inputs
   are zero-masked, while the clean pooled invariant geometry target remains
   available to the loss. The head predicts all invariant geometry channels.
2. **Bonded-distance denoising.** For directed covalent edges, an MLP predicts
   the clean bonded distance from endpoint invariant atom features and the
   input distance. The API accepts a noised-input encoding and a separate
   clean `target_encoding`; no force or future target is used.

The tests use small coordinate noise and a fixed masked-residue pattern. The
objective is optimized only for a bounded overfit smoke, not as a training
claim.

## Modality ablations and invariance checks

The synthetic tests explicitly compare normal output with sequence modality
dropout and geometry modality dropout; both change `joint_scalar`. They also
check that sequence-derived ESM-C inputs are not silently ignored through the
fusion path.

The synthetic checks cover atom permutation invariance, residue rotation
equivariance, empty residue behavior, atom-order-preserving broadcast,
non-collapse of same-residue atom features, scalar-only Transformer geometry
input, unchanged frozen ESM-C tensor, rotation-invariant `joint_scalar`,
rotation-equivariant `C_atom`, both modality dropouts, and gradients through
fusion/refinement.

## Parameter and memory accounting

Using the Stage 2 default architecture with an explicit test stub only for
parameter accounting (the real ESM-C backend is frozen):

```text
trainable parameters: 19,827,577 (19.828M)
```

The real bounded mdCATH profile had `56,095` trainable parameters because it
used the small geometry/fusion profile above; the ESMC-300M weights remain
frozen. CUDA measurements for that run were:

```text
real ESMC-300M model-loaded baseline: ~1307.22 MiB
peak after reset during bounded forward: 1317.82 MiB
increment above loaded baseline: ~10.60 MiB
synthetic bounded stub forward peak: 32.36 MiB
```

These are allocator measurements for one `1a0rP01` batch, not a full-training
memory estimate. The local ESMC-300M checkpoint reports hidden dimension 960;
the ESM-C backend is detached/frozen and only its projection is trainable.

## Tests and smoke results

Stage 1 was rechecked first:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ubuntu/miniforge3/envs/md/bin/python -m pytest -q tests/heavy_flow
19 passed
```

The Stage 2 synthetic noised/masked auxiliary overfit decreased:

```text
initial 5.3536987 -> final 0.1995855  (12 bounded head steps)
```

The real mdCATH subset was `data/mdcath_dataset_1a0rP01.h5`, domain `1a0rP01`:

```text
condition: sequence [1,129], history [1,1,1017,3], direct force [1,1017,3]
graph: 1017 nodes, 3089 directed edges
forward: pooled [1,129,15], G_inv [1,129,7], joint [1,129,32], C_atom [1017,15]
grad-enabled auxiliary loss: finite; fusion_grad=True; refine_grad=True
esm_frozen.requires_grad: False
real noised/masked auxiliary overfit: 3.2069838 -> 0.9771878 (8 head steps)
```

No full training was run. The ESM-C runtime still reports nonblocking fallback
kernel notices (Transformer Engine, xFormers/flash-attn, fused rotary kernel)
in the `esm3` environment; the official local checkpoint load and all finite
forward/backward checks succeeded.
