# task059 — archived history

Superseded prose moved out of `task059.md` per WORKER_PROTOCOL staging rule
(manual content kept ≤ ~15 KB; only the CURRENT best 557 floor proof stays inline).

## Superseded 861 Einsum-basis-family floor proof (2026-06-10, provisional per #134)

This was the floor proof for OUR staged 861 `task059_einsum_basis5` graph. It is
SUPERSEDED by the team-best Aditya 557 ship (a different, cheaper family — see the
"Aditya 557 ship — exact graph + tightened forced-component floor" section in
`task059.md`). Kept here for provenance only.

**861 is the minimum achievable with the Einsum-basis free-output algorithm family.**

Forced components (each independently verified):

1. **Counting basis** [5,30] F32 = 150 params: any alternative that computes per-compartment
   object counts without this basis requires first slicing [1,10,11,11] F32 = 4840 bytes (a
   charged intermediate that alone exceeds the entire 861 budget). Sparse basis rejected by
   ONNX scorer (see attempts). The [5,30] F32 basis is minimal: only 11 non-zero elements
   but sparse storage is blocked.

2. **Rendering basis** [5,30] F16 = 150 params: free-output Einsum requires the spatial basis
   to map 5 groups → 30 positions. Per #132: "make the basis map straight to the full 30-axis
   so the einsum output IS the 30×30 canvas." Any crop+Pad approach creates a charged
   intermediate. One shared basis + Cast costs +150 bytes net (save 150 params, gain 300B
   Cast intermediate). The dtype split (F32 for counting, F16 for rendering) is unavoidable.

3. **Comparison chain**: ReduceMax [4B] + Equal→bool [25B] + Cast→F16 [50B] = 79 bytes minimum.
   These are single-node outputs with no cheaper alternative (no UINT8 Einsum, no BOOL Einsum).

4. **Color detection**: ReduceMax→present_f32 [40B] + Cast→present_num [20B] = 60 bytes minimum.
   The F32→F16 Cast is required because input is F32 but rendering Einsum is F16.

5. **Dynamic-plane assembly**: ch3 [60B] + coef3 [150B] + fill_delta [20B] = 230 bytes minimum.
   The Concat to form ch3 and coef3 is mandatory (Einsum can't add independent components).

6. **Auxiliary params**: count_ch (10) + count_one (1) + fill_sub (10) + batch1 (1) + static_ch2
   (20) + static_coef2 (50) = 92 params minimum. Each encodes non-trivial information
   (channel mask, fill formula, static plane weights) that cannot be further compressed.

Sum: 392 params + 469 bytes = **861**.

Per #134 discipline: this is a **per-algorithm conditional floor** for the Einsum-basis family.
A from-scratch rewrite with a different algorithm could potentially beat it, but:
- All alternative counting methods tried cost MORE due to creating [1,10,11,11] intermediates
- The Einsum free-output approach is #132's optimal construction for fixed-layout stamp-tiling
- No cheaper algorithm structure for (count → max → compare → render) has been found

Label: **CONDITIONAL FLOOR** — reopen only with a provably-cheaper from-scratch algorithm.
