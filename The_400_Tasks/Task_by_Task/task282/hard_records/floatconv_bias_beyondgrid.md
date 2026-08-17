# DEAD ROUTE — float Conv+bias (drop ConvInteger) — task282

- **Live bar:** 345 (manifest, authoritative), sha256 `c1d1471a4566e8315d6901754864eb540ae28baece27ec97f1f6ff3760673ffb`, owner Vamshi.
- **Date:** 2026-07-06 · worker `bo_ng_cld_worker3`

## Rule (decoded)
Input 9×9, isolated color-5 seeds (rows/cols 1–7, min Chebyshev spacing **3** → 3×3
stamps never overlap; ≤1 seed per 3×3 window). Each seed gets a 3×3 stamp centered on
it: center→0, orthogonal→1, diagonal→5; everything else→0. Verified on all 265 pairs.

## Attempted route
Replace live `Slice(f32)→QuantizeLinear→ConvInteger` with `Slice(f32)→float Conv[10,1,3,3]+bias`.
Motivation: a float Conv reads the f32 seed slice directly, so it needs NO uint8 `code`
plane (−49 B memory). Clean integer weights: ch0 = ortho/diag −2, center 0, **bias 1**;
ch1 = +1 at N/S/E/W; ch5 = +1 at diagonals; ch2,3,4,6,7,8,9 = 0.
Built `prototypes/floatconv_304.onnx`.

## Cost / gate
- score_v2 = (0, **196**, **108**) = **304** (would beat 345 by −41).
- `ng_gate visible --task 282` = **0/265 FAIL**.

## Why it's dead (forced-component argument)
The in-grid 9×9 decode is **perfect** — the stamp weights are correct. It fails ONLY the
beyond-grid region: the output is [1,10,30,30] and the expected one-hot is all-zero
outside the 9×9. A float Conv **zero-pads**, so every beyond-grid cell sees an all-0
window → channel-0 = **bias(1) > 0** (630 spurious positives per pair). The bias is
mandatory because in-grid *background* cells (color-0, far from any seed) also have
all-0 seed-neighborhoods and need channel-0 > 0. A Conv+bias **cannot distinguish**
in-grid background from beyond-grid background — both are all-0 input windows.

The live `ConvInteger` sidesteps this because a quantized Conv **pads with the
zero-point** (=1): beyond-grid windows contribute `(1−1)=0` → all channels 0, while
real in-grid background 0-cells become `(0−1)=−1` → the channel-0 background signal.
This zero-point-padding + input-shift is the ONLY cheap mechanism that zeros beyond-grid
while lighting in-grid background, and it **requires an integer Conv input** → the f32
seed slice MUST be cast to a uint8 `code` plane (+49 B). Hence the code plane is forced,
not removable.

## Remaining hypothesis (also priced out — see task282.md floor proof)
Slicing the ch0 (background) plane instead would distinguish in-grid (ch0=1) from
beyond-grid (ch0=0), but ch0 spans the full 9×9 → f32 slice = 324 B > 196 B, so the
family is strictly worse (≥432). **Do not re-explore float-Conv or ch0-slice.**
The live 345 is the forced floor (proof in task282.md).
