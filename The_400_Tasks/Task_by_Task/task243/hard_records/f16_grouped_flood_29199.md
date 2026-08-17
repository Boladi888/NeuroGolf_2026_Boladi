# task243 DEAD ROUTE — F16 grouped flood (2nd-pass, bo_ng_cld_worker3 2026-07-12)

**Verdict: DEAD. A CORRECT F16 single-axis flood costs 29,199 — 4.86× the F32 6014.**
Do not re-attempt F16 (or any sub-F32 dtype) for this task's flood carriers.

## What was tried
Genuinely-different DTYPE rewrite of the 6014 rank1_alt: the identical 64-round
single-axis alternating gated flood, but in FLOAT16 to halve every state tensor
(1296 F32 → 648 F16). Builder: `prototypes/build_task243_f16_grouped.py`.

## Why it must be GROUPED (technique #77 + probe_f16_range.py)
The flood is walk-COUNTING (pure sum-of-products of 0/1 → integer counts that
grow ~×3 per gated round; F32 maxval 8.6e27 over 64 rounds). An ORT F16 Einsum
accumulates internally in F32 but **stores its final output in F16** (#77). A
single deep Einsum output therefore overflows F16 (3^11 = 177147 > 65504 → inf,
then inf×0 = NaN). To keep every stored output finite, each Einsum is capped at
≤10 rounds (3^10 = 59049 < 65504) and REBINARIZED with `Min(_,1)` (exact for
integer walk counts) between groups. 64 rounds → 7 Einsum groups + 6 Min nodes.

## Measured cost (scorer path, visible-confirmed correct)
- **visible 265/265 PASS** — grouping DID prevent overflow; the graph is correct.
- **score_v2 = 29,199** (memory 28,296 + params 903).
- Per-tensor memory (why it loses):
  - `input16` (F32→F16 Cast of the whole [1,10,30,30] input) = **18,000 B**.
    The F32 flood consumes `input` DIRECTLY, and `input` is memory-EXCLUDED
    (`calculate_memory` line 234). F16 Einsum needs an F16 operand → a charged
    full-input cast. This tax alone (18,000) exceeds the entire F32 cost (6014).
  - F16 flood **body** = 10,296 B (gate 648 + 7 state groups + 6 Min-rebins +
    fill 324 + fill_b30 900). Even body-only is 2× the whole F32 graph (5112).

## The forced-component reason (general, not just this build)
`calculate_memory` SUMS every intermediate tensor. The Einsum-fold's entire value
is FEW tensors: F32 holds ~32 rounds/Einsum (letter-budget-limited), so 56+ rounds
= exactly 2 state tensors (2×1296 = 2592). F16 holds only ≤10 rounds/Einsum
(overflow-limited) → ≥6 Einsums for 56 rounds → ≥6 states + 5 rebins = 11 tensors
× 648 = 7128 > 2592, BEFORE the 18,000 input-cast tax. Halving per-tensor bytes
can never beat the ≥5.5× tensor-count multiplier that rebinarization forces.
Rebinarization is non-multilinear ⇒ cannot fold into Einsum ⇒ the fold is lost.

**Net: F32 2-Einsum (6014) is the dtype-optimal point. F16 is categorically worse.**
Cross-ref: `probe_f16_range.py` (overflow measurement), techniques #77 (F16 exec
model), core.md sparse note (the sibling dead lever). Floor detail: the 6014
staged `manifest.md` "Cost floor note".
