# task066 ArgMax/index-plane route — dead

- Live: profile cost 2689, SHA-256 `6eb4f4fe602493ef95da9e8e45df7c9281f2b210da93c711a603dd5b2fa7731e`; effective documented bar 2657 from the prompt/shared-basis probe.
- Attempt: one full `[1,1,30,30]` INT64 `ArgMax(input, axis=1)` color-index plane, then `Equal` for colors 2/3 and F32 scalar-moment masks, with all downstream live branches preserved.
- Failed gate: `profile_real_cost.py --arcgen 266` → `static=None profile=18871 (memory 18631 + params 240; 266 real inputs)` after visible `266/266 PASS`.
- Dead because: the index plane alone is 7200 bytes; named added carriers total 16200 memory bytes, so the formulation is +16214 over the 2657 bar and +18151 over the 720 target. Do not build cropped, Gather-LUT, or Equal-tail variants in this representation family.
- Floor cross-check: matrix ranks `(path, 1-path)` over all 266 bundled pairs are `{(2,3):73, (3,4):193}`. For the current explicit-base input-reading separable renderer, the rank-4 cases require four F32 row and four F32 column selectors, or `4*30*2*4 = 960` charged bytes before detection/params. This is family-specific evidence, not a universal ONNX lower-bound theorem, because a fused channel-gated factorization could exploit don't-care cells. Equal-tail unconditionally forces a 900-byte U8 canvas.
- One remaining hypothesis: a sub-720 graph would require a non-separable primitive that is itself the free graph output and extracts dynamic path geometry without a spatial carrier; no concrete legal construction is known.
- Breadcrumb only, not a floor route: reusing `active_basis[4,10]` for the four marker moments could predictably shave ~12 params to ~2645, but the task prompt explicitly says to stop at this priced wall rather than grind rung-epsilon variants.
