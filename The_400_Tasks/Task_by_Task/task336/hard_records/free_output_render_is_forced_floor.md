# free_output_render_is_forced_floor — task336 WALL at 400

## Live bar (refreshed this batch)
- **cost 400** = memory 181 + params 219. static == profile == 400 (31 real inputs).
- owner Vamshi, sha256 `418eee6e3107fe8c2c28ab6ecf0464fc47e7b07f007446ba5d25c1d83fcabc89`,
  source `Team_Submission/Backend/manifest.csv` (authoritative) + workbook, updated 2026-07-02T02:35:15Z.
- `tools/ng_current_bar.py --task 336` authoritative_source=manifest, not stale.

## Prompt premise is STALE
Prompt (`Prompt_...Task336_ChannelCollapseEqualTail.md`) targets `floor_est ~260` by "collapsing two
F16 [1,10,10,10] planes (4000 B each) to a U8 color-index". That describes the OLD 14670-cost
93-node ArgMax/Conv graph. The **live 400 graph is already a from-scratch free-output rewrite**
(`task336_rank2_quadratic_final_einsum`): scalar descriptors (T,MR,MC via Einsum) + a rank-2 quadratic
separable render that writes **directly to `output`**. There are no F16 planes to collapse; the 260
figure is "1.5x below a number that no longer describes the live graph." No 260 to reach.

## Cost model (read from `neurogolf_utils.calculate_memory/params`, the official scorer)
- **params** = sum of ALL initializer element counts (dtype-independent; F16 does NOT cut params).
- **memory** = sum of bytes of ALL intermediate node-output tensors (max static/runtime),
  **but graph `input` and `output` are EXCLUDED**.
- Exact split: params 219 = B[3,30]=90 · Vfac[5,5]=25 · U[5,5]=25 · cout[2,10]=20 · cin[2,10]=20 ·
  Q[2,5,2]=20 · {one/x/z/z2}coef 12 · graymode 2 · 5 scalars 5.
  memory 181 = features[5,3]=60 · ucoef 12 · qD 12 · obasis 8 · 22 scalars×4=88 · Hbool(bool)=1.

## Cornerstone (EMPIRICAL): the output is free; any pre-output materialization is fatal
Wrapped the final render Einsum output in `Identity render_out -> output` and scored:
`(0, 36181, 219)` = **36,400** (memory jumped 181 -> 36,181, +36,000 B for the [1,10,30,30] F32 render).
=> the minimal graph MUST render directly into `output`. Any crop/canvas/mask formulation that
materializes even a [1,10,10,10] F32 result before the output pays +4,000 B memory; a 30x30 pays +36,000.

## Routes attempted / killed
1. **Crop -> narrow basis -> pad** (B[3,30]->[3,10] saves 60 params): DEAD by the cornerstone.
   Cropping forces a pre-pad render intermediate [1,10,10,10] F32 = 4,000 B memory (currently the render
   writes free-to-`output`). Net ~ -60 params +4,000 memory = catastrophic loss. Not built (analytic).
2. **Mask-cumulative / Where-canvas (Family C)**: DEAD by the cornerstone + traps §109. Every 10x10 F32
   plane (interior mask, ray mask, colour assembly) = 400 B memory each; need >=2-3. Provable >=~800.
3. **Feature/degree truncation** (prior batch): drop z2 quadratic row + shrink U/Vfac rows ->
   static 375 but `ng_gate.py visible --task 336` = **0/31 FAIL**. Degree-2 basis and the 5-feature bank
   are load-bearing (see `reduced_quadratic_basis.md`). Certain hidden 0.
4. **ReduceMax carrier fold** (prompt's stale angle): no ReduceMax exists in the live graph
   (see `reducemaxcarrier_absent_in_live.md`).
5. **F16 the pipeline**: params are dtype-independent (no gain). Memory F16 blocked: the render Einsum
   shares dtype with the F32 `input`, and casting the [1,10,30,30] input to F16 materializes an
   18,000 B intermediate. Partial-F16 features can't feed a mixed-dtype Einsum. No gain.

## Forced-floor argument (why 400 stands)
Minimal graph => render directly to `output` (cornerstone) => separable polynomial basis over the full
30-wide frame => B[3,30]=90 params forced (degree-2 necessary [route 3]; full-width necessary [route 1]).
Rank-5 feature factorization (U+Vfac+Q=70) each encodes a distinct geometric role
(constant/interior, linear ray-shift, quadratic solid, z-shift, z^2) — feature-row truncation already
visible-0. Colour maps cin+cout=40 forced by the fixed 10-channel one-hot color axis and the 2-mode
(background->8 / gray passthrough) remap. Remaining 19 params are irreducible poly coeffs + scalars.
=> **WALL at 400.**

## One remaining unfalsified hypothesis (for the next batch)
The *coupling rank* r (not the feature count) is 5 in U[5,5]/Vfac[5,5]/Q[2,5,2]. Prior work only killed
feature-ROW truncation, not a genuine rank-4 REFIT. If the 5x5 bilinear coupling is numerically rank-4,
a from-scratch refit to U[5,4]/Vfac[5,4]/Q[2,4,2] would save ~14 params -> ~386. Requires re-solving
Vamshi's exact byte-equal one-hot factorization; low prior (5 features encode 5 independent roles), high
effort. Only route not yet analytically or empirically closed. Do NOT re-run routes 1-5.
