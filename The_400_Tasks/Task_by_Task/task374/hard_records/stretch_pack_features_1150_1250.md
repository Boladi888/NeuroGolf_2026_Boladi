# task374 hard record — stretch 1150-1250 (pack row/col count / min-max bits) is a WALL

- **Live bar**: 1366 (Vamshi), sha256 `8b6fdcc9538df7001e3b861cdae1861f27b94ba5dcabfbee31a94584cb90426a`,
  workbook `NG_OnnxlyFans_TaskStatus.xlsx` Tasks row + `Team_Submission/Backend/manifest.csv`, updated 2026-07-02T02:35:15Z.
- **Current best (staged, holds)**: `task374_qscale_share_cost1365` cost 1365 (memory 1310 + params 55), profile 1365,
  sha256 `a4b47de87e8f317148931ead2f15f21c0b0479e1824b25f665dd674a230e5e35`. -1 vs bar. byte-equal to submission.zip.
- **Route attempted / walled**: pack the two feature planes into one before the final `ConvInteger` (the named 1150-1250 lever).

## Architecture (decoded from the staged 1365 graph)
`Slice(input[:,5:6,:10,:10])`→F32 crop→`Cast`→gray_u8. `QLinearConv` chain derives `row_lengths`
(horizontal segment len per row) and `col_lengths` (vertical segment len per col). Global max/min over
present lengths. Two per-cell U8 feature planes encode `(is_min?0:10, is_max?0:10)`:
`selected_features = Where(horizontal_bool, row_features, col_features)`; `masked_features = selected * gray_u8`;
`output = ConvInteger(masked_features - 5, final_weights - 5)` with pads `[0,0,20,20]` expanding 10x10→30x30 free.

`final_weights[10,2,1,1]` is a **linear** 2-channel decoder. After `- feature_zp(5)` the four codes map to argmax colors:
`(5,-5)→c1`(max/longest), `(-5,5)→c2`(min/remaining), `(5,5)→c4`(mid/second), `(-5,-5)→c0`(bg / non-gray).

## Why 1-plane packing is INFEASIBLE (forced-component argument)
A single feature channel `v` through a 1x1 `ConvInteger` gives `output[c] = (v - zp)*(w[c]-zp)` — **linear in v**.
Argmax over c of a linear ramp yields only **three** distinct outcomes: `v>zp → argmax(w)`, `v<zp → argmin(w)`,
`v=zp → all-zero → c0`. But the decoder needs **four** distinct outcomes (bg, max, min, mid) = **three non-background
gray classes**. Three gray classes cannot be separated by a 2-way sign split. ⇒ ≥2 feature channels are required for
a linear ConvInteger decoder. Packing min/max bits to one plane is provably impossible without changing the final op.

## Why the two 200-B planes can't collapse to one
`selected_features` (Where select row-vs-col, 200 B) and `masked_features` (Mul by per-cell gray, 200 B) are both
counted (score_v2 memory is a **raw sum of every intermediate tensor**, no liveness/peak). A 2-branch `Where` yields
only 2 outcomes; the third (non-gray→0) needs a separate per-cell mask (`row_features[1,2,10,1]`/`col_features[1,2,1,10]`
carry no per-cell resolution). Any arithmetic select (`col_f + h*(row_f-col_f)`) or nested Where produces ≥the same
number of `[1,2,10,10]` tensors — every re-expression tested costs ≥ current 400 B. The `horizontal` bool (100 B) is the
cheapest possible select condition; replacing it with integer-Mul select costs +300 B.

## Memory floor for this decoder family (all tensors forced)
`gray_crop_f` F32 400 (input Slice inherits F32; any Cast-first is 9000 B) + `gray_u8` 100 + `horizontal_u8` 100 +
`horizontal` bool 100 + `selected_features` 200 + `masked_features` 200 + row/col lengths & features & max/min ≈ 210
≈ **1310**. params 55 is tight (final_weights 20 fixed by 10 output colors; Slice starts/ends 8; kernels 23; 4 scalars).
No sub-1365 exact graph fits with these forced tensors present.

## Alternative expander (also worse)
Replacing the ConvInteger 30x30 pad-expander with an Equal/OneHot one-hot expander reintroduces the 900-B `label30`
sentinel path (old v8 lineage, already documented dead). ConvInteger pad-expansion is strictly cheaper.

## Remaining hypothesis (only escape)
A different *output convention* (non-one-hot final tensor the grader still accepts) or a scorer loophole would be
needed to drop below 1365. Not pursued — this is not a floor-proving task and the named lever is falsified above.
1365 recapture is the banked win.
