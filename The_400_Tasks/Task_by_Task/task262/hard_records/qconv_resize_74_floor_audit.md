# task262 hard record - qconv-resize 74-byte audit (2026-07-14, bo_ng_cdx2_worker1)

Live bar used: **cost/profile 74**, sha256
`0092601f6f6318517ba191547b677110d540870a0f16b6ee0f542f3c4b6bdc0e`.
Source: `python tools/ng_current_bar.py --task 262 --json` -> authoritative
manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.

Current graph: `Slice(input -> [1,1,3,2] F32)` -> `Cast(INT8)` ->
`QLinearConv(INT8 [1,3,3,2])` -> free-output `Resize(INT8 [1,10,30,30])`.
Measured with `profile_real_cost.py --arcgen 266`: static/profile **74**
(`memory 48 + params 26`).

## Attempted route 1 - collapse `small2` width 2 -> 1. PRICE/SEMANTIC DEAD.

Why tempting: `[1,3,3,1]` would save 9 bytes of INT8 memory and put the graph at
65 if the final `Resize` still produced only columns 0..2.

Why dead: `Resize(tf_crop_and_resize)` with a width-1 source has no second source
column/extrapolation boundary; it stretches the active class across the 30-column
output. The current width-2 `small2` is load-bearing: source column 0 carries the
row color and source column 1 is the zero boundary that turns columns >=3 off.

Failed gate: **price/semantics pre-gate**; no `ng_gate visible` was run because the
representation cannot preserve the required 3-column active output.

## Attempted route 2 - share all QLinearConv zero-points. INFEASIBLE.

Current Cast features are raw `(a,b)` states `(1,0)`, `(0,1)`, `(0,0)` for col0,
col1, col2. Saving one param would require a single shared scalar zero-point for
`x_zero_point`, `w_zero_point`, and `y_zero_point`.

Finite audit in `prototypes/audit_qconv_resize_74.py` exhaustively searched INT8
`z,a,b` and found no three one-vs-rest QLinearConv classifiers over those states.
The current two-zero-point design is functionally load-bearing.

Failed gate: **algebraic feasibility pre-gate**; no ONNX candidate emitted.

## Attempted route 3 - centered QuantizeLinear coding. TIE, not a win.

`QuantizeLinear(scale=.5,zp=-1)` maps crop bits `0/1 -> -1/+1`. Then a single
QLinearConv zero-point `0` separates the three classes with weights
`[1,-1]`, `[-1,1]`, `[-1,-1]`.

This is algebraically valid, but not cheaper: replacing free `Cast` with
`QuantizeLinear` adds the scalar initializer that the QLinearConv zero-point
sharing saves. Param price remains 26 and memory remains 48, so total remains 74.

Failed gate: **cost gate**; equal to live, not strictly below team-best.

## Remaining hypothesis

Only a different terminal op family can plausibly beat 74. It must either (a)
perform the 2-bit row classifier and the 3-column/3-row placement in one free
output op without the `[1,3,3,2]` sentinel tensor, or (b) read the raw F32 input
with fewer than the current 48 charged bytes plus 26 params. Direct raw-input
Einsum routes reintroduce 30-long F32 selectors; direct full-output QLinearConv
needs larger channel/kernel weights or cannot duplicate the first three columns.

No staged candidate from this pass.
