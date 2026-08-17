# task262 hard record - final conv placer tail is dead (2026-07-15, bo_ng_cdx2_worker2)

Reviewer-locked hypothesis tested: fuse live `small2 -> Resize` into a final
free-output placer using #67/#68/#85, with #117 as the live Resize baseline.

Authoritative bar: **74** from `python tools/ng_current_bar.py --task 262 --json`.
Manifest owner Vamshi, updated `2026-07-12T15:15:54Z`, sha256
`0092601f6f6318517ba191547b677110d540870a0f16b6ee0f542f3c4b6bdc0e`.
Live profile: `static=74 profile=74 (memory 48 + params 26)`.

## Probe 1 - direct QLinearConv placer

Path:
`The_400_Tasks/Task_by_Task/task262/prototypes/task262_direct_qlinearconv_placer.onnx`

Build:
`Slice(ch5, rows0:3, cols0:2)` -> `Cast(INT8)` ->
`QLinearConv(output [1,10,30,30])`, `kH=1`, `kW=3`,
`pads=[0,1,27,29]`, no `small2`, no `Resize`.

Cost/profile:
`static=71 profile=71 (memory 30 + params 41)`.
SHA-256:
`0981750f59c78d0198d0854062c4b193542a1cb1d0fca7098d1231378c5b5fbb`.

Gate:
`python tools/ng_gate.py visible --task 262 --cand ...task262_direct_qlinearconv_placer.onnx --no-watch-queue`
-> `[visible] 0/11 (0.00%)  FAIL`.

Verdict: `REJECT_LOCAL`; visible-zero family.

## Probe 2 - direct ConvInteger placer

Path:
`The_400_Tasks/Task_by_Task/task262/prototypes/task262_direct_convinteger_placer.onnx`

Build:
same crop/cast -> `ConvInteger(output [1,10,30,30])`, `kH=1`,
`kW=3`, `pads=[0,1,27,29]`, no `small2`, no `Resize`.

Cost/profile:
`static=70 profile=70 (memory 30 + params 40)`.
SHA-256:
`5e06f16941593591f128a83abba2116c03d686c3b1cdcb340cf941b69a4b6660`.

Gate:
`python tools/ng_gate.py visible --task 262 --cand ...task262_direct_convinteger_placer.onnx --no-watch-queue`
-> `[visible] 0/11 (0.00%)  FAIL`.

Verdict: `REJECT_LOCAL`; visible-zero family.

## Why the priced final-conv route cannot work below 74

The live graph's decisive trick is that `QLinearConv` emits only
`small2 [1,3,3,2]` and free-output `Resize` expands/places it across both the
channel axis and the spatial width. A direct final conv must instead emit
10 output channels itself.

For any strict sub-74 final `QLinearConv` placer using the required crop:

- Memory floor is already `24 + 6 = 30` (`F32 crop [1,1,3,2]` plus INT8 cast).
- Slice params are 8, scalar qparams are at least 3, and dense weights cost
  `10*kH*kW`.
- `30 + 8 + 3 + 10*kH*kW < 74` forces `kH=1` and `kW<=3`.

For `ConvInteger`, scalar params drop to 2, but the same inequality still forces
`kH=1`, `kW<=3`. F32 `Conv` has fewer scalar params, but without a bias it cannot
make the "neither input bit is present" color positive; with a bias it violates
#85's bias-leak law and paints the padded/off-band canvas.

With `kW<=3`, consecutive output columns use overlapping shifted two-bit windows.
To paint output columns 0, 1, and 2 identically, the same asymmetric two-bit
classifier must be copied at three adjacent phases. The overlapping phases force
one kernel position to serve as both the col0 coefficient and the col1 coefficient.
Those coefficients must be different to distinguish `col0 -> color2` from
`col1 -> color4`, so the copy cannot be represented by a single translation-
invariant width-3 kernel. Increasing stride to avoid overlap requires a replicated
bank spanning at least `2 + 2*stride` columns; the first non-overlapping case
(`stride=2`) needs `kW>=6`, adding at least 60 dense weight params and pushing the
QLinearConv route above 100 total cost.

Therefore the only known way to keep the bank small is the live charged sentinel
`[1,3,3,2]` followed by free `Resize`. Removing that sentinel requires a larger
weight/basis bank, which loses the byte target.

No staged candidate.
