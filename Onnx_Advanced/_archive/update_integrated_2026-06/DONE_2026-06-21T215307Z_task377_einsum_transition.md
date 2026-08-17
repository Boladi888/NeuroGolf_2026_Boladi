# task377 ordered Einsum transition-count selector

## What I discovered

For nested-frame/onion tasks, the "deepest column" selector does not need to
materialize a vertical color-diff plane. On task377 v8, I replaced:

```text
Conv(input, color-ramp-diff) -> diff F32[1,1,29,30] -> ReduceL1 -> ArgMax
```

with one ordered four-operand Einsum:

```text
Einsum("bchw,hg,bdgw,cd->bw", input, adj_next[30,30], input, neq_cd[10,10])
```

This computes the count of adjacent active row-pairs whose colors differ, per
column. The ordered operands are important for ORT's left-deep contraction:
`input x adj_next` first keeps the prefix at `[B,C,W,G]`, then the second input
and the small color-mismatch matrix collapse to `[B,W]`.

## Where

- Task: task377
- Builder: `The_400_Tasks/Task_by_Task/task377/prototypes/build_task377_rewrite_v8_einsum_transition.py`
- Staged: `staged_submits/task377_v8_einsum_transition_cost5689/`
- Candidate SHA256: `0ac07313ad9e1bd4237e181df3bf0277c9a3a7ee122a3a6c3f8f281639954f3a`

## Why it matters

Measured cost went from the consumed/probe-ok v7 floor 8192 to 5689:

- Removed `diff` F32 `[1,1,29,30]` = 3480B.
- Added `adj_next` 900 params, `neq_cd` 100 params, and `tc` F32 `[1,30]` = 120B.
- Static/profile match: `profile_real_cost --arcgen 266` returned 5689.
- Correctness gates: visible 266/266, truth_full 199746/200000, AB full vs probe-ok v7 200266/200266 byte-equal.

This extends techniques.md #52 and #89: #52 already says TV-argmax equals
transition-count argmax, and #89 covers in-kernel Einsum temporaries, but the
library did not spell out this adjacency/mismatch transition-count formulation
as a replacement for the F32 Conv-diff carrier.

## Sorter note: techniques.md #52 / #89 - extend existing

Extend the task377 onion technique with the ordered
`bchw,hg,bdgw,cd->bw` transition-count selector and measured v8 result
8192 -> 5689. Also route from by_symptom nested-frame/onion if desired.

## Dynamic selected-column read without `col_oh`

After v8, the next largest avoidable tensor was `col_oh` F32 `[1,10,30]`.
I replaced the dynamic `Gather(input, cstar, axis=3)` with:

```text
OneHot(cstar1) -> csel F32[1,30]
Einsum("bchw,qw,c->bh", input, csel, ramp[10]) -> colcolor_f F32[1,30]
Einsum("bchw,qw->bh", input, csel) -> active_f F32[1,30]
```

This keeps the selected column read as two 1-D outputs instead of a full
channel-by-row slice. On task377 it moved cost 5689 -> 4554 while preserving
visible 266/266 and AB 2266/2266 against v8 before v10 superseded it.

## Sorter note: techniques.md #52 / #89 / #42 - extend existing

Add a small note that dynamic scalar selection can sometimes be cheaper as
`OneHot(index)` + input-Einsum contractions than as a high-channel Gather,
when consumers only need scalar row values and activity rather than the full
slice.

## Small BOOL one-hot then Pad-as-output

The output-fold used by task377 v6/v7/v8/v9 paid a full U8 value canvas:

```text
v_total[9,9] -> Unsqueeze[1,1,9,9] -> Pad U8[1,1,30,30] -> Equal == output
```

At opset14, BOOL Pad works, so for a max 9x9 rendered bullseye the cheaper tail
is:

```text
Equal(v232u[1,1,9,9], ramp[1,10,1,1]) -> small_oh BOOL[1,10,9,9]
Pad(small_oh) == output
```

This changed task377 v9 4554 -> v10 4464. It is the #81 small-grid inversion
applied after the v6 output-fold: below 90 active cells, the small decoded
one-hot can beat the fixed 900B padded value canvas when the Pad can be the
graph output.

Final staged candidate: `staged_submits/task377_v10_einsum_column_smallbool_cost4464/`,
SHA `22c23324671a33896a1b1c43583d7956f4234afe7544450505c672191c44751f`, profile
4464, full AB vs probe-ok v7 200266/200266.

## Sorter note: techniques.md #81 - extend existing

Extend the task377 output-fold corollary with the opset14 BOOL-Pad small-grid
counterpart: if `10*h*w < 30*30` and Pad can be the output, build the small
BOOL one-hot first and Pad it, rather than padding a single-channel value canvas
then making Equal the output.
