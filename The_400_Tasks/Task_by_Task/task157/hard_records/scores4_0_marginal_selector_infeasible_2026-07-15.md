# task157 scores4_0 marginal-selector infeasibility

Worker: `bo_ng_worker1`  
Date: 2026-07-15  
Route: reviewer-locked `scores4_0` marginal/Einsum selector replacement

## Bar

Authoritative bar rechecked with:

```powershell
python tools/ng_current_bar.py --task 157 --json
```

Result: manifest/team-best Vamshi cost/profile `4039`, SHA-256
`14f7ecf0fa38c088cdaeed1d5f142bcf04082d4d54600159f2e7e05fa3a7cc6a`.
The `team_best` source reports stale cost `3982` for the same SHA; manifest/workbook
agree on `4039`, so `4039` is the bar used.

Current live profile rechecked:

```powershell
python tools/agent_helpers/profile_real_cost.py --task 157 --cand Team_Best_Onnx/task157.onnx --arcgen 266
```

Result line:

```text
task157 task157.onnx: static=4039 profile=4039 (memory 3847 + params 192; 265 real inputs)
```

## Locked Hypothesis Tested

Replace Vamshi's first matcher score plane:

```text
scores4_0: UINT8 [1,4,4,15]
producer: QLinearConv(rem0_padded, compw_u), pads=[1,0,3,0]
```

with row/column/color marginals or a separable Einsum selector that still yields
the same:

```text
compscore_0 -> bcomp0
rowpick_sq_0 -> brow0
scorerow0 -> bcol0_i64
```

This follows `Onnx_Advanced/techniques.md` #89/#109 and the marginal-family
route in `by_symptom.md`: use separable/marginal reductions only when the
required statistic is separable and does not require row-column association.

## Fresh Check Attempt

I attempted the requested fresh ARC-GEN/numpy separability script once:

```powershell
python The_400_Tasks/Task_by_Task/task157/prototypes/task157_scores4_0_separability_check.py
```

Exact tool result:

```text
Exit code: 124
Wall time: 124.4 seconds
Output:
command timed out after 124398 milliseconds
```

The timeout left PID `25940` running
`The_400_Tasks/Task_by_Task/task157/prototypes/task157_scores4_0_separability_check.py`;
I stopped only that process. No JSON counterexample artifact was written by the
fresh run.

## Counterexample Certificate

Static generator inspection:

```text
external/ARC-GEN/tasks/task_6a1e5592.py
```

uses `common.continuous_creature(size, wide, tall)` with widths `1..4`, heights
`2..4`, connected shapes, and local `(0,0)` included. The following two
generator-legal connected 4x2 shapes both include `(0,0)` and satisfy the
generator size bounds.

Source shape A cells:

```text
(0,0), (0,1), (0,2), (1,0), (1,2), (1,3)
```

Source shape B cells:

```text
(0,0), (0,2), (0,3), (1,0), (1,1), (1,2)
```

When placed by the generator as gray shapes at rows 8..9 and then viewed in
Vamshi's `five_f_b = input[channel 5, rows 6:10, cols 0:15]` component kernel
frame, their 4x4 `compw_u` masks are:

```text
A =
0000
0000
1110
1011

B =
0000
0000
1011
1110
```

They have identical row, column, and color-count marginals:

```text
row sums = [0, 0, 3, 3]
col sums = [2, 1, 2, 1]
color count = 6
```

Now place source shape A in the top red bar at rows 1..2. For `scores4_0`
output row `brow=0`, the exact 4x4 `rem0_padded` window seen by the convolution
is:

```text
W_A =
0000
0000
1110
1011
```

The Vamshi score algebra is fixed by the graph:

```text
rem0_score_u8 = rem0_u * 129
QLinearConv x_zero_point = 128
QLinearConv w_zero_point = 0
QLinearConv y_zero_point = 128
q_scale = 1.0

score(C, W) = clip_uint8(128 + hits(C,W) - 128 * misses(C,W))
```

For the same window `W_A`:

```text
score(A, W_A): hits=6, misses=0 -> 134
score(B, W_A): hits=4, misses=2 -> 0
```

If the top window is shape B instead, the comparison flips:

```text
W_B =
0000
0000
1011
1110

score(A, W_B): hits=4, misses=2 -> 0
score(B, W_B): hits=6, misses=0 -> 134
```

Both inputs have the same row/column/color marginals for the competing
components and the same row/column/color marginals for the top window. A
marginal-only or separable selector that has discarded row-column association
therefore sees the same summaries in both cases and must make the same
`bcomp0` decision. The exact `scores4_0` plane makes opposite decisions. This
is the concrete nonseparable per-cell state the reviewer asked for.

## Why The Locked Replacement Is Dead

The first matcher is a 2D cross-correlation between arbitrary connected 4x2/4x4
component masks and a shifted top-hole window. Row sums and column sums preserve
only independent marginals; they do not preserve which row is paired with which
column. Techniques #89/#109 allow separable Einsum reductions when the needed
mask is separable or low-rank in the retained state. Here the decision depends
on the entrywise product `C[r,c] * W[r,c]`, and the counterexample above has
identical marginals but different entrywise products. Reconstructing the exact
comparison requires per-cell association or an equivalent nonseparable state,
which is precisely what the 240-byte `scores4_0` materialization carries.

No ONNX skeleton attempts were made after this certificate, and I did not touch
the final Pad/Equal/QLinearConv tail again.

## Remaining Hypothesis

Within the reviewer-locked route, the `scores4_0` row/column/color marginal
replacement is dead. Any sub-4039 route now has to preserve row-column coupling
in a cheaper representation, or attack a different matcher-side tensor such as
later score planes/component extraction with a value-preserving graph rewrite.
