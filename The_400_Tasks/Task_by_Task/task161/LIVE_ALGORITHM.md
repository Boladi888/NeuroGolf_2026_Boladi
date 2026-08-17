# task161 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task161.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `neurogolf-v2-task161-boundary-pair-lines`
- **Graph name**: `task161_boundary_pair_lines`
- **Opset**: 16
- **Nodes**: 49
- **Initializers**: 17

## Algorithm hints (from naming)

- boundary-touching condition

## Op frequency

| Op | Count |
|---|---:|
| `And` | 11 |
| `ReduceSum` | 9 |
| `Greater` | 9 |
| `Cast` | 9 |
| `Equal` | 3 |
| `Slice` | 2 |
| `CumSum` | 2 |
| `Or` | 1 |
| `Not` | 1 |
| `Concat` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
crop_starts
crop_ends
crop_axes
crop_steps
pad_to_external
pad_zero
one
four
axes_c
axes_hw
axes_h
axes_w
axis_h_scalar
axis_w_scalar
nonbg_ends
first_col_bool
first_row_bool
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, crop_starts, crop_ends, crop_axes, crop_steps) -> input_inner
    1: ReduceSum      {'keepdims': 1} (input_inner, axes_c) -> valid_count
    2: Greater        (valid_count, pad_zero) -> valid_bool
    3: Cast           {'to': 1} (valid_bool) -> valid_f
    4: Slice          (input_inner, axes_c, nonbg_ends, axes_c, axes_c) -> nonbg
    5: Greater        (nonbg, pad_zero) -> nonbg_bool
    6: ReduceSum      {'keepdims': 1} (nonbg, axes_hw) -> color_count
    7: Equal          (color_count, four) -> count4_bool
    8: ReduceSum      {'keepdims': 1} (valid_f, axes_h) -> col_valid_count
    9: Greater        (col_valid_count, pad_zero) -> col_valid_bool
   10: Cast           {'to': 1} (col_valid_bool) -> col_valid_f
   11: CumSum         {'reverse': 1} (col_valid_f, axis_w_scalar) -> col_valid_rev
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 161
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
