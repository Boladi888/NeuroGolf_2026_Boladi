# task260 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task260.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task260_dynamic_diagonal_rule`
- **Opset**: 13
- **Nodes**: 26
- **Initializers**: 15

## Algorithm hints (from naming)

- diagonal-line / cell-coordinate rule

## Op frequency

| Op | Count |
|---|---:|
| `ReduceMax` | 4 |
| `Where` | 4 |
| `Greater` | 3 |
| `Equal` | 3 |
| `Slice` | 2 |
| `And` | 2 |
| `Or` | 2 |
| `Mul` | 1 |
| `ReduceMin` | 1 |
| `Less` | 1 |
| `Add` | 1 |
| `Sub` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
slice_starts
slice_ends
slice_axes
green_starts
green_ends
channel_axis
pads
diag_grid
color_keep
zero_vec
neg99
pos99
two
half
pad_value
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, slice_starts, slice_ends, slice_axes) -> x10
    1: Slice          (x10, green_starts, green_ends, channel_axis) -> green
    2: Mul            (x10, color_keep) -> colored_ch
    3: ReduceMax      {'axes': [1], 'keepdims': 1} (colored_ch) -> colored
    4: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (colored_ch) -> color_vec
    5: Greater        (colored, half) -> colored_bool
    6: Greater        (green, half) -> green_bool
    7: Where          (colored_bool, diag_grid, neg99) -> diag_colored
    8: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (diag_colored) -> d0
    9: Where          (green_bool, diag_grid, neg99) -> diag_green_hi
   10: Where          (green_bool, diag_grid, pos99) -> diag_green_lo
   11: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (diag_green_hi) -> gmax
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 260
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
