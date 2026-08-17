# task031 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task031.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `neurogolf-v2-task031_symbolic_nonzero_bbox_crop`
- **Graph name**: `task031_symbolic_nonzero_bbox_crop`
- **Opset**: 16
- **Nodes**: 28
- **Initializers**: 13

## Op frequency

| Op | Count |
|---|---:|
| `Cast` | 4 |
| `ArgMax` | 4 |
| `ReduceSum` | 2 |
| `Greater` | 2 |
| `Where` | 2 |
| `Sub` | 2 |
| `Add` | 2 |
| `Gather` | 2 |
| `LessOrEqual` | 2 |
| `Unsqueeze` | 2 |
| `Mul` | 2 |
| `Slice` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
starts
ends
axes
steps
row_reduce_axes
col_reduce_axes
zero
coord_weight
row_offsets
col_offsets
row_unsqueeze
col_unsqueeze
pads
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, starts, ends, axes, steps) -> nonzero_channels
    1: ReduceSum      {'keepdims': 0} (nonzero_channels, row_reduce_axes) -> row_counts
    2: ReduceSum      {'keepdims': 0} (nonzero_channels, col_reduce_axes) -> col_counts
    3: Greater        (row_counts, zero) -> row_present_bool
    4: Greater        (col_counts, zero) -> col_present_bool
    5: Cast           {'to': 1} (row_present_bool) -> row_present
    6: Cast           {'to': 1} (col_present_bool) -> col_present
    7: ArgMax         {'axis': 1, 'keepdims': 0} (row_present) -> first_row
    8: ArgMax         {'axis': 1, 'keepdims': 0} (col_present) -> first_col
    9: Where          (row_present_bool, coord_weight, zero) -> row_weighted
   10: Where          (col_present_bool, coord_weight, zero) -> col_weighted
   11: ArgMax         {'axis': 1, 'keepdims': 0} (row_weighted) -> last_row
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 31
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
