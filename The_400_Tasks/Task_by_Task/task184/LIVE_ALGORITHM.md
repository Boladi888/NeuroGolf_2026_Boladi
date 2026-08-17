# task184 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task184.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `neurogolf-v2-task184-matmul`
- **Graph name**: `task184_room_color_grid_matmul`
- **Opset**: 16
- **Nodes**: 51
- **Initializers**: 14

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 8 |
| `Cast` | 8 |
| `Equal` | 7 |
| `ReduceSum` | 4 |
| `Where` | 4 |
| `Greater` | 3 |
| `Reshape` | 3 |
| `Conv` | 2 |
| `Sub` | 2 |
| `CumSum` | 2 |
| `Concat` | 2 |
| `MatMul` | 2 |
| `Transpose` | 2 |
| `ReduceMax` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
nonzero_color_gate
nonzero_any_axes
row_counts_axes
col_counts_axes
row_active_scalar
row_prev_weight
row_one
row_cumsum_axis
row_rank_2_scalar
row_rank_3_scalar
col_prev_weight
col_cumsum_axis
col_rank_0_shape
output_pads
```

## First 12 nodes (algorithm entry point)

```
    0: Mul            (input, nonzero_color_gate) -> nonzero
    1: ReduceSum      {'keepdims': 1} (nonzero, nonzero_any_axes) -> nonzero_any_out
    2: ReduceSum      {'keepdims': 1} (nonzero_any_out, row_counts_axes) -> row_counts_out
    3: ReduceSum      {'keepdims': 1} (nonzero_any_out, col_counts_axes) -> col_counts_out
    4: Greater        (row_counts_out, row_active_scalar) -> row_active_bool
    5: Cast           {'to': 1} (row_active_bool) -> row_active_float
    6: Greater        (col_counts_out, row_active_scalar) -> col_active_bool
    7: Cast           {'to': 1} (col_active_bool) -> col_active_float
    8: Conv           {'kernel_shape': [2, 1], 'pads': [1, 0, 0, 0]} (row_active_float, row_prev_weight) -> row_prev
    9: Sub            (row_one, row_prev) -> row_prev_inactive
   10: Where          (row_active_bool, row_prev_inactive, row_active_scalar) -> row_start
   11: CumSum         (row_start, row_cumsum_axis) -> row_cumsum
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 184
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
