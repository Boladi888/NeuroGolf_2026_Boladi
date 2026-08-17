# task316 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task316.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task316_column_snake_pack`
- **Opset**: 12
- **Nodes**: 49
- **Initializers**: 15

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 11 |
| `ReduceSum` | 11 |
| `Equal` | 9 |
| `Where` | 9 |
| `Concat` | 4 |
| `ReduceMax` | 1 |
| `CumSum` | 1 |
| `Sub` | 1 |
| `Add` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
nonzero_gate
axis_col
rank_0
rank_1
rank_2
rank_3
rank_4
rank_5
rank_6
rank_7
rank_8
canvas3
zero_channel
pads
pad_value
```

## First 12 nodes (algorithm entry point)

```
    0: Mul            (input, nonzero_gate) -> colored_out
    1: ReduceMax      {'axes': [2], 'keepdims': 1} (colored_out) -> col_color
    2: ReduceSum      {'axes': [1], 'keepdims': 1} (col_color) -> col_occupied
    3: CumSum         (col_occupied, axis_col) -> col_rank1
    4: Equal          (col_rank1, rank_0) -> rank_eq_0
    5: Where          (rank_eq_0, col_occupied, pad_value) -> rank_mask_0_out
    6: Mul            (col_color, rank_mask_0_out) -> picked_cols_0_out
    7: ReduceSum      {'axes': [3], 'keepdims': 1} (picked_cols_0_out) -> picked_color_0
    8: Equal          (col_rank1, rank_1) -> rank_eq_1
    9: Where          (rank_eq_1, col_occupied, pad_value) -> rank_mask_1_out
   10: Mul            (col_color, rank_mask_1_out) -> picked_cols_1_out
   11: ReduceSum      {'axes': [3], 'keepdims': 1} (picked_cols_1_out) -> picked_color_1
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 316
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
