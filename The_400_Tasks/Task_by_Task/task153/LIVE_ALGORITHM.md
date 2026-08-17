# task153 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task153.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 259
- **Initializers**: 29

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 73 |
| `Pad` | 73 |
| `Gather` | 45 |
| `Cast` | 24 |
| `Add` | 20 |
| `Reshape` | 6 |
| `Mul` | 5 |
| `ReduceSum` | 3 |
| `ReduceMax` | 2 |
| `ArgMax` | 2 |
| `Equal` | 2 |
| `Concat` | 1 |
| `Transpose` | 1 |
| `MatMul` | 1 |
| `Conv` | 1 |

## Initializer names (first 30)

```
tbt_fg_starts
tbt_fg_ends
tbt_desc_row
tbt_desc_col
tbt_rs9
tbt_range3
tbt_color_idx_0
tbt_color_idx_2
tbt_color_idx_3
tbt_color_idx_4
tbt_color_idx_5
tbt_color_idx_6
tbt_color_idx_7
tbt_color_idx_8
tbt_place_axes
tbt_pl_0_1_starts
tbt_pl_0_1_ends
tbt_pl_0_1_steps
tbt_pl_0_2_ends
tbt_pl_1_1_ends
tbt_pl_1_2_ends
tbt_pl_2_0_ends
tbt_pl_2_1_ends
tbt_rs_flat
tbt_rs_81_1
tbt_rs_1_81
tbt_upper
tbt_rs_sel
tbt_conv_w
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, tbt_fg_starts, tbt_fg_ends, tbt_fg_starts, tbt_fg_sta) -> tbt_fg9
    1: ReduceMax      {'axes': [3], 'keepdims': 1} (tbt_fg9) -> tbt_row_has
    2: ReduceMax      {'axes': [2], 'keepdims': 1} (tbt_fg9) -> tbt_col_has
    3: Mul            (tbt_row_has, tbt_desc_row) -> tbt_w_first_row
    4: Mul            (tbt_col_has, tbt_desc_col) -> tbt_w_first_col
    5: ArgMax         {'axis': 2, 'keepdims': 0} (tbt_w_first_row) -> tbt_first_row_3d
    6: ArgMax         {'axis': 3, 'keepdims': 0} (tbt_w_first_col) -> tbt_first_col_3d
    7: Reshape        (tbt_first_row_3d, tbt_rs9) -> tbt_first_row_i
    8: Reshape        (tbt_first_col_3d, tbt_rs9) -> tbt_first_col_i
    9: Cast           {'to': 1} (tbt_first_row_i) -> tbt_first_row_f
   10: Cast           {'to': 1} (tbt_first_col_i) -> tbt_first_col_f
   11: Gather         {'axis': 0} (tbt_first_row_f, tbt_color_idx_0) -> tbt_c0_r0_f
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 153
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
