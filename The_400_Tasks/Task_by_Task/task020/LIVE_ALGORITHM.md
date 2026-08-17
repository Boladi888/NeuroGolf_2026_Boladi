# task020 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task020.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 104
- **Initializers**: 18

## Op frequency

| Op | Count |
|---|---:|
| `Cast` | 14 |
| `Mul` | 12 |
| `Reshape` | 10 |
| `Less` | 10 |
| `Max` | 8 |
| `Sub` | 7 |
| `Gather` | 6 |
| `Add` | 5 |
| `Or` | 5 |
| `Not` | 5 |
| `Min` | 5 |
| `ArgMax` | 4 |
| `Slice` | 3 |
| `ReduceMax` | 3 |
| `Div` | 2 |

## Initializer names (first 30)

```
rsc_cr_starts
rsc_cr_ends
rsc_cr_axes
rsc_cr_steps
rsc_sl_starts
rsc_sl_ends
rsc_desc_row
rsc_asc_row
rsc_desc_col
rsc_asc_col
rsc_const_2_f
rsc_range_10_f
rsc_const_9_f
rsc_const_0_f
rsc_const_1_4d
rsc_rs_row_4d
rsc_rs_col_4d
rsc_nz_ends
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, rsc_cr_starts, rsc_cr_ends, rsc_cr_axes, rsc_cr_steps) -> rsc_crop
    1: ReduceSum      {'axes': [1], 'keepdims': 1} (rsc_crop) -> rsc_is_active
    2: Slice          (rsc_crop, rsc_sl_starts, rsc_sl_ends, rsc_sl_ends, rsc_sl_en) -> rsc_ch0
    3: Sub            (rsc_is_active, rsc_ch0) -> rsc_fg
    4: ReduceMax      {'axes': [3], 'keepdims': 1} (rsc_fg) -> rsc_row_has
    5: ReduceMax      {'axes': [2], 'keepdims': 1} (rsc_fg) -> rsc_col_has
    6: Mul            (rsc_row_has, rsc_desc_row) -> rsc_w_first_row
    7: Mul            (rsc_row_has, rsc_asc_row) -> rsc_w_last_row
    8: Mul            (rsc_col_has, rsc_desc_col) -> rsc_w_first_col
    9: Mul            (rsc_col_has, rsc_asc_col) -> rsc_w_last_col
   10: ArgMax         {'axis': 2, 'keepdims': 0} (rsc_w_first_row) -> rsc_first_row_3d
   11: ArgMax         {'axis': 2, 'keepdims': 0} (rsc_w_last_row) -> rsc_last_row_3d
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 20
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
