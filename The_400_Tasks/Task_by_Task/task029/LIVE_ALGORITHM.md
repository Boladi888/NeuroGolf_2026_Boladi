# task029 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task029.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 72
- **Initializers**: 18

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 12 |
| `Cast` | 11 |
| `Reshape` | 10 |
| `Sub` | 8 |
| `Add` | 7 |
| `Less` | 6 |
| `Gather` | 6 |
| `ArgMax` | 5 |
| `ReduceMax` | 2 |
| `Min` | 2 |
| `Slice` | 1 |
| `ReduceSum` | 1 |
| `Where` | 1 |

## Initializer names (first 30)

```
mri_s1_st
mri_s1_en
mri_desc_row
mri_asc_row
mri_desc_col
mri_asc_col
mri_rs_19
mri_one_f
mri_two_f
mri_four_f
mri_half_f
mri_two5_f
mri_sentinel
mri_neg_one_f
mri_range_30_f
mri_const_29_f
mri_rs_row
mri_rs_col
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, mri_s1_st, mri_s1_en, mri_s1_st, mri_s1_st) -> mri_ch1_to_9
    1: ReduceSum      {'axes': [2, 3], 'keepdims': 0} (mri_ch1_to_9) -> mri_counts_f
    2: ReduceMax      {'axes': [3], 'keepdims': 1} (mri_ch1_to_9) -> mri_row_has
    3: ReduceMax      {'axes': [2], 'keepdims': 1} (mri_ch1_to_9) -> mri_col_has
    4: Mul            (mri_row_has, mri_desc_row) -> mri_w_fr
    5: Mul            (mri_row_has, mri_asc_row) -> mri_w_lr
    6: Mul            (mri_col_has, mri_desc_col) -> mri_w_fc
    7: Mul            (mri_col_has, mri_asc_col) -> mri_w_lc
    8: ArgMax         {'axis': 2, 'keepdims': 0} (mri_w_fr) -> mri_per_fr_3d
    9: ArgMax         {'axis': 2, 'keepdims': 0} (mri_w_lr) -> mri_per_lr_3d
   10: ArgMax         {'axis': 3, 'keepdims': 0} (mri_w_fc) -> mri_per_fc_3d
   11: ArgMax         {'axis': 3, 'keepdims': 0} (mri_w_lc) -> mri_per_lc_3d
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 29
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
