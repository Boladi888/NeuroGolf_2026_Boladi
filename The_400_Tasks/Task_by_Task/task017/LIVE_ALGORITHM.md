# task017 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task017.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 136
- **Initializers**: 29

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 39 |
| `Mul` | 29 |
| `ReduceSum` | 27 |
| `Sub` | 11 |
| `Cast` | 10 |
| `Less` | 9 |
| `Concat` | 2 |
| `Reshape` | 2 |
| `ArgMax` | 1 |
| `Gather` | 1 |
| `Conv` | 1 |
| `Greater` | 1 |
| `Add` | 1 |
| `ReduceMax` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
phf_one
phf_half
phf_hw_st
phf_hw_en
phf_hw_ax
phf_step2
phf_ch_ax
phf_ch0_st
phf_fg_en
phf_kernel_bank
phf_kernel_shape
phf_p1_left_en
phf_p1_right_en
phf_p1_col_ax
phf_p2_left_en
phf_p2_right_st
phf_p3_left_en
phf_p4_left_en
phf_p4_right_st
phf_p5_left_en
phf_p5_right_st
phf_p6_left_en
phf_p6_right_st
phf_p7_left_en
phf_p7_right_st
phf_p8_left_en
phf_p8_right_st
phf_p9_left_en
phf_p9_right_st
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, phf_hw_st, phf_hw_en, phf_hw_ax, phf_step2) -> phf_x
    1: Slice          (phf_x, phf_ch0_st, phf_ch_ax, phf_ch_ax, phf_ch_ax) -> phf_ch0
    2: Slice          (phf_x, phf_ch_ax, phf_fg_en, phf_ch_ax, phf_ch_ax) -> phf_fg
    3: Sub            (phf_one, phf_ch0) -> phf_nonzero
    4: Slice          (phf_fg, phf_ch0_st, phf_p1_left_en, phf_p1_col_ax, phf_ch_ax) -> phf_p1_fg_left
    5: Slice          (phf_fg, phf_ch_ax, phf_p1_right_en, phf_p1_col_ax, phf_ch_ax) -> phf_p1_fg_right
    6: Mul            (phf_p1_fg_left, phf_p1_fg_right) -> phf_p1_same_by_ch
    7: ReduceSum      {'axes': [1], 'keepdims': 1} (phf_p1_same_by_ch) -> phf_p1_same
    8: Slice          (phf_nonzero, phf_ch0_st, phf_p1_left_en, phf_p1_col_ax, phf_) -> phf_p1_nz_left
    9: Slice          (phf_nonzero, phf_ch_ax, phf_p1_right_en, phf_p1_col_ax, phf_) -> phf_p1_nz_right
   10: Mul            (phf_p1_nz_left, phf_p1_nz_right) -> phf_p1_both
   11: Sub            (phf_p1_both, phf_p1_same) -> phf_p1_conflict_cells
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 17
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
