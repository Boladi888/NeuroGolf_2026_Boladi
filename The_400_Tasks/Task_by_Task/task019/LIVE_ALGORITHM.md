# task019 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task019.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 146
- **Initializers**: 35

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 33 |
| `Cast` | 33 |
| `Add` | 31 |
| `Equal` | 30 |
| `Slice` | 5 |
| `ReduceSum` | 2 |
| `ReduceMax` | 2 |
| `ArgMax` | 2 |
| `Reshape` | 2 |
| `Gather` | 2 |
| `Conv` | 1 |
| `Greater` | 1 |
| `Sub` | 1 |
| `Concat` | 1 |

## Initializer names (first 30)

```
tdnm_ridx1
tdnm_ridx2
tdnm_ridx3
tdnm_ridx4
tdnm_ridx5
tdnm_ridx6
tdnm_ridx7
tdnm_ridx8
tdnm_ridx9
tdnm_ridx10
tdnm_ridx11
tdnm_ridx12
tdnm_ridx13
tdnm_ridx14
tdnm_ridx15
tdnm_ch0_s
tdnm_ch1_s
tdnm_ch8_s
tdnm_ch8e
tdnm_ch9_e
tdnm_asc_row
tdnm_asc_col
tdnm_diag_w
tdnm_half
tdnm_Hc2
tdnm_Hc3
tdnm_Hc4
tdnm_Hc5
tdnm_Hc6
tdnm_Hc7
```

## First 12 nodes (algorithm entry point)

```
    0: ReduceSum      {'axes': [1], 'keepdims': 1} (input) -> tdnm_is_active
    1: ReduceMax      {'axes': [3], 'keepdims': 1} (tdnm_is_active) -> tdnm_row_active
    2: ReduceMax      {'axes': [2], 'keepdims': 1} (tdnm_is_active) -> tdnm_col_active
    3: Mul            (tdnm_row_active, tdnm_asc_row) -> tdnm_row_w
    4: ArgMax         {'axis': 2, 'keepdims': 0} (tdnm_row_w) -> tdnm_last_row_3d
    5: Reshape        (tdnm_last_row_3d, tdnm_ch1_s) -> tdnm_last_row
    6: Add            (tdnm_last_row, tdnm_ch1_s) -> tdnm_H_val
    7: Mul            (tdnm_col_active, tdnm_asc_col) -> tdnm_col_w
    8: ArgMax         {'axis': 3, 'keepdims': 0} (tdnm_col_w) -> tdnm_last_col_3d
    9: Reshape        (tdnm_last_col_3d, tdnm_ch1_s) -> tdnm_last_col
   10: Add            (tdnm_last_col, tdnm_ch1_s) -> tdnm_W_val
   11: Equal          (tdnm_H_val, tdnm_ch1_s) -> tdnm_isH1
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 19
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
