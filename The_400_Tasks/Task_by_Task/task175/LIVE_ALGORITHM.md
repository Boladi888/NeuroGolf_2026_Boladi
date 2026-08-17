# task175 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task175.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task175_huge_static`
- **Opset**: 17
- **Nodes**: 392
- **Initializers**: 124

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 150 |
| `Mul` | 120 |
| `Add` | 60 |
| `Concat` | 31 |
| `Transpose` | 30 |
| `Identity` | 1 |

## Initializer names (first 30)

```
nz_mask
zero_px
zero_row
axes4
row_0_starts
row_0_ends
col_0_ends
row_0_zero_ends
diag_0_ends
row_1_starts
row_1_ends
col_1_starts
col_1_ends
row_2_starts
row_2_ends
col_2_starts
col_2_ends
row_3_starts
row_3_ends
col_3_starts
col_3_ends
row_4_starts
row_4_ends
col_4_starts
col_4_ends
row_5_starts
row_5_ends
col_5_starts
col_5_ends
row_6_starts
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, row_0_starts, row_0_ends, axes4) -> row_0
    1: Slice          (input, row_0_starts, col_0_ends, axes4) -> col_0_raw
    2: Transpose      {'perm': [0, 1, 3, 2]} (col_0_raw) -> col_0
    3: Mul            (row_0, nz_mask) -> row_0_nz
    4: Mul            (col_0, nz_mask) -> col_0_nz
    5: Slice          (row_0, row_0_starts, row_0_zero_ends, axes4) -> row_0_zero
    6: Slice          (col_0, row_0_starts, row_0_zero_ends, axes4) -> col_0_zero
    7: Mul            (row_0_zero, col_0_zero) -> both_zero_0
    8: Slice          (zero_row, row_0_starts, diag_0_ends, axes4) -> diag_0_tail
    9: Concat         {'axis': 3} (zero_px, diag_0_tail) -> diag_0
   10: Mul            (both_zero_0, diag_0) -> carry_0
   11: Add            (row_0_nz, col_0_nz) -> base_0
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 175
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
