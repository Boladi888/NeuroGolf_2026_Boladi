# task071 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task071.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task071`
- **Opset**: 10
- **Nodes**: 48
- **Initializers**: 12

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 9 |
| `Sub` | 8 |
| `ArgMax` | 6 |
| `ReduceMax` | 5 |
| `Add` | 4 |
| `Cast` | 4 |
| `ReduceSum` | 3 |
| `Slice` | 3 |
| `Equal` | 2 |
| `Greater` | 1 |
| `Reshape` | 1 |
| `MatMul` | 1 |
| `Max` | 1 |

## Initializer names (first 30)

```
ch_mask
rev_starts
rev_ends
rev_axes_r
rev_axes_c
c29_i64
c1_i64_s
zero_f
one_f
shape_empty
ch0_sel
ipj
```

## First 12 nodes (algorithm entry point)

```
    0: Mul            (input, ch_mask) -> input_masked
    1: ReduceSum      {'axes': [2, 3], 'keepdims': 1} (input_masked) -> count_c
    2: ReduceMax      {'axes': [3], 'keepdims': 1} (input_masked) -> rows_ne
    3: ReduceMax      {'axes': [2], 'keepdims': 1} (input_masked) -> cols_ne
    4: ArgMax         {'axis': 2, 'keepdims': 1} (rows_ne) -> r0_i64
    5: ArgMax         {'axis': 3, 'keepdims': 1} (cols_ne) -> c0_i64
    6: Slice          (rows_ne, rev_starts, rev_ends, rev_axes_r, rev_starts) -> rows_ne_rev
    7: ArgMax         {'axis': 2, 'keepdims': 1} (rows_ne_rev) -> r1_rev_i64
    8: Slice          (cols_ne, rev_starts, rev_ends, rev_axes_c, rev_starts) -> cols_ne_rev
    9: ArgMax         {'axis': 3, 'keepdims': 1} (cols_ne_rev) -> c1_rev_i64
   10: Sub            (c29_i64, r1_rev_i64) -> r1_i64
   11: Sub            (c29_i64, c1_rev_i64) -> c1_i64
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 71
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
