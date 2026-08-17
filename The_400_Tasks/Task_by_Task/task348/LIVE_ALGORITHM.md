# task348 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task348.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 11
- **Nodes**: 22
- **Initializers**: 14

## Op frequency

| Op | Count |
|---|---:|
| `Sub` | 4 |
| `ReduceSum` | 3 |
| `Cast` | 3 |
| `Slice` | 2 |
| `Mul` | 2 |
| `ArgMax` | 1 |
| `Abs` | 1 |
| `Add` | 1 |
| `Greater` | 1 |
| `Where` | 1 |
| `Mod` | 1 |
| `Concat` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
ch7s
ch7e
ax1
col_idx
row_idx
one_f
two_i32
zero_ch
crop_starts
crop_ends
crop_axes
crop_steps
pad_to_external
pad_zero
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, crop_starts, crop_ends, crop_axes, crop_steps) -> input_inner
    1: ReduceSum      {'axes': [1], 'keepdims': 1} (input_inner) -> inside_grid
    2: Slice          (input_inner, ch7s, ch7e, ax1) -> ch7
    3: ReduceSum      {'axes': [1, 2, 3], 'keepdims': 1} (ch7) -> L_val
    4: ReduceSum      {'axes': [2], 'keepdims': 1} (ch7) -> col_sum7
    5: ArgMax         {'axis': 3, 'keepdims': 1} (col_sum7) -> c0_i64
    6: Cast           {'to': 1} (c0_i64) -> c0_f
    7: Sub            (col_idx, c0_f) -> col_diff
    8: Abs            (col_diff) -> col_diff_abs
    9: Sub            (L_val, one_f) -> L_minus_1
   10: Add            (row_idx, col_diff_abs) -> sum_rc
   11: Greater        (sum_rc, L_minus_1) -> out_tri_bool
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 348
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
