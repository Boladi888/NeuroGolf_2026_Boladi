# task249 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task249.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 67
- **Initializers**: 31

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 16 |
| `Cast` | 16 |
| `Add` | 15 |
| `Equal` | 15 |
| `ReduceSum` | 1 |
| `ReduceMax` | 1 |
| `ArgMax` | 1 |
| `Reshape` | 1 |
| `Gather` | 1 |

## Initializer names (first 30)

```
ht2_idx_w1
ht2_idx_w2
ht2_idx_w3
ht2_idx_w4
ht2_idx_w5
ht2_idx_w6
ht2_idx_w7
ht2_idx_w8
ht2_idx_w9
ht2_idx_w10
ht2_idx_w11
ht2_idx_w12
ht2_idx_w13
ht2_idx_w14
ht2_idx_w15
ht2_asc_col
ht2_shape1
ht2_Wc2
ht2_Wc3
ht2_Wc4
ht2_Wc5
ht2_Wc6
ht2_Wc7
ht2_Wc8
ht2_Wc9
ht2_Wc10
ht2_Wc11
ht2_Wc12
ht2_Wc13
ht2_Wc14
```

## First 12 nodes (algorithm entry point)

```
    0: ReduceSum      {'axes': [1], 'keepdims': 1} (input) -> ht2_is_active
    1: ReduceMax      {'axes': [2], 'keepdims': 1} (ht2_is_active) -> ht2_col_active
    2: Mul            (ht2_col_active, ht2_asc_col) -> ht2_col_weighted
    3: ArgMax         {'axis': 3, 'keepdims': 0} (ht2_col_weighted) -> ht2_last_col_3d
    4: Reshape        (ht2_last_col_3d, ht2_shape1) -> ht2_last_col
    5: Add            (ht2_last_col, ht2_shape1) -> ht2_W_val
    6: Equal          (ht2_W_val, ht2_shape1) -> ht2_isW1
    7: Cast           {'to': 1} (ht2_isW1) -> ht2_isW1f
    8: Mul            (ht2_idx_w1, ht2_isW1f) -> ht2_cW1
    9: Equal          (ht2_W_val, ht2_Wc2) -> ht2_isW2
   10: Cast           {'to': 1} (ht2_isW2) -> ht2_isW2f
   11: Mul            (ht2_idx_w2, ht2_isW2f) -> ht2_cW2
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 249
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
