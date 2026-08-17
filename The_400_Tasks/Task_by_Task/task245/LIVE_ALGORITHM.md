# task245 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task245.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task245`
- **Opset**: 10
- **Nodes**: 43
- **Initializers**: 16

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 7 |
| `Reshape` | 6 |
| `Sub` | 5 |
| `ReduceMax` | 4 |
| `ArgMax` | 4 |
| `Slice` | 2 |
| `Add` | 2 |
| `Greater` | 2 |
| `Less` | 2 |
| `And` | 2 |
| `Cast` | 2 |
| `Where` | 2 |
| `Gather` | 2 |
| `Sum` | 1 |

## Initializer names (first 30)

```
offsets30
one_i
neg_one
zero_i_30
one_f
ch2_s
ch2_e
ch3_e
axis1
shape_30
thirty_i
shape_1_1_30_1
shape_1_1_1_30
keep_vec
ch0_add_vec
ch2_vec
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, ch2_s, ch2_e, axis1) -> ch2
    1: Slice          (input, ch2_e, ch3_e, axis1) -> ch3
    2: ReduceMax      {'axes': [3], 'keepdims': 1} (ch2) -> row_any_2_4d
    3: Reshape        (row_any_2_4d, shape_30) -> row_any_2
    4: ReduceMax      {'axes': [2], 'keepdims': 1} (ch2) -> col_any_2_4d
    5: Reshape        (col_any_2_4d, shape_30) -> col_any_2
    6: ReduceMax      {'axes': [3], 'keepdims': 1} (ch3) -> row_any_3_4d
    7: Reshape        (row_any_3_4d, shape_30) -> row_any_3
    8: ReduceMax      {'axes': [2], 'keepdims': 1} (ch3) -> col_any_3_4d
    9: Reshape        (col_any_3_4d, shape_30) -> col_any_3
   10: ArgMax         {'axis': 0, 'keepdims': 1} (row_any_2) -> y0_2
   11: ArgMax         {'axis': 0, 'keepdims': 1} (col_any_2) -> x0_2
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 245
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
