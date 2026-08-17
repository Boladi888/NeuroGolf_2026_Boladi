# task004 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task004.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 12
- **Nodes**: 19
- **Initializers**: 10

## Op frequency

| Op | Count |
|---|---:|
| `ReduceMax` | 5 |
| `Mul` | 3 |
| `Max` | 2 |
| `Sub` | 2 |
| `Slice` | 1 |
| `Gather` | 1 |
| `Equal` | 1 |
| `Where` | 1 |
| `Conv` | 1 |
| `Concat` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
v0
v1
v2
v3
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
    1: ReduceMax      {'axes': [1], 'keepdims': 1} (input_inner) -> v4
    2: Gather         {'axis': 1} (input_inner, v3) -> v5
    3: Mul            (v5, v1) -> v6
    4: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (v6) -> v7
    5: Equal          (v1, v7) -> v8
    6: Where          (v8, v5, pad_zero) -> v10
    7: ReduceMax      {'axes': [3], 'keepdims': 1} (v10) -> v11
    8: ReduceMax      {'axes': [2], 'keepdims': 1} (v10) -> v12
    9: Max            (v11, v12) -> v13
   10: Mul            (v5, v13) -> v14
   11: Sub            (v0, v14) -> v15
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 4
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
