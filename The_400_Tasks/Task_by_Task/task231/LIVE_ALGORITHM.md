# task231 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task231.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 45
- **Initializers**: 13

## Op frequency

| Op | Count |
|---|---:|
| `Gather` | 9 |
| `Sub` | 6 |
| `Mul` | 6 |
| `Cast` | 5 |
| `Less` | 5 |
| `Abs` | 4 |
| `ReduceMax` | 4 |
| `Conv` | 1 |
| `ReduceSum` | 1 |
| `Add` | 1 |
| `Sum` | 1 |
| `Where` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
v0
v1
v2
v4
v5
v7
v8
v9
v10
v11
v12
v13
_boolmask_where_zero
```

## First 12 nodes (algorithm entry point)

```
    0: Conv           {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0]} (input, v0) -> v14
    1: Gather         {'axis': 2} (v14, v5) -> v15
    2: Cast           {'to': 7} (v15) -> v16
    3: ReduceSum      {'axes': [0, 1, 2, 3], 'keepdims': 1} (v16) -> v17
    4: Add            (v17, v17) -> v18
    5: Less           (v10, v18) -> v19
    6: Gather         {'axis': 2} (input, v4) -> v21
    7: Gather         {'axis': 3} (v21, v5) -> v22
    8: Gather         {'axis': 3} (v21, v7) -> v23
    9: Gather         {'axis': 3} (v21, v8) -> v24
   10: Gather         {'axis': 3} (v21, v9) -> v25
   11: Sub            (v22, v23) -> v26
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 231
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
