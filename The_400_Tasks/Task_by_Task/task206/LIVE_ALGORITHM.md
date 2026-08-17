# task206 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task206.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 223
- **Initializers**: 33

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 51 |
| `Slice` | 41 |
| `Pad` | 41 |
| `ReduceSum` | 37 |
| `Where` | 36 |
| `Gather` | 5 |
| `Sum` | 5 |
| `Conv` | 3 |
| `Sub` | 2 |
| `Less` | 1 |
| `Concat` | 1 |

## Initializer names (first 30)

```
v3
v4
v5
v6
v7
v8
v9
v10
v11
v12
v13
v14
v15
v16
v29
v30
v32
v33
v47
v48
v17
v18
v92
v93
v107
v108
v122
v123
v137
v138
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, v3, v4, v5) -> v149
    1: Gather         {'axis': 1} (v149, v6) -> v167
    2: Gather         {'axis': 1} (v149, v7) -> v172
    3: Gather         {'axis': 1} (v149, v8) -> v177
    4: Gather         {'axis': 1} (v149, v9) -> v182
    5: Gather         {'axis': 1} (v149, v10) -> v256
    6: Conv           {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0]} (v149, v11) -> v155
    7: Conv           {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0]} (v149, v12) -> v156
    8: ReduceSum      {'axes': [1, 2, 3], 'keepdims': 1} (v156) -> v157
    9: Conv           {'kernel_shape': [3, 3], 'pads': [0, 0, 2, 2]} (v156, v13) -> v158
   10: Pad            {'pads': [0, 0, 0, 0, 0, 0, 1, 1]} (v256) -> v163
   11: Pad            {'pads': [0, 0, 0, 0, 0, 0, 1, 0]} (v256) -> v186
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 206
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
