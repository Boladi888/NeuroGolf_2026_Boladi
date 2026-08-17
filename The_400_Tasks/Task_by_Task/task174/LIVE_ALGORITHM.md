# task174 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task174.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 826
- **Initializers**: 22

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 145 |
| `Cast` | 127 |
| `Gather` | 109 |
| `Sub` | 74 |
| `Less` | 64 |
| `Unsqueeze` | 54 |
| `Equal` | 45 |
| `ArgMax` | 36 |
| `Squeeze` | 36 |
| `ReduceSum` | 36 |
| `ReduceMax` | 19 |
| `Add` | 19 |
| `Greater` | 18 |
| `Reshape` | 18 |
| `Sum` | 11 |

## Initializer names (first 30)

```
v0
v1
v2
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
v16
v17
v18
v20
v23
v24
v25
v26
v28
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, v5, v6, v7) -> v29
    1: Gather         {'axis': 1} (v29, v8) -> v30
    2: Split          {'axis': 1} (v30) -> v31
    3: ReduceMax      {'axes': [3], 'keepdims': 0} (v31) -> v40
    4: Greater        (v40, v4) -> v41
    5: Cast           {'to': 1} (v41) -> v42
    6: Reshape        (v42, v12) -> v43
    7: Unsqueeze      {'axes': [0, 1]} (v43) -> v44
    8: ArgMax         {'axis': 2, 'keepdims': 0} (v44) -> v45
    9: Squeeze        {'axes': [0, 1]} (v45) -> v46
   10: Add            (v46, v9) -> v47
   11: Gather         {'axis': 2} (v44, v10) -> v48
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 174
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
