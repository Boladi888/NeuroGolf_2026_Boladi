# task383 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task383.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 11
- **Nodes**: 84
- **Initializers**: 29

## Op frequency

| Op | Count |
|---|---:|
| `Greater` | 13 |
| `Cast` | 13 |
| `Slice` | 10 |
| `Sum` | 10 |
| `Mul` | 10 |
| `Pad` | 8 |
| `Where` | 6 |
| `ReduceSum` | 6 |
| `Sub` | 4 |
| `Equal` | 2 |
| `Conv` | 1 |
| `Concat` | 1 |

## Initializer names (first 30)

```
v0
v1
v2
v3
v4
v5
v6
v7
v9
v11
v12
v13
v14
v15
v19
v20
v21
v23
v24
v25
v27
v31
v35
v36
v37
v39
v40
v41
_boolmask_where_zero
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, v6, v7, v7) -> v43
    1: Slice          (input, v7, v9, v7) -> v44
    2: Conv           {'kernel_shape': [1, 1]} (input, v4) -> v45
    3: Greater        (v45, v0) -> v46
    4: Cast           {'to': 1} (v46) -> v47
    5: Pad            (v43, v11, v1) -> v48
    6: Slice          (v48, v12, v13, v14) -> v49
    7: Pad            (v43, v15, v1) -> v50
    8: Slice          (v50, v12, v13, v14) -> v51
    9: Sum            (v49, v51) -> v52
   10: Greater        (v52, v0) -> v53
   11: Cast           {'to': 1} (v53) -> v54
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 383
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
