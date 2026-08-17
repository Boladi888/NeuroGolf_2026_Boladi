# task263 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task263.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 145
- **Initializers**: 27

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 25 |
| `Cast` | 20 |
| `Slice` | 19 |
| `ReduceSum` | 19 |
| `Sub` | 19 |
| `Where` | 15 |
| `Less` | 15 |
| `Greater` | 6 |
| `Sum` | 6 |
| `Pad` | 1 |

## Initializer names (first 30)

```
v0
v2
v3
v4
v5
v6
v7
v8
v12
v13
v18
v19
v21
v22
v30
v31
v33
v34
v42
v43
v45
v46
v54
v55
v57
v58
_boolmask_where_zero
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, v3, v4, v5) -> v66
    1: ReduceSum      {'axes': [0, 1, 2, 3], 'keepdims': 1} (v66) -> v67
    2: Greater        (v67, v0) -> v68
    3: Slice          (input, v6, v7, v8) -> v71
    4: Slice          (v71, v12, v13, v13) -> v73
    5: ReduceSum      {'axes': [0, 1, 2, 3], 'keepdims': 1} (v71) -> v74
    6: ReduceSum      {'axes': [0, 1, 2, 3], 'keepdims': 1} (v73) -> v75
    7: Sub            (v74, v75) -> v76
    8: Where          (v68, v76, v76) -> v83
    9: Greater        (v83, v0) -> v84
   10: Cast           {'to': 1} (v84) -> v85
   11: Slice          (input, v18, v19, v8) -> v86
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 263
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
