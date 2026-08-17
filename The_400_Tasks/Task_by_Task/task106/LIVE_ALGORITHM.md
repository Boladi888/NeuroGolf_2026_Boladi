# task106 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task106.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task106_xsot_rot_quilt`
- **Opset**: 12
- **Nodes**: 21
- **Initializers**: 11

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 3 |
| `Squeeze` | 2 |
| `Cast` | 2 |
| `ReduceMax` | 2 |
| `Sub` | 2 |
| `Add` | 2 |
| `Where` | 2 |
| `Conv` | 1 |
| `Greater` | 1 |
| `Reshape` | 1 |
| `ReduceMin` | 1 |
| `Equal` | 1 |
| `Gather` | 1 |

## Initializer names (first 30)

```
shape_flat10
one_i
two_i
thirty_i
color_weights
row_pos1d
t106_idx2
t106_idx3
zero_f
t106_valid2
t106_valid3
```

## First 12 nodes (algorithm entry point)

```
    0: Conv           (input, color_weights) -> scalar
    1: Squeeze        {'axes': [1]} (scalar) -> scalar2d
    2: Greater        (scalar2d, zero_f) -> occ_bool
    3: Cast           {'to': 1} (occ_bool) -> occ_float
    4: ReduceMax      {'axes': [2], 'keepdims': 0} (occ_float) -> row_occ
    5: Reshape        (input, shape_flat10) -> flat
    6: Cast           {'to': 7} (row_occ) -> bbox106_row_occ_i
    7: Sub            (one_i, bbox106_row_occ_i) -> bbox106_row_inv
    8: Mul            (bbox106_row_inv, thirty_i) -> bbox106_row_penalty
    9: Add            (row_pos1d, bbox106_row_penalty) -> bbox106_row_pos_pen
   10: ReduceMin      {'axes': [1], 'keepdims': 1} (bbox106_row_pos_pen) -> bbox106_first_row
   11: Mul            (bbox106_row_occ_i, row_pos1d) -> bbox106_row_keep
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 106
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
