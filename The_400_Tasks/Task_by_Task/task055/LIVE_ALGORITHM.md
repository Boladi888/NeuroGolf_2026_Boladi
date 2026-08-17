# task055 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task055.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task055`
- **Opset**: 17
- **Nodes**: 167
- **Initializers**: 37

## Op frequency

| Op | Count |
|---|---:|
| `Gather` | 62 |
| `Add` | 62 |
| `Mul` | 11 |
| `Equal` | 10 |
| `Sub` | 7 |
| `Where` | 6 |
| `Cast` | 4 |
| `Concat` | 3 |
| `Conv` | 2 |

## Initializer names (first 30)

```
k_row
k_col
zero_map
f0
f1
f2
idx_ch0
idx_ch8
idx_1
idx_2
idx_3
idx_4
idx_5
idx_6
idx_7
idx_9
idx_10
idx_11
idx_12
idx_13
idx_14
idx_15
idx_16
idx_17
idx_18
idx_19
idx_20
idx_21
idx_22
idx_23
```

## First 12 nodes (algorithm entry point)

```
    0: Gather         {'axis': 1} (input, idx_ch0) -> zero
    1: Gather         {'axis': 1} (input, idx_ch8) -> eight
    2: Conv           (eight, k_row) -> row_sum
    3: Conv           (eight, k_col) -> col_sum
    4: Equal          (row_sum, f0) -> equal_5
    5: Cast           {'to': 1} (equal_5) -> row_is_zero
    6: Equal          (col_sum, f0) -> equal_8
    7: Cast           {'to': 1} (equal_8) -> col_is_zero
    8: Equal          (row_sum, f2) -> equal_11
    9: Cast           {'to': 1} (equal_11) -> row_is_two
   10: Equal          (col_sum, f2) -> equal_14
   11: Cast           {'to': 1} (equal_14) -> col_is_two
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 55
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
