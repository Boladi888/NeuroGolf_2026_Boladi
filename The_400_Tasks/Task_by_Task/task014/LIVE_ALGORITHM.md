# task014 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task014.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 42
- **Initializers**: 14

## Op frequency

| Op | Count |
|---|---:|
| `Add` | 6 |
| `Reshape` | 5 |
| `Sub` | 5 |
| `Gather` | 5 |
| `ArgMax` | 4 |
| `Less` | 4 |
| `Cast` | 3 |
| `Mul` | 3 |
| `ReduceMax` | 2 |
| `Where` | 2 |
| `ReduceSum` | 1 |
| `Greater` | 1 |
| `ArgMin` | 1 |

## Initializer names (first 30)

```
cfbcl_offsets30
cfbcl_twenty_nine
cfbcl_one_i
cfbcl_big_f
cfbcl_zero_f
cfbcl_one_f
cfbcl_shape_10
cfbcl_bg_penalty
cfbcl_shape_30
cfbcl_reverse_idx
cfbcl_thirty_i
cfbcl_twenty_nine_30
cfbcl_shape_1_1_30_1
cfbcl_shape_1_1_1_30
```

## First 12 nodes (algorithm entry point)

```
    0: ReduceSum      {'axes': [2, 3], 'keepdims': 1} (input) -> cfbcl_counts_4d
    1: Reshape        (cfbcl_counts_4d, cfbcl_shape_10) -> cfbcl_counts
    2: Greater        (cfbcl_counts, cfbcl_zero_f) -> cfbcl_present_b
    3: Cast           {'to': 1} (cfbcl_present_b) -> cfbcl_present
    4: Sub            (cfbcl_one_f, cfbcl_present) -> cfbcl_absent
    5: Mul            (cfbcl_absent, cfbcl_big_f) -> cfbcl_absent_penalty
    6: Add            (cfbcl_absent_penalty, cfbcl_bg_penalty) -> cfbcl_total_penalty
    7: Add            (cfbcl_counts, cfbcl_total_penalty) -> cfbcl_counts_masked
    8: ArgMin         {'axis': 0, 'keepdims': 1} (cfbcl_counts_masked) -> cfbcl_color
    9: Gather         {'axis': 1} (input, cfbcl_color) -> cfbcl_mask_col
   10: ReduceMax      {'axes': [3], 'keepdims': 1} (cfbcl_mask_col) -> cfbcl_row_any_4d
   11: Reshape        (cfbcl_row_any_4d, cfbcl_shape_30) -> cfbcl_row_any
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 14
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
