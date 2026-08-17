# task123 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task123.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 13
- **Initializers**: 13

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 3 |
| `Gather` | 3 |
| `ReduceSum` | 2 |
| `Max` | 1 |
| `Greater` | 1 |
| `Where` | 1 |
| `Concat` | 1 |
| `Squeeze` | 1 |

## Initializer names (first 30)

```
starts_40
ends_40
axes_40
starts_04
ends_04
c_half
starts_col0
ends_col0
axes_col0
idx_p5
idx_p4
zero_row
level_map
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, starts_40, ends_40, axes_40) -> input_40
    1: ReduceSum      {'axes': [1], 'keepdims': 1} (input_40) -> s40
    2: Slice          (input, starts_04, ends_04, axes_40) -> input_04
    3: ReduceSum      {'axes': [1], 'keepdims': 1} (input_04) -> s04
    4: Max            (s40, s04) -> flag_P5
    5: Greater        (flag_P5, c_half) -> is_P5
    6: Slice          (input, starts_col0, ends_col0, axes_col0) -> input_col0
    7: Gather         {'axis': 2} (input_col0, idx_p5) -> colors_p5
    8: Gather         {'axis': 2} (input_col0, idx_p4) -> colors_p4
    9: Where          (is_P5, colors_p5, colors_p4) -> colors_ext
   10: Concat         {'axis': 2} (colors_ext, zero_row) -> colors_ext_padded
   11: Squeeze        {'axes': [3]} (colors_ext_padded) -> colors_flat
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 123
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
