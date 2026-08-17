# task389 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task389.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `t389_swap_5_other_color_keep_pos`
- **Opset**: 13
- **Nodes**: 10
- **Initializers**: 10

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 2 |
| `Cast` | 2 |
| `ReduceMax` | 2 |
| `Where` | 2 |
| `Mul` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
axes_hw
spatial_st
spatial_en
ch_5_st
ch_5_en
axis_1_arr
mask_no_5
bg_inside_oh
zero_oh
pad_to_30
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, spatial_st, spatial_en, axes_hw) -> input_5
    1: Slice          (input_5, ch_5_st, ch_5_en, axis_1_arr) -> ch_5_f
    2: Cast           {'to': 9} (ch_5_f) -> ch_5_b
    3: ReduceMax      {'axes': [1], 'keepdims': 1} (input_5) -> in_grid_f
    4: Cast           {'to': 9} (in_grid_f) -> in_grid_b
    5: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (input_5) -> per_ch_max
    6: Mul            (per_ch_max, mask_no_5) -> other_color_oh
    7: Where          (ch_5_b, other_color_oh, bg_inside_oh) -> inner
    8: Where          (in_grid_b, inner, zero_oh) -> output_f
    9: Pad            (output_f, pad_to_30) -> output
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 389
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
