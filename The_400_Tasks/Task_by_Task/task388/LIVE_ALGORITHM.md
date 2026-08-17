# task388 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task388.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `t388_2x2_tile_color_smallframe`
- **Opset**: 13
- **Nodes**: 48
- **Initializers**: 30

## Op frequency

| Op | Count |
|---|---:|
| `Pad` | 19 |
| `Where` | 12 |
| `Equal` | 5 |
| `Slice` | 2 |
| `ReduceSum` | 2 |
| `Mul` | 2 |
| `Conv` | 1 |
| `Sub` | 1 |
| `ReduceMax` | 1 |
| `Sqrt` | 1 |
| `Add` | 1 |
| `Sum` | 1 |

## Initializer names (first 30)

```
axes_hw
axis_1_in
loc_st_hw
loc_en_in
ch_0_st
pad_TL
pad_pick_W_to_12
pad_pick_H_to_12
pad_to_30
pad_TRshort_2
pad_BLshort_2
pad_BR_2
pad_TRshort_3
pad_BLshort_3
pad_BR_3
pad_TRshort_4
pad_BLshort_4
pad_BR_4
pad_TRshort_5
pad_BLshort_5
pad_BR_5
pad_TRshort_6
pad_BLshort_6
pad_BR_6
n_const_2
n_const_3
n_const_4
n_const_5
conv_w
eight
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, loc_st_hw, loc_en_in, axes_hw) -> input_6
    1: Conv           {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]} (input_6, conv_w) -> encoded
    2: Slice          (input_6, ch_0_st, axis_1_in, axis_1_in) -> ch_0_in
    3: ReduceSum      {'keepdims': 1} (input_6, axis_1_in) -> in_grid
    4: Sub            (in_grid, ch_0_in) -> nonbg
    5: ReduceMax      {'axes': [2], 'keepdims': 1} (nonbg) -> col_has_nonbg
    6: ReduceSum      {'keepdims': 1} (in_grid, axes_hw) -> n_sq
    7: Sqrt           (n_sq) -> n_count
    8: Mul            (ch_0_in, col_has_nonbg) -> mask_8_f
    9: Mul            (mask_8_f, eight) -> delta_8
   10: Add            (encoded, delta_8) -> base
   11: Pad            (base, pad_TRshort_2) -> TRshort_2
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 388
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
