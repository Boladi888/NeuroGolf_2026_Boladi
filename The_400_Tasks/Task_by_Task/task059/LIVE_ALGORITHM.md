# task059 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task059.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 11
- **Nodes**: 20
- **Initializers**: 19

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 2 |
| `Conv` | 2 |
| `Sub` | 2 |
| `Greater` | 2 |
| `Reshape` | 2 |
| `MatMul` | 2 |
| `Mul` | 2 |
| `ReduceMax` | 1 |
| `Cast` | 1 |
| `ReduceSum` | 1 |
| `Where` | 1 |
| `Sum` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
W_mask
W_sum
s_starts
s_ends
s_axes
s_steps
half
T_3x30
Tt_30x3
rs_1_1_30_30
zero
lines_const
ch0_base_const
ch0_onehot
crop_starts
crop_ends
crop_axes
crop_steps
pad_to_external
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, crop_starts, crop_ends, crop_axes, crop_steps) -> input_inner
    1: Conv           {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0]} (input_inner, W_mask) -> M
    2: Conv           {'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [4, 4]} (M, W_sum) -> region_sums_7x7
    3: Slice          (region_sums_7x7, s_starts, s_ends, s_axes, s_steps) -> region_sums
    4: ReduceMax      {'axes': [2, 3], 'keepdims': 1} (region_sums) -> max_rs
    5: Sub            (max_rs, half) -> max_minus
    6: Greater        (region_sums, max_minus) -> mbest_bool
    7: Cast           {'to': 1} (mbest_bool) -> Mbest
    8: Reshape        (Mbest, s_ends) -> Mbest_2d
    9: MatMul         (Mbest_2d, T_3x30) -> mid
   10: MatMul         (Tt_30x3, mid) -> region_mask_2d
   11: Reshape        (region_mask_2d, rs_1_1_30_30) -> region_mask
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 59
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
