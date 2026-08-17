# task320 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task320.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 11
- **Nodes**: 10
- **Initializers**: 12

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 2 |
| `Add` | 2 |
| `ReduceSum` | 1 |
| `Conv` | 1 |
| `Greater` | 1 |
| `Where` | 1 |
| `Mul` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
c2_s
c2_e
c_ax
io_only_pad_zero
ConvMulFusion_W_W_cum
c1
delta_w
io_only_crop_starts
io_only_crop_ends
io_only_crop_axes
io_only_crop_steps
io_only_pad_to_external
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, io_only_crop_starts, io_only_crop_ends, io_only_crop_) -> input_inner
    1: Slice          (input_inner, c2_s, c2_e, c_ax, c_ax) -> M
    2: ReduceSum      {'axes': [2], 'keepdims': 1} (M) -> col_count
    3: Add            (col_count, c1) -> Lp1
    4: Conv           {'group': 1, 'kernel_shape': [30, 1], 'pads': [29, 0, 0, 0], 'strides': [1, 1]} (M, ConvMulFusion_W_W_cum) -> twoP
    5: Greater        (twoP, Lp1) -> gt_bool
    6: Where          (gt_bool, M, io_only_pad_zero) -> paint8
    7: Mul            (paint8, delta_w) -> delta
    8: Add            (input_inner, delta) -> output_inner
    9: Pad            (output_inner, io_only_pad_to_external, io_only_pad_zero) -> output
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 320
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
