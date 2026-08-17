# task280 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task280.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 11
- **Nodes**: 197
- **Initializers**: 53

## Op frequency

| Op | Count |
|---|---:|
| `Slice` | 61 |
| `Pad` | 61 |
| `Mul` | 31 |
| `Sub` | 14 |
| `Gather` | 11 |
| `Sum` | 10 |
| `CumSum` | 8 |
| `Concat` | 1 |

## Initializer names (first 30)

```
v5
v6
v7
v10
v1
v2
v0
v3
v8
v9
padslice_shift_conv_6_ends
padslice_shift_conv_6_steps
padslice_shift_conv_6_pads
padslice_shift_conv_6_value
padslice_shift_conv_9_starts
padslice_shift_conv_9_ends
padslice_shift_conv_9_pads
padslice_shift_conv_23_starts
padslice_shift_conv_23_pads
padslice_shift_conv_31_starts
padslice_shift_conv_31_pads
padslice_shift_conv_39_starts
padslice_shift_conv_39_pads
padslice_shift_conv_47_starts
padslice_shift_conv_47_pads
padslice_shift_conv_71_ends
padslice_shift_conv_71_pads
padslice_shift_conv_79_ends
padslice_shift_conv_79_pads
padslice_shift_conv_87_ends
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, v5, v6, v7) -> v292
    1: Gather         {'axis': 1} (v292, v10) -> v293
    2: Gather         {'axis': 1} (v292, v1) -> v294
    3: Gather         {'axis': 1} (v292, v2) -> v295
    4: Sum            (v294, v295) -> v296
    5: Sub            (v293, v293) -> v522
    6: Slice          (v296, v5, padslice_shift_conv_6_ends, v7, padslice_shift_con) -> padslice_shift_conv_6_slice_out
    7: Pad            (padslice_shift_conv_6_slice_out, padslice_shift_conv_6_pads,) -> v298
    8: Slice          (v296, padslice_shift_conv_9_starts, padslice_shift_conv_9_en) -> padslice_shift_conv_9_slice_out
    9: Pad            (padslice_shift_conv_9_slice_out, padslice_shift_conv_9_pads,) -> v301
   10: Slice          (v296, padslice_shift_conv_23_starts, padslice_shift_conv_9_e) -> padslice_shift_conv_23_slice_out
   11: Pad            (padslice_shift_conv_23_slice_out, padslice_shift_conv_23_pad) -> v317
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 280
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
