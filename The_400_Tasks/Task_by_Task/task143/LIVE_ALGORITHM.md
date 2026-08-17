# task143 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task143.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task143`
- **Opset**: 17
- **Nodes**: 23
- **Initializers**: 15

## Op frequency

| Op | Count |
|---|---:|
| `ReduceSum` | 5 |
| `Slice` | 3 |
| `Cast` | 3 |
| `Equal` | 2 |
| `Greater` | 2 |
| `Mul` | 2 |
| `Gather` | 1 |
| `Tile` | 1 |
| `Conv` | 1 |
| `Sub` | 1 |
| `Where` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
cand_idx
starts_00
ends_11
ends_33
axis_c
axes_hw
reps_8
zero
ch5
crop_starts
crop_ends
crop_axes
crop_steps
pad_to_external
pad_zero
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, crop_starts, crop_ends, crop_axes, crop_steps) -> input_inner
    1: Gather         {'axis': 1} (input_inner, cand_idx) -> cand
    2: Slice          (cand, starts_00, ends_33, axes_hw) -> ref_part
    3: ReduceSum      {'keepdims': 1} (ref_part, axis_c) -> ref
    4: ReduceSum      {'keepdims': 1} (ref, axes_hw) -> ref_count
    5: Tile           (ref, reps_8) -> ref8
    6: Conv           {'group': 8, 'kernel_shape': [3, 3], 'pads': [0, 0, 2, 2]} (cand, ref8) -> corr
    7: Equal          (corr, ref_count) -> match_b
    8: Cast           {'to': 1} (match_b) -> match
    9: ReduceSum      {'keepdims': 1} (match, axes_hw) -> match_sum
   10: Slice          (match, starts_00, ends_11, axes_hw) -> match_00
   11: Sub            (match_sum, match_00) -> match_else
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 143
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
