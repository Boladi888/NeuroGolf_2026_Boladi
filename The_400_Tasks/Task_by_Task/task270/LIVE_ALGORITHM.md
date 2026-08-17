# task270 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task270.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `neurogolf-v2-task270-compact-markers`
- **Graph name**: `task270_compact_direction_markers`
- **Opset**: 17
- **Nodes**: 56
- **Initializers**: 21

## Algorithm hints (from naming)

- anchor/marker-pattern (silent-fail-trap family)
- directional projection / compaction

## Op frequency

| Op | Count |
|---|---:|
| `Greater` | 11 |
| `Sub` | 9 |
| `Where` | 8 |
| `Conv` | 8 |
| `Slice` | 4 |
| `CumSum` | 4 |
| `ReduceSum` | 4 |
| `Sum` | 3 |
| `Cast` | 3 |
| `Concat` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
axis_row
axis_col
pad_to_30
zero
canvas15
zero15
_boolmask_where_zero
slice_pad_shift_conv_w_22
slice_pad_shift_conv_w_24
slice_pad_shift_conv_w_26
slice_pad_shift_conv_w_28
slice_chain_starts_1
slice_chain_ends_1
slice_chain_axes_1
slice_chain_steps_1
slice_chain_starts_2
slice_chain_ends_2
slice_chain_starts_3
slice_chain_ends_3
slice_chain_starts_4
slice_chain_ends_4
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, slice_chain_starts_1, slice_chain_ends_1, slice_chain) -> center1
    1: Slice          (input, slice_chain_starts_2, slice_chain_ends_2, slice_chain) -> center2
    2: Slice          (input, slice_chain_starts_3, slice_chain_ends_3, slice_chain) -> marker3_in
    3: Slice          (input, slice_chain_starts_4, slice_chain_ends_4, slice_chain) -> marker7_in
    4: CumSum         (marker7_in, axis_col) -> marker7_out_left_inclusive
    5: Sub            (marker7_out_left_inclusive, marker7_in) -> marker7_out_left_seen_raw
    6: Greater        (marker7_out_left_seen_raw, zero) -> marker7_out_left_seen_bool
    7: ReduceSum      {'keepdims': 1} (marker7_in, axis_col) -> marker7_out_row_tot
    8: Sub            (marker7_out_row_tot, marker7_out_left_inclusive) -> marker7_out_right_seen_raw
    9: Greater        (marker7_out_right_seen_raw, zero) -> marker7_out_right_seen_bool
   10: CumSum         (marker7_in, axis_row) -> marker7_out_up_inclusive
   11: Sub            (marker7_out_up_inclusive, marker7_in) -> marker7_out_up_seen_raw
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 270
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
