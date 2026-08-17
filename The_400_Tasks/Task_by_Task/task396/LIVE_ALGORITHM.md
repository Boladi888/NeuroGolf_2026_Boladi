# task396 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task396.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `graph`
- **Opset**: 10
- **Nodes**: 417
- **Initializers**: 87

## Algorithm hints (from naming)

- largest-rectangle scoring

## Op frequency

| Op | Count |
|---|---:|
| `Reshape` | 84 |
| `Cast` | 62 |
| `Mul` | 58 |
| `Less` | 32 |
| `ArgMax` | 28 |
| `Sub` | 28 |
| `Conv` | 25 |
| `ReduceMax` | 25 |
| `Div` | 25 |
| `Floor` | 25 |
| `Add` | 9 |
| `Gather` | 7 |
| `Concat` | 3 |
| `Min` | 2 |
| `Slice` | 1 |

## Initializer names (first 30)

```
lrrc_sl_st_ch
lrrc_sl_en_ch
lrrc_one
lrrc_half
lrrc_one_pt_five
lrrc_neg_one
lrrc_sentinel_neg
lrrc_const_29
lrrc_rs11
lrrc_rs_row
lrrc_rs_col
lrrc_rs_1x10x1x1
lrrc_k_4x4
lrrc_thr_4x4
lrrc_score_4x4
lrrc_rs_flat_4x4
lrrc_out_w_4x4
lrrc_k_4x5
lrrc_thr_4x5
lrrc_score_4x5
lrrc_rs_flat_4x5
lrrc_out_w_4x5
lrrc_k_4x6
lrrc_thr_4x6
lrrc_score_4x6
lrrc_rs_flat_4x6
lrrc_out_w_4x6
lrrc_k_4x7
lrrc_thr_4x7
lrrc_score_4x7
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, lrrc_sl_st_ch, lrrc_sl_en_ch, lrrc_sl_st_ch, lrrc_sl_) -> lrrc_no_bg
    1: ReduceSum      {'axes': [2, 3], 'keepdims': 0} (lrrc_no_bg) -> lrrc_counts
    2: ArgMax         {'axis': 1, 'keepdims': 0} (lrrc_counts) -> lrrc_frame_idx0
    3: Reshape        (lrrc_frame_idx0, lrrc_sl_st_ch) -> lrrc_frame_idx0_1d
    4: Cast           {'to': 1} (lrrc_frame_idx0_1d) -> lrrc_frame_idx0_f
    5: Add            (lrrc_frame_idx0_f, lrrc_one) -> lrrc_frame_color_f
    6: Cast           {'to': 7} (lrrc_frame_color_f) -> lrrc_frame_color
    7: Less           (lrrc_counts, lrrc_half) -> lrrc_zero_counts
    8: Mul            (lrrc_counts, lrrc_neg_one) -> lrrc_neg_counts
    9: Where          (lrrc_zero_counts, lrrc_sentinel_neg, lrrc_neg_counts) -> lrrc_least_scores
   10: ArgMax         {'axis': 1, 'keepdims': 0} (lrrc_least_scores) -> lrrc_marker_idx0
   11: Reshape        (lrrc_marker_idx0, lrrc_sl_st_ch) -> lrrc_marker_idx0_1d
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 396
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
