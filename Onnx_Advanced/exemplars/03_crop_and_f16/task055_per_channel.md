# task055 — F16 per-channel placement (vs pre-cast)

| Field | Value |
|---|---|
| Task | task055 |
| Pattern | F16 placement — per-channel Cast |
| Measured saving | 14,400 bytes (5,400 per-channel vs 19,800 pre-cast = 14,400 differential) |

## The placement choice

task055's model gathers individual channels and runs F16 ops on each. There are TWO ways to insert the F16 Cast:

### Option A (naive): pre-cast the full input

```
Cast(input [1,10,30,30] F32, F16) → input_f16 [1,10,30,30] F16 = 18,000 bytes intermediate
  → Gather(input_f16, channel_3) → ch3_f16 [1,1,30,30] F16 = 1,800 bytes
  → Gather(input_f16, channel_7) → ch7_f16 [1,1,30,30] F16 = 1,800 bytes
  → Gather(input_f16, channel_9) → ch9_f16 [1,1,30,30] F16 = 1,800 bytes
  → ... 3 channels gathered total
```

Pre-cast intermediate: 18,000 bytes. Plus 3 × 1,800 = 5,400 bytes for gathered channels. Total: 23,400 bytes (the gathered channels are F16, but the full input cast persisted as an intermediate).

Actually since only 3 channels are consumed, the FULL input cast is wasted overhead.

Measured: pre-cast approach = 19,800 bytes total F16 intermediate cost on task055.

### Option B (correct): per-channel cast

```
Gather(input, channel_3) → ch3_f32 [1,1,30,30] F32 = 3,600 bytes
Cast(ch3_f32, F16) → ch3_f16 [1,1,30,30] F16 = 1,800 bytes
Gather(input, channel_7) → ch7_f32 [1,1,30,30] F32 = 3,600 bytes
Cast(ch7_f32, F16) → ch7_f16 [1,1,30,30] F16 = 1,800 bytes
Gather(input, channel_9) → ch9_f32 [1,1,30,30] F32 = 3,600 bytes
Cast(ch9_f32, F16) → ch9_f16 [1,1,30,30] F16 = 1,800 bytes
```

But wait — the F32 gather outputs are intermediates too. Net cost depends on whether the F32 intermediates are reused or discarded after the cast.

Measured: per-channel approach = 5,400 bytes total F16 intermediate cost.

**Net saving: 19,800 - 5,400 = 14,400 bytes** on task055 by switching from pre-cast to per-channel cast.

## Why per-channel wins (when N < 10)

The pre-cast approach creates an intermediate proportional to the FULL input size (`[1, 10, 30, 30]` F16 = 18,000 bytes). The per-channel approach creates intermediates proportional to ONLY the consumed channels (`N × [1, 1, 30, 30]` F16 = N × 1,800 bytes).

Break-even is at N = 10 (all channels consumed). For N < 10, per-channel wins. Break-even per channel:

| N consumed | pre-cast cost | per-channel cost | per-channel wins by |
|---:|---:|---:|---:|
| 1 | 18,000 | 1,800 | 16,200 |
| 2 | 18,000 | 3,600 | 14,400 |
| 3 | 18,000 | 5,400 | 12,600 |
| 5 | 18,000 | 9,000 | 9,000 |
| 7 | 18,000 | 12,600 | 5,400 |
| 9 | 18,000 | 16,200 | 1,800 |
| 10 | 18,000 | 18,000 | 0 (break-even) |

## Generalizes to

`techniques.md` §4 (F16 placement rules) — Rule B: Cast per-channel when N < 10.

The same logic applies to any narrowing op, not just Gather:
- Slice narrowing channels: cast AFTER the Slice.
- Slice narrowing spatial dimensions: cast AFTER the Slice.
- Reduce ops: cast AFTER the Reduce.

**Decision procedure** (memorize this):
1. Find the smallest tensor the F16-eligible compute runs on.
2. Place the F16 Cast immediately upstream of that point. NOT on the model input.
3. Place the F32 Cast-back as far downstream as possible. Drop entirely if F16-output trick applies.

## Anti-pattern: blanket `fp16_selective`

`tools/spread_patterns/pass_fp16_selective.py` runs onnxconverter_common's blanket F16 conversion. It pre-casts the full input by default, which is why it regresses ~134 of 401 tasks. Placement matters more than the trick.

For F16 work, use `pass_fp16_placement_advisory` (flags placement opportunities WITHOUT transforming) and apply manually per the rules above.

## Reproducibility

For any task where the existing graph Gathers individual channels:

1. Inspect the graph: count Gather nodes with channel-axis (`axis=1`) and small index sets.
2. For each Gather, insert a Cast(F16) IMMEDIATELY AFTER it (not before).
3. Continue the rest of the chain in F16.
4. If the chain ends back at F32 (Mul against F32, etc.), insert a Cast(F32) at that boundary — OR investigate dropping the F32 promotion altogether (see techniques.md §3).

Verify byte-exact against LIVE on visible+arc-gen. For pure-arithmetic chains on integer-quantized data, F16 should pass byte-exact under the official decoder's `(out > 0.0)` thresholding.

## Cross-ref

- `task117_crop15_f16.md` — F16 stacked with crop.
- `task384_f16_drop.md` — F16-output trick alone.
- `ort_reality.md` — F16 dtype handling in ORT.
