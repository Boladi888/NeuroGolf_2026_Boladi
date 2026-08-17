# DEAD ROUTE — task308 batched broadcast one-hot (the prompt's RectU8IndexBatch idea)

- **Live bar:** 1460 (manifest, owner Boladi, sha `14fac9a488a9…`, updated 2026-07-02). Local staged best `task308_u8_bd_shift_cost1394` (worker3, fully gated, below bar; captain not yet promoted).
- **Attempted route:** batch the 9 rectangles + cross into the 9-channel one-hot **directly by broadcasting** each color's `(mag_y, mag_x)` signature against the 7×7 grid — the prompt's "single U8 color-index [1,1,7,7]=49 + Equal-tail, drop the 9-ch BOOL" formulation. Built on v1's verified count-4 moment head. Probe: `prototypes/task308_broadcast_onehot.onnx` (sha256 `36bb76b9…`), builder `prototypes/build_task308_broadcast_onehot.py`.
- **Exact gate result:** `[visible] 266/266 (100.00%) PASS` — the formulation is CORRECT — but **`score_v2 = (0, 4852, 116)` → cost 4968**, i.e. **3.6× WORSE** than the 1394 scatter graph. Cost-dead; no truth/ab/battery run (protocol: don't gate a cost-loser).
- **Why it's dead (forced, not a tuning miss):** any channel-space (broadcast) construction materialises **[1,9,7,7]=441-byte** intermediates *per stage*. `largest_intermediate` shows **seven** of them: `rect_paint`, `cross_paint`, `fg_paint`, `fg_u8`, `bg_paint`, `onehot_pre`, `onehot_bool` = 7×441 ≈ 3,087 B alone. Rectangles vs cross must be OR-merged per channel, and the background-fill must be OR-ed into the bg channel — each merge is another [1,9,7,7]. There is no way to fold rect+cross+bg without ≥3 such tensors.

## The "550 floor" is a MISREAD of the output tail
The prompt frames it as "U8 index [1,1,7,7]=49 **NOT** 9-ch BOOL[1,9,7,7]=441". These are **sequential, not alternatives**:
- The U8 color-index canvas (49 B) is the **input** to the Equal-tail.
- The Equal-tail **output** is `onehot_bool [1,9,7,7]=441`, which **is the graph's required output** (fed to the free channel-0+spatial `Pad` → [1,10,30,30]). 441 is the minimal static one-hot (9 colors × 7×7 worst case) and is **not eliminable**.

The live 1394 graph **already** builds a U8 index canvas (`labels_grid[7,7]=49`) then Equal-tail → `onehot_bool 441`. So the floor is **49 + 441 + machinery**, never 49 alone.

## Forced-component floor (practical ≈ 800, not 550)
| component | bytes | forced because |
|---|---:|---|
| `onehot_bool` [1,9,7,7] | 441 | the output one-hot; Pad adds ch-0 + spatial for free |
| moment tracks 5×F32[1,10] | 200 | count-4 method needs sum_y/x, sum_y2/x2, count; Einsum(F32 in)→F32, and casting the [1,10,30,30] input to F16 costs 18,000 B, so [1,10] can't be narrowed pre-Einsum |
| scatter indices [1,16] I32 | 64 | 4 active colors × 4 corners; sparse position-space placement |
| U8 index canvas [7,7] | 49 | feeds the Equal-tail |
| bg/active/classify/window | ~50 | ArgMax bg, count==4, mag/cross thresholds |
| **practical floor** | **≈ 800** | |

The live 1394 sits ~600 above this; that gap is the **TopK-selection path** (top_idx/active_score/GatherElements ≈ 216 B) + duplicate flat/reshape board tensors — worker3 already ground the scalar arithmetic out and called it exhausted. Closing the remaining gap is **epsilon-shaving of the same moments→TopK→scatter→Equal structure**, not a structural rewrite to 550.

## Why scatter is structurally forced (not replaceable by broadcast)
Building the U8 index canvas `[1,1,7,7]` without a [1,9,·,·] intermediate requires, per cell (r,c), a reverse map `(|dy|,|dx|) → color`. Built from the ≤4 `(my,mx,color)` triples, **that reverse map IS a scatter** (scatter each color to its 4 corner positions). The patterns are provably **disjoint** (distinct colors ⇒ distinct (my,mx) corner-sets; the single cross sits at edge-midpoints (ctr,0)/(0,ctr) that no rectangle corner (my≥1,mx≥1) occupies; center (0,0) is always bg), so no sequential/priority state is needed — but the placement is inherently **position-space (ScatterElements)**, and every channel-space alternative pays 441/slot (proven: 4968).

## Remaining hypothesis (only lever left, incremental not structural)
Trim the TopK path: scatter all 9 colors' corners with a bg-mask for inactive slots (indices grow [1,16]→[1,36] I32 = +80 B but delete the ~216 B TopK+gather path), or drop the duplicate flat-board/reshape copy (Equal on flat `labels[1,49]` then reshape the BOOL, saving `labels_grid` 49 B). Estimated best ≈ 1,150–1,250 — a real but modest sub-1394 shave, **still ~2× above the prompt's 550 target**. Not attempted here (epsilon-class; worker3 owns the 1394 lineage).

**VERDICT: WALL vs the 550 target.** 550 rests on double-counting the forced 441 output tail. Practical floor ≈ 800; batched-broadcast family is dead (4968, proven). Live best remains the 1394 scatter graph.
