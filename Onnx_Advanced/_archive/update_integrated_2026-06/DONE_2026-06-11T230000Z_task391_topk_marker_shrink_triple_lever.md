# TopK-marker-shrink triple narrowing lever (task391: Aditya 304 -> 236 byte-equal)

**Family:** Aditya all-F32/I64 `AveragePool/ReduceSum -> Slice -> TopK -> OneHot -> Pad`
nets that emit a small one-hot answer grid (counts -> rank -> place). Extends #116 (small-grid
OneHot) and #81 (small-grid pad-floor) with a dtype-narrowing recipe.

**The lever (stack all three; each verified byte-equal over 200k AB):**
1. **OneHot values/output = F16** (decoder thresholds `>0`, F16 one-hot bits exact). NOTE:
   BOOL is cheaper on paper (1B/elem) but **OneHot has no BOOL ORT kernel** — `OneHot(11)`
   returns NOT_IMPLEMENTED. F16 is the onehot-output floor.
2. **Cast the TopK index I64 -> I32 immediately after TopK**, then run the drop-Slice + Reshape
   on the I32 tensor (halves every downstream index tensor). OneHot `depth` dtype must MATCH
   the indices (trap §2) — make depth I32 too.
3. **Cast the count tensor F32 -> F16 BEFORE the bg-Slice.** Counter-intuitive but a net win:
   the [1,10,1,1] F16 cast tensor (20B) is cheaper than the F32 colors+top_values it narrows
   (52B -> 26B). Apply this BEFORE the index I32 cast — doing I32 first without the F16
   count-cast actually regresses (task391 v1-i32 = 262 vs v5 = 236).

**Hard floors (do not chase below them):**
- The count tensor is **F32-bound**: AveragePool/ReduceSum/Conv/MatMul all emit F32 from the
  F32 one-hot input; casting the input to F16 first materializes [1,10,30,30] = 18,000B.
- The **TopK index is I64** by spec (mandatory ~32B for k=4) and the output's descending-count
  order is a genuine sort — no shortcut when the dominant color is uniformly distributed.
- **Every named Reshape that drops a rank costs a full counted tensor copy.** OneHot inserts the
  depth axis, so to emit final [1,9,3,1] without a post-OneHot reshape the index must be [1,3,1];
  TopK emits rank-4 [1,4,1,1] -> exactly ONE rank-dropping reshape is required. Place it on the
  smallest tensor (the [1,3] I32 index, 12B). Trailing `1,1` dims cost 0 — never squeeze them
  with extra Reshapes (task391 v2/v3 regressed +13/+11 doing so).

**Rejected:** suppress-max + TopK k=3 to delete the drop-Slice (ReduceMax+Equal+Cast+Mul+Sub
costs more than the Slice it removes: 272 vs 236).

**Transfer:** any Aditya net of shape `count/pool -> Slice -> TopK -> OneHot -> Pad` with all-F32
intermediates and I64 indices. Re-price the count tensor as F32-mandatory + F16-cast, the index
chain as I64-mandatory + I32-cast, and the OneHot output as F16. task391 304->236, +0.253 LB est.
