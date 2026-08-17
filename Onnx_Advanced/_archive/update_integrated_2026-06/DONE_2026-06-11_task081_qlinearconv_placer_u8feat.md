# DONE 2026-06-11 — task081: the free-output PLACER can be a QLinearConv reading a u8 feat (no F16 cast)

**Transferable lesson (worker-1, task081 LCornerFreeout, 1819 -> 636, byte-equal + ARC-GEN 5000/5000):**

The #78 free-output placer Conv normally forces its input `feat` to **F16** (Conv has no u8 kernel),
so a 3-channel 7x7 placer feed costs 294 B (F16) AND you pay a separate Cast. **Make the placer a
`QLinearConv` instead** and feed it the u8 `feat` directly — the F16 feat and its Cast both vanish:

```
featu = Concat([ones7u, is_cyan_u8, is_blue_u8])  # [1,3,7,7] u8 = 147 B
output = QLinearConv(featu, s1,z0, Wp[10,3,1,1] INT8, s1,z0i8, s1,z0,  pads=[0,0,23,23])  # FREE
```

- Weights go **INT8** so the complement channel works: `ch0 = 1*active - 1*cyan - 1*blue` (the −1's
  need a signed weight dtype; u8 weights can't). All final channel values here are {0,1} ⇒ u8 output, decode `>0`.
- `y_zero_point` MUST be u8 (INT8-y is NOT_IMPLEMENTED — already in #85). Scales=1, zps=0 ⇒ exact integer conv.
- **BIAS-LEAK still applies** (#85.3): no Conv bias / constant channel — the in-grid indicator (`ones7`)
  must arrive as a real **data** channel, else the 30x30 padding region lights up (certain fail).
  This is why the feed is 3 channels, not 2 (a 2-ch {cyan,blue}+bias map can't tell in-grid-bg from padding).
- Net on task081: F16 placer feed 294 + Cast vs u8 QLinearConv placer 147, no Cast — and it cascades:
  the whole detection then stays u8 (QLinearConv count + int32-bias binarize, #85.1) at 49 B/tensor.
  Only the single F32 entry read (cropping ch8 -> [1,1,7,7] F32 = 196 B) stays F32; everything else u8.

**Also banked (smaller):** when "covered cells" = {the L's 3 cyan} ∪ {the 1 hole} and every cyan cell
is provably inside its own hit-window, `is_hole = covered - cyan` is **one u8 Sub** (no underflow,
no separate not-cyan mask + Mul). Check the "every X is covered" invariant on the generator before relying on it.

**Floor pattern for always-fixed-size grids:** read the ARC-GEN generator FIRST — task081's grid is
ALWAYS 7x7 (`generate(size=7)`), so a fully static 7x7 crop is hidden-safe and the #81 Equal-tail 900 B
canvas is never the floor; the 7x7 placer feed (147 B) dominates it. The single 3x3 conv hole-detector
was falsified (2827/5000 vs the generator) — the 2-stage count→spread is the minimal 2-layer boolean.

Suggested home: techniques.md #85 (QLinearConv toolkit) as a new bullet "placer-conv-as-QLinearConv
reads u8 feat directly (i8 weights for complement channels)"; cross-ref #78.
