# task026 — BOOL complementary-channel canvas + OR/NOR algebraic collapse (worker-6, 2026-06-11)

**Family:** halves fold/overlay → per-cell boolean combine, color-painted output.
**Result:** LIVE 1174 / team-best(aditya) 724 → **356** byte-equal (200268/200268 AB, 200000/200000 truth).

## Transferable techniques

### 1. OR/NOR/XOR boolean chains collapse to one product
A graph that computes `lr=L*R; lpr=L+R; orv=lpr-lr; combined=1-orv; out=1-combined`
(5 ops) is just `out = 1 - L*R` when L,R ∈ {0,1}. Identity:
`OR(L,R) = (1-L)+(1-R)-(1-L)(1-R) = 1 - LR` (where L,R are the *complement* inputs).
Always algebra-reduce a Sub/Add/Mul boolean cascade before optimizing storage — it
often drops to one Mul + one Sub (or one `And` + one `Not` in BOOL).

### 2. Complementary output channels are BOTH required when both colors are real
Trap (cost me a 267-cost dead end): the rule was "color 8 where NOR holds, else color 0".
I dropped channel 0 thinking "absence = background". WRONG — color 0 is a REAL output
color and must have channel 0 SET; an unset cell decodes to "no color"(10) ≠ 0.
If the output palette is {color A, color B}, the canvas needs BOTH channel A and
channel B (complementary BOOL planes), not one. Truth crashed to 1.65% with one channel.

### 3. BOOL canvas + single channel-and-spatial Pad to output
For a small painted result, build the channel canvas at grid size `[1,C,h,w]` in **BOOL**
(1 byte/elem vs 2 F16 / 4 F32), then ONE `Pad` (channel-after + spatial-after) → `output`.
The `output` tensor is SKIPPED from memory, so the canvas `[1,C,h,w]` is the only charged
big tensor. Trim C to `max_used_channel+1` (here ch8 → 9-channel canvas, then Pad +1 channel
to reach 10) — saves one channel-plane. Logic in BOOL: `Cast(slice)→Cast(slice)→And→Not`
(60 B of [1,1,h,w] intermediates) beats `Mul`-in-F32-then-Cast (90 B).

### 4. Two narrow F32 slices beat one wide one
Extracting two non-adjacent half-columns: two `[1,1,5,3]` F32 slices = 120 B total
beats one `[1,1,5,7]` F32 slice (140 B) or a Gather of `[0,1,2,4,5,6]` (`[1,1,5,6]`=120 B + reduce).
Slice preserves input F32 dtype (no fused slice-cast op), so the F32 half-values are an
irreducible extraction floor — minimize their element count, not their dtype.
