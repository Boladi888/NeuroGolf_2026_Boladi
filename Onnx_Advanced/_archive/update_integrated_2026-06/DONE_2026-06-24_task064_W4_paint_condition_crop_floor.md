# DONE drop — task064 W4 (2026-06-24): the "crop the 900B paint plane" lever is structurally dead

**Context:** task064 v12 (#110 dual-pack ray renderer) charged-tensor #1 is `paint30` BOOL
`[1,1,30,30]` = 900 B, the condition of the final `Where(paint30, dot_onehot_f32, input)`.
`crop_to_size_advisory` flagged the `[...,30,30]` intermediate as removable (max active grid
24×24), suggesting a crop to `[1,1,24,24]` (576 B, −324 B). This is a FALSE lever.

**Why the crop is impossible (durable, transferable to any "crop the mask plane" prompt):**
The NeuroGolf encoder pins BOTH the graph input and output to `[1,10,30,30]` (30×30 is the
canonical padded grid; the active 24×24 is a sub-region). A `Where(cond, A, B)` that produces
the `[1,10,30,30]` output requires its condition to BROADCAST against `[1,10,30,30]` — so a
spatial dim must be 1 or 30, never 24. A `[1,1,24,24]` condition cannot broadcast to 30×30.

The only way to use a 24×24 mask is to do the `Where` in 24-space and Pad the RESULT to 30×30:
`Where(paint24, dot24, input24) -> [1,10,24,24] -> Pad -> [1,10,30,30]`. But the dot color is a
10-channel one-hot, so that intermediate is `[1,10,24,24]` F32 = **23,040 B** — 25× worse than
the 900 B BOOL condition. The 900 B 30×30 BOOL condition plane is the CHEAPEST possible way to
mask a multi-channel 30×30 output: it is single-channel (×1 vs ×10) and BOOL (×1 vs ×4 F32).

**General rule:** when the charged plane is the *condition* of the output `Where`, its extent is
forced by the OUTPUT tensor shape, not by the active grid. `crop_to_size_advisory` is blind to
broadcast constraints on Where conditions — treat its `[...,30,30]` flag as advisory only and
verify the plane is not a broadcast-pinned Where condition before believing the saving.

**Verdict:** task064 v12 = 9751 stands as the proven renderer floor. no_candidate (exact tie).
