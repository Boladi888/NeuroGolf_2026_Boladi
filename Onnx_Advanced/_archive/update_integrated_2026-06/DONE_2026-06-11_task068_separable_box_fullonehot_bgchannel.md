# task068 — deep-separable box rebuild + the full-one-hot ch0 trap (worker-2, 2026-06-11)

**Result:** task068 12589 → **1423** byte-equal-vs-LIVE (200266/200266, divergences=0; truth
200000/200000; visible 266/266; profile==static==1423). Staged
`staged_submits/task068_separable_einsum_cost1423/`.

**Rule:** unique color (count==1 among nonzero) → 3×3 box centered on its cell, border=color 2,
center=unique color. All grids 10×10; box never reaches edge.

## Transferable lesson — free-output one-hot MUST include the background channel (extends #109)

When you rebuild a `[1,10,30,30]` one-hot via a free-output Einsum / broadcast, the graded
comparison (`ng_gate visible` → `mode_visible`, and the official `convert_to_numpy`) is
**byte-equal on the FULL one-hot including ch0=1 for every background cell inside the active
grid**. A "sparse" output that only lights the colored channels (here ch2 + ch_unique) scores
**0/266** even though the color-index decode looks perfect and a naive `>0` argmax test passes.

Fix (the task234 background pattern, now confirmed on a 2nd task): add the background channel as
its own separable term `ch0 = active − colored`, where `active = g_r ⊗ g_c` is the active-grid
indicator. When the grid size is fixed (here always 10×10, and LIVE hard-codes the 10×10 crop),
`g_r = g_c = [1]*H + [0]*(30−H)` is a **constant** rank-1 term — one extra einsum operand row, no
runtime cost beyond [3,30] vs [2,30] factors. Combine terms by shared (R,C): box and active both
touch ch0/ch2 via signed channel selectors `K=[e2−e0 ; Ku−e2 ; e0]`, so 3 rule-pieces (box,
center, background) stay 3 einsum terms, not 5.

Build-your-own-oracle note: verify against the **full one-hot tensor** (`encode_grid(output)`),
not a color-index decode — the color-index decode hides the missing ch0 and gives a false 266/266.

## is_unique cheaply
`is_unique = Cast(Equal(ReduceSum(input,[2,3]), 1)) * ch_mask` (ch_mask kills ch0). Replaces the
LIVE Relu-triangle `r0−2r1+r2` (8 ops → 3 ops, −270 B of intermediates here). Sidesteps the
1-ch0 trap because it reads per-channel counts, not a `1−ch0` foreground.

## Floor honesty
1423 is the floor for the 3-term einsum formulation (R,C forced [3,30]; the two input-marginal
Einsums forced F32, 240 B). #119 claims Aditya's task068 at ~150 params via an abstract
position-basis `P2[2,30]` reused 6× — not reproduced; placing an arbitrary (r*,c*) through a
small basis is the open lead.
