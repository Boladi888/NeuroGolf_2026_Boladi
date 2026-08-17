# task035 — directional ray-projection collapsed to single index plane + decode/output forced-component floor

**Where:** task035 (ARC-GEN `task_1f642eb9`). Builder `The_400_Tasks/Task_by_Task/task035/prototypes/build_scanfold.py`;
staged `staged_submits/task035_scanfold_cost7155/`. Cost 11,984 → 7,155 (−40%), byte-equal-LIVE 200266/200266, truth 200k 100%, profile==static.

## What I discovered

**1. "Border pixel projects color onto nearest pool-edge cell along its ray" collapses to: single F16 color-index plane + 4 MaxPool directional prefix-max scans + a boundary-write Where, with NO 10-channel one-hot carriers.**
The LIVE graph (70 nodes) carried four directional scan planes plus per-color one-hot tails. The whole rule is:
- `idx` = Conv weighted-sum decode (#35) on the **uncounted input**, asymmetric kernel `[1,10,21,1]` crops rows for free → `[1,1,10,30]`, Slice cols.
- `cval = where(idx==8, 0, idx)` (zero the pool/cyan and bg; colors survive as the ray carrier).
- `proj_dir = MaxPool(cval)` with a full-length kernel + asymmetric pad = directional **prefix-max** (a "ray").
- **Key simplification: the LIVE shift-before-pool is dead.** At a cyan cell `cval=0`, so inclusive prefix-max == exclusive prefix-max — drop all 4 shift Pad+Slice pairs.
- boundary = `cyan AND (neighbor in reverse-ray dir is NOT cyan)` (the pool edge facing the ray); paint with sequential `Where(boundary & proj>0, proj, cur)`. Sequential down→up→right→left == a max-combine of the 4 (verified 20k pairs), and sequential is cheaper (no extra Max temporaries).

**2. Forced-component floor for "recolor a few cells of a one-hot input" rules.** Two costs are FORCED and bound the floor far above a naive target:
- **Output one-hot ≥ ~1,000B**: a `[1,10,30,30]` one-hot BOOL output needs a `[1,10,10,10]` (1,000 BOOL elem) cropped one-hot before the free Pad. The #292 Where-as-output substitution trick does NOT apply when the substituted color **varies per cell** (the overlay would be full-canvas `[1,10,30,30]` = 9,000B, not a broadcast `e_k[1,10,1,1]`). The 10×10 one-hot sits exactly at the #81 crossover (10·h·w=1000 vs the 900B Equal-tail canvas), and the u8-cast needed to reach the canvas form eats the 100B difference → net tie.
- **Color-index decode ≥ ~1,200B**: single-axis Conv crop → `[1,1,10,30]` F32. Dual-axis conv crop = `[1,10,21,21]` = 4,410 params (>> mem saved). Reduce-first (`Mul(input,labels)→ReduceSum`) = 36,000B product. Crop-input-first = 4,000B. So 1,200B is the conv-decode floor.

This let me HONESTLY reject the prompt's ~1,400 target with a forced-component argument rather than "angles I tried failed": realistic floor ≈ 3,000–3,500B here.

## Why it matters
−40% byte-equal on a directional-scan family task; reusable recipe for any "edge pixel shoots a ray to the nearest blocker" rule. The dead-shift insight (cyan/blocker cell zeroes the carrier so inclusive==exclusive prefix-max) removes 4 Pad+Slice pairs from the canonical LIVE scan structure.

## What's missing from the library
- by_symptom.md §"4-8 directional Pad+Slice probes" + §"geometric ray/diagonal drawing" cover the Pad/Slice→MaxPool conversion, but NOT the **single-index-plane formulation** (avoid 10-channel carriers entirely by decoding to a color index first, scanning the index, rebuilding one-hot only at the end) nor the **"blocker cell zeroes the carrier ⇒ inclusive prefix-max needs no shift"** identity.
- The forced-component floor for "decode (≥1,200B single-axis conv) + one-hot output (≥1,000B) ⇒ ~3k floor" would be a useful entry in `cost_floor_reference.md`'s plane-floor taxonomy for recolor-a-few-cells rules.

## Sorter note:
- ADD to `by_symptom.md` (directional ray/scan section): single-color-index-plane formulation of ray-projection rules (decode→scan index→rebuild one-hot last) + the "blocker-cell-zeroes-carrier ⇒ inclusive==exclusive prefix-max, drop the shift" identity. Exemplar: task035 scanfold, 11,984→7,155 byte-equal, builder `Task_by_Task/task035/prototypes/build_scanfold.py`.
- ADD to `cost_floor_reference.md` (plane-floor taxonomy): recolor-few-cells-of-one-hot floor = single-axis conv color-decode (~1,200B) + cropped one-hot output (~1,000B); #292 substitution-Where only collapses the output when the substituted value is a per-cell-CONSTANT `e_k` (broadcast), not a spatially-varying color.
