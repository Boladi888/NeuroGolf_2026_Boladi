# task158 — two durable findings (generator-invariant plane fold + stamp-render separability)

Source: worker-158 session 2026-06-19. Builder `The_400_Tasks/Task_by_Task/task158/prototypes/build_task158_v111_nonbg_fold.py`; staged `staged_submits/task158_v111_nonbg_fold_cost24148_sha9349EA5A/`. Beat team-best v110 (25,447) -> 24,148 (-1,299), byte-equal to LIVE across full 200k AB.

## Finding 1 (techniques / #136 "derive don't recompute" family): generator-invariant lets out-of-grid cells self-exclude

A "non-background active cell" mask is commonly built as `non_bg = (label > 0) AND (label != bg)` = 4 bool planes (active_cell, is_bg, not_bg, And) + 2 casts. On grids where the ARC-GEN generator draws colors only from 1..9 (`common.random_colors`/`random_color` both exclude 0), the out-of-grid one-hot padding encodes to label 0, and in-grid cells are never 0. So a cell is non-background iff `label != 0 AND label != bg`, which collapses to:

```
is_bg_b = Equal(labels, bg)            # needed anyway for the bg map
nonbg_label = Where(is_bg_b, 0, labels) # bg -> 0; out-of-grid already 0
non_bg_u8 = Min(nonbg_label, 1)         # clamp the surviving 1..9 to 1
```

Two planes (`active_cell_b`, `not_bg_b`) vanish — exact, byte-equal, -1,300 B on a 26x25 grid. `Min(U8, 1)` is a valid ORT opset-13 clamp-to-{0,1}. The lever: when colors are guaranteed nonzero (verify over the corpus: `histogram[0]==0`), the `label>0` active guard is redundant with `label!=bg` after a zeroing Where — don't materialize the active plane.

## Finding 2 (confirms #133 separability caution on a STAMP task; #109/#118/#119/#132 NOT a fit here)

task158 paints rotated/scaled copies of a legend sprite at runtime-detected marker-square pairs. A worker prompt proposed dissolving the per-scale dense valid-score planes + conv render into one free-output separable Einsum (`tk,tr,tc->bkrc`). Assessed NOT-VIABLE, two independent reasons:

1. **Detection is irreducibly 2D.** A valid stamp = two marker squares at a fixed *diagonal* relative offset (corners (0,0)/(2,2) under 4 rotations x mag 1/2/3). A diagonal-offset coincidence is a rank>1 2D correlation — it does NOT factor into (row-condition AND col-condition), exactly the task093/task350 case #133 warns about. The valid-score plane cannot become an outer product R[t]⊗C[t].
2. **Placement is data-dependent.** Stamp positions are detected at runtime and the count varies 2-4 per input, so there is no build-time-fixed basis `M[h,H]` (which #118/#132 require). A runtime basis is the dynamic-shape construction that reprices UP (#109's own caveat).

So the dense valid-score planes + a translation-invariant QLinearConv render ARE the structural floor for this family — the convolution is the *correct* tool for data-dependent translation-invariant detect-and-place. Suggested library touch: add a one-line cross-ref in #132/#133 that "stamp-tiling free-output einsum needs build-time-KNOWN placements; data-dependent stamp positions => conv render is the floor, einsum is wrong (task158)."

## What was missing from the library
Nothing was wrong, but #133's separability caution is filed under render rules; a stamp-specific worker reflexively reaches for #132 (stamp-tiling einsum). A breadcrumb from #132 -> #133 for the data-dependent-placement case would have saved the feasibility detour.
