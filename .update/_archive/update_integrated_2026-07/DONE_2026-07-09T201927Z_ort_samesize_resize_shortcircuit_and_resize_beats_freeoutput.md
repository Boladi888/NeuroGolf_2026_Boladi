# Transferable insight — (1) ORT short-circuits same-size Resize (kills the "direct-input" dream); (2) Resize BEATS free-output-Einsum for integer nearest-neighbour block expansion

**Source**: task289 batch5608 (`bo_ng_cld_worker1`, Claude, 2026-07-09). Audited LIVE **324** (Sho,
Resize family). Floor re-confirmation (WALL @ 324), not a new win. **Companion / counterpoint** to
`DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md` (task254)
and `DONE_2026-07-09T184440Z_gather_index_beats_free_output_for_permute_tile.md` (task249) — same
"FreeOutputPadTail/color-index collapse" sweep prompt, same `floor_est ≈ best/2.5` heuristic, but a
**different already-optimal op** (Resize here) and one genuinely new ORT mechanism.

## NEW mechanism 1 — ORT Resize short-circuits when out_spatial == in_spatial
The `floor_est ~129` dream for task289 (block-expand the 3×3 source by k) was a **direct-input
Resize**: `output[n,r,c] = input[n, floor(r/k), floor(c/k)]` via
`Resize(input[1,10,30,30], roi=[0,0,1/k,1/k], sizes=[..30,30], tf_crop_and_resize, nearest, floor)`.
Input free + output free ⇒ charged ≈ k-detection only (~98). **It does not work:**
- Real gate `ng_gate visible --task 289` = **0/268**. ORT returns `output == input` UNCHANGED on all
  268 whenever `out_H == in_H` (30==30), **ignoring roi / scales / mode / nearest_mode entirely**.
- The short-circuit is **shape-based**, not scale-based: `scales=1.0166` (output still 30) still
  returns input; `scales=0.9834` yields output size 29. So you cannot defeat it by nudging scales at
  30→30. Verified across `sizes` and `scales` inputs, floor/round_prefer_floor.
- **Consequence**: a Resize used for expansion needs a spatially-`<30` input. Minimum working for a
  3×3 source is **`[1,10,4,4]` U8 = 160 B** (a 3×3 pre-Resize fails 0/268 — no zero-border row and ORT
  nearest tf_crop_and_resize does not apply `extrapolation_value` on this path). The working recipe
  (Sho 324): build the small U8 one-hot via a label round-trip (Conv neg-pad `[0,0,-27,-27]` crops the
  3×3 + collapses channels → `idx+1`; quadratic QLinearConv `logit[n]=1-(idx-n)^2` with `pads=[0,0,1,1]`
  gives `[1,10,4,4]` U8 + zero border), then `Resize 4×4→30 @ scales=7.5`, `roi_end = 9.8/k` (9.8 ≈
  29/3=9.667 nudged UP so the background boundary floors to index 3 (=0 border) robustly instead of
  landing on the exact 3.0 float boundary — a naive `roi_end=29/(3k)` fails only k=5 on that boundary).

## NEW mechanism 2 — Resize BEATS a free-output Einsum here (opposite of task254/task249)
task254/task249 concluded "pad-tail/label rewrites regress because the best is ALREADY free-output."
task289 is the mirror image: the free-output Einsum is the LOSER, and a **Resize** is the champion.
- Free-output Einsum (`bnxy,xp,yq,kpr,kqc,k→bnrc`, expand_reuse) = **529**: memory 69 but a forced
  k-dependent expansion **bank `expand[5,3,30]` = 450 params** (or a runtime-computed `[3,30]` F32 =
  360 B; params count ELEMENTS so U8/F16/sparse don't help; F32-input locks the Einsum to F32).
- Resize = **324**: memory 267 (dominated by the `[1,10,4,4]` U8 = 160 B) + params 57. **Resize is
  k-general — the k-dependence lives in a cheap dynamic `roi_end` scalar, so there is NO per-k bank.**
  That is exactly why it beats the Einsum: it dissolves the 450-param placement basis into ORT's
  built-in nearest-neighbour upsampling.
- Low-rank/CP factoring of the Einsum bank is dead: grader = `np.array_equal((out>0), target)`
  (`run_raw` normalises `(out>0.0)`), so off-band must be **≤0**; an approximate bilinear basis
  contaminates off-band via cross-terms → off-band positives (cp_rank11: 0/268). Direct pair rank ≥41.

## Nuance — "background needs an explicit channel-0 plane" is grid-bounds-dependent
task254's DONE says background (color 0) needs an explicit hot ch0 plane. True only for color-0 cells
**inside** the target grid bounds. task289's expected output is `3k×3k` inside a 30×30 canvas; the
encoder (`ng_harness/encode.py`) writes one-hot **within grid bounds** (black→ch0=1) and **all-zero
outside**. So black cells inside the 3k×3k region DO need ch0=1, but everything outside 3k×3k is
all-zero (no channel needed) — and an all-zero-background candidate passes truth 200000/200000. Always
decode a real target (`encode_grid`) to see which regime you're in before assuming a ch0 plane.

## Rule of thumb (adds to the task254 rule)
For integer nearest-neighbour upsample / block-expand by a runtime factor k: **prefer a small-input
Resize (scales≠1) over a free-output Einsum** — it is k-general and avoids the O(k·W) placement bank.
But you must build a spatially-`<30` (≥4×4) charged U8 one-hot to feed it, because ORT will not honour
a same-size (30→30) crop-ROI Resize. If that pre-Resize tensor is itself expensive, the Einsum bank may
win — measure both. Do NOT trust a `floor_est ≈ best/2.5` "SWEEP" estimate that assumes the pre-Resize
`[1,10,H,W]` is removable; it is forced by the ORT short-circuit.
