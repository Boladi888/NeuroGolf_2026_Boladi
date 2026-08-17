# Second confirmation: "color-index collapse SWEEP" false floor + a fast arithmetic tell for `floor_est = best/N` estimates

**Source**: task208 (`bo_ng_cld_worker2`, 2026-07-11). Refreshed LIVE **3752**
(Boladi, manifest sha `df3c70f1…`, == staged `task208_f16seedscale_cost3752`).
**DROP / floor re-confirmation**, no new win. This CORROBORATES the existing rule
in `DONE_2026-07-10T202012Z_argmax_indexplane_is_int64_costtrap_coloridx_collapse_falsefloor.md`
(task064) — same "SWEEP prompt targets a per-color carrier the live best already
dissolved" trap, on a different task and a different forced component. Merge, don't
duplicate.

## What's new vs the task064 packet (two additions)

1. **A 10-second arithmetic tell for a FALSE `floor_est`.** task208's prompt gave
   `floor_est ~720, class SWEEP, gap 5.21x`. Note `3752 / 5.21 = 720.2`. The
   "floor" is literally `live_cost / class_gap_multiplier` — a mechanical EV
   estimate with ZERO grounding in the graph's forced tensors. **Before working a
   SWEEP target, divide `live_cost` by the quoted gap; if it reproduces the
   `floor_est`, the floor is unproven-by-construction** and you must re-derive it
   from forced components, not chase it. (task064's `best/9` and task254's estimate
   were the same shape.)

2. **The forced component for a COPY/PAINT-into-canvas rule is the output-tail
   broadcast mask, not a per-row projection.** task064's forced plane was a
   per-row F32 projection (ray extent). task208 is different: the rule paints a
   hollow-rect frame into empty regions, output `[1,10,30,30]` F32 one-hot, final
   op `Where(mask30[1,1,30,30], paint_onehot[1,10,1,1], input[1,10,30,30])`. The
   condition MUST be `[1,1,30,30]` to broadcast against the 10-channel output ⇒
   **`mask30` BOOL = 900 B is structurally forced, and 900 > the 720 target by
   180 B**. Any SWEEP est below the forced output-tail mask is impossible on its
   face. (Confirmed the collapse is already done: live graph's only `[1,10,·,·]`
   intermediates are `counts10`/`paint_onehot_f32` = two 40 B `[1,1]`-spatial
   scalar reductions; every spatial tensor is single-channel. No carrier to delete.)

## Generalized rule of thumb (extends the task064 rule)
For any `[1,N,H,W]` one-hot output, the final `Where`/select forces a
`[1,1,H,W]` broadcast mask = `H*W` bytes (BOOL) as an irreducible output-tail
floor — 900 B at 30×30. So `floor_est < H*W` for a per-cell recolor/paint/copy
task is false by inspection, regardless of the gap multiplier. Combine with the
task064 checks: (a) if the live best already carries color as scalar
ArgMax/moments (no spatial index plane) the carrier is gone → DROP; (b) never
build a spatial `ArgMax` index plane as a cost play (INT64, 4608–7200 B).

## Sorter note: merge into the existing color-index-collapse false-floor rule (the task064 DONE / its integrated home in `cost_floor_reference.md` or `by_symptom.md`) — add task208 as a SECOND exemplar, and add the two heuristics above: (1) `floor_est == live/gap ⇒ unproven mechanical estimate`; (2) `[1,N,H,W] one-hot output ⇒ forced [1,1,H,W] broadcast mask = H*W B output-tail floor`.
