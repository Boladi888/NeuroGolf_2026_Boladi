# task138 — two transferable insights (worker bo_ng_cld_worker1, 2026-06-30)

Where: task138 (Frame-raycast-crop). Build script
`The_400_Tasks/Task_by_Task/task138/prototypes/build_task138_robust_u8.py`; analysis in
`The_400_Tasks/Task_by_Task/task138.md` (CRITICAL + Floor proof sections).

## Insight 1 — U8 color canvas defeats the "F32-canvas-mandatory" floor proof

For object/color-index tasks that re-emit per-cell colors from the `[1,10,H,W]` F32 one-hot input,
the prevailing floor argument was: "ORT forces Conv/Einsum out dtype = F32, so the per-cell color
canvas must stay F32 (4 B/cell) ⇒ floor is large." (task138 old proof: floor 13,204; cf
techniques.md §8b task400.)

**This is escapable.** Collapse channels with a Conv (F32 out, forced) producing the *small*
collapsed canvas, then immediately **Cast that canvas to U8** and run the ENTIRE downstream pipeline
in U8/BOOL (1 B/cell). You pay ONE F32 plane once (the collapsed canvas, ~2.6–2.8 KB at the
crop-source size), not F32 across every working plane. On task138 this took the robust floor from
13,204 (v9, all-F32-ish) to **11,114** (robust, fully 200k-correct + edge-robust 120/120), and the
edge-cropped variant to 10,791. The Cast is cheap because it's on the *collapsed* `[1,1,H,W]` canvas,
NOT a full-input `[1,10,H,W]` cast (which would be 9k–18k and lose).

Key sub-points:
- `Cast(F32→U8)` AFTER the channel collapse, BEFORE any gather/crop. Gather/Where/ArgMax/ReduceMax/
  Equal/Pad/LessOrEqual all run on U8.
- One-based-ish color trick: `color_w = arange(10)-255`, `color_b=255` ⇒ color c→c, background→0,
  zero-padded (outside grid)→255. The 255 is a FREE per-cell sentinel: route out-of-crop gather
  indices to a guaranteed-zero input row/col (=255) so the final `Equal([0..9])` makes them all-zero
  output — no extra inside-mask Where, no extra plane.

## Sorter note: Onnx_Advanced/techniques.md §8b — extend with "Cast collapsed canvas to U8 before crop/gather; pay the F32 plane once, not per-plane" and the color_w=arange-255/b=255 free-255-sentinel trick. Cross-link the ORT dtype facts in WORKER_PROTOCOL digest line #5.

## Insight 2 — GRADER-GAP via crop UNDERSIZE: dummy-pass, real-input-crash

The shipped team-best task138 (manifest cost 10,377) sizes its crop/shift gathers to 22×22 but the
generator-proven crop bound is 24×23. `score_v2`'s dummy run uses an input whose frame is small, so
the oversized gather never fires → it scores 10,377 and looks like the team best. **On real inputs
it throws** `Gather idx=22 out of [-22,21]`:
- `ng_gate visible` (official_pairs incl arc-gen) → CRASH
- `profile_real_cost --arcgen 266` → CRASH
- `ng_gate ab/battery` for ANY candidate → CRASH (the harness must run the broken LIVE as reference)

⇒ near-certain Kaggle hidden-0 for that task, while the local manifest credits full points. This is
the same family as the §GRADER-GAP / task225 reprice warning but the failure mode is **shape
under-provisioning** (gather index bound), not dynamic-shape pricing. Lesson: the mandatory
REAL-INPUT PROFILE (`profile_real_cost.py`) would have caught it instantly — a candidate that
crashes the profiler is never a ship, regardless of `score_v2` cost. Also: when LIVE itself crashes,
`ab`/`adversarial`/`battery` (all byte-vs-LIVE) are unusable — `truth` (vs independent ARC-GEN) is
the only valid correctness gate.

## Sorter note: Onnx_Advanced/traps.md §GRADER-GAP — add "crop/gather UNDERSIZE dummy-pass" sub-case: a static graph can pass score_v2 (dummy input) yet crash on real inputs when an index range is sized below the generator-proven bound; always run profile_real_cost; if LIVE crashes, ab/adversarial/battery are invalid and truth is the only correctness gate.
