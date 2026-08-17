# task396 hard record: DepthToSpace block-stamp prompt route is INFEASIBLE (task mismatch)

**Recorded:** 2026-07-12 by bo_ng_cld_worker1 (Claude lane).

## Live bar
- Authoritative bar: **cost 3378** from `python tools/ng_current_bar.py --task 396 --json` (authoritative_source = manifest, owner Boladi, updated 2026-07-09T02:03:48Z, not stale).
- Live SHA-256: `0f56efef5a391212c94950780a3d7245ef0bedf2ed96cd6a15b625e9c9ffc567`.
- The live graph == staged `task396_fused_f32_prefix_cost3378` == `Team_Best_Onnx/task396.onnx` (all three SHAs match). It is a **structural run-detection** graph, byte-equal-chained to the accepted team graph (NOT the hidden-bad count-hash).

## Attempted route (the prompt's suggested angle)
Prompt `Prompt_20260711T000000Z_6220_Task396_DepthToSpaceBlockStamp.md` framed task396 as a
"DepthToSpace block stamp" (pack a 2×2/3×3 motif into channels, use `DepthToSpace` as the cheap
expansion step; floor_est ~1200). The prompt itself states: *"The prompt was selected from cheap
ORT/ONNX primitives first, then matched to this task."* Corpus keywords that triggered the card:
block, checker, 3x3, copy.

## Priced fast-probe + kill-rule (BEFORE any build — per prompt instruction)
task396's actual rule (`LIVE_ALGORITHM.md`, `task396.md`, task JSON, and the live graph inspected):
**largest valid hollow frame-color rectangle recolor crop** — an Object_Based / Cropping / Object
Detection task, NOT a fixed tiling task.

- Rectangle **size** is data-dependent (H,W ∈ 4..8, decided at runtime from the input's frame-color
  run lengths) and **placement/crop** is data-dependent.
- `DepthToSpace` rearranges `[N, C·b², H, W] → [N, C, H·b, W·b]` with **block size `b` a STATIC
  attribute**. A static `b` cannot divide a runtime-variable rectangle geometry.
- There is **no repeated motif**: the output is a single hollow rectangle (perimeter = marker color,
  interior = marker cells else background), not a tiled/interleaved/checker pattern. So "pack a motif
  and expand via DepthToSpace" has no target to expand into.
- **Kill-rule fires** (verbatim from the prompt): *"if block sizes do not divide the target geometry
  or the packed motif is as large as the expanded canvas, stop."* Both conditions hold: no static
  block divides the runtime geometry, and any faithful "motif" would already be the full 8×8 crop
  (motif == canvas). The keyword hits (block/checker/3x3/copy) are coincidental.

## Why dead
DepthToSpace is a static reshape; the task needs a runtime rectangle search + dynamic crop + recolor.
No base ONNX feeds a *useful* DepthToSpace here. Library confirms: `lib_query.py op DepthToSpace` →
"no ORT row"; `search "rectangle crop frame marker prefix run"` → no hits. No prior art suggests a
DepthToSpace formulation.

## Remaining hypothesis
None for DepthToSpace. The only safe, generalizing formulation is the existing horizontal/vertical
run-detection graph. Its per-tensor floor proof (why 3378 is the safe wall, and why the cheaper
bbox/count geometry is priced-out + hidden-bad) is in
`hard_records/formulation_floor_3378_wall.md`.
