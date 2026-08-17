# bo_ng_worker1 scratch - task066

Current authoritative bar: profile/live cost 2689 from
`python tools/ng_current_bar.py --task 66 --json`, source `manifest`,
sha256 `6eb4f4fe602493ef95da9e8e45df7c9281f2b210da93c711a603dd5b2fa7731e`.
The task-local `task066_idxshare_cost2689` candidate is now live, so it is a
tie only and not reviewer-ready.

Rule: input has background 0, walls 8, a color-3 source domino, and a color-2
target domino with matching orientation. Output preserves the input grid and
paints a color-3 orthogonal connector from source toward target. The bend is
selected by wall/clearance tests on the canonicalized vertical form, then
transposed back for horizontal cases.

Output-size relation: same active HxW as input, embedded in the fixed
NeuroGolf `[1,10,30,30]` one-hot output. The final output node is free; only
the intermediate selector vectors/slices and initializers count.

Object/color roles: color 3 is both source and paint color, color 2 is the
target anchor, color 8 is an obstacle/wall and endpoint cue, color 0 is
background.

Falsification cases:
- Horizontal source/target cases must route through the transpose/canonical
  path and then restore orientation.
- 20x20 grids can need inner coordinate 18; shortening the inner scan below 18
  is unsafe.
- Target-side and source-side wall-line reads are load-bearing; visible/arc-gen
  failures are expected if either clearance leg is dropped.

Families to price:
- Param-bank factor/dedup: rank-test and direct rewrite of the final
  `seg_rank/rank_src/rank_tgt` and selector initializers. Target charged object:
  params under the current 258, especially the 120-element rank bank.
- U8/F32 leg shave: try to reduce or delete the two 72B dynamic F32 wall-line
  slices, or avoid the two 4x30 F32 casts feeding final `Einsum`. Likely hard
  because `Einsum` is F32-locked by the graph input and ORT rejected mixed
  UINT8/FLOAT final operands in the previous attempt.
