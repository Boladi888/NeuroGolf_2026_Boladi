# task142 hard_record — sub-90 dead routes (2026-07-07, bo_ng_cld_worker1 / Claude)

**Live bar at audit:** cost **90** · sha `140f8984be503355003c64ebb46a6ed9d9bc0e03dfd25ab350a81952cd08180d`
(manifest, owner Vamshi, 2026-07-02; `ng_current_bar.py --task 142`).

**LIVE structure (the thing to beat):** single free-output Einsum
`bcrs,hp,rp,wq,sq->bchw`, ONE shared `[30,3]` F32 map named `m` used 4× →
`score_v2 = (macs 0, memory 0, params 90)`. Input `[1,10,30,30]` free, output
`[1,10,30,30]` free; the map is the entire cost. Rank-3 latent = the 3 distinct
input rows; symmetric mirror lets one map serve both selection (`rp,sq`) and
placement (`hp,wq`), and its identity block (rows 0–2) coincides with the
placement's first 3 rows so a single matrix does both.

## DEAD: sparse_initializer undercount (params 6 instead of 90)
The grader `calculate_params` counts a `sparse_initializer` as
`math.prod(values.dims)` = **nnz** (the `m` map has only 6 nonzeros). Probe
`prototypes/task142_sparse_probe.onnx` (COO `m`, 6 nnz) **runs correctly in raw
ORT** (produces the exact 6×6 mirror) and `calculate_params` returns 6 — BUT it
is UNSCOREABLE / UNSHIPPABLE, rejected at the scorer path (matches
`Onnx_Advanced/core.md:44-45`, `traps.md §8e`, task337 Floor-proof Lemma 4):
- **gate (c) sanitize_model** renames the Einsum's node input (`m→safe_name_0`)
  but does NOT rename `graph.sparse_initializer` names → dangling ref →
  `ORT load: INVALID_ARGUMENT "Node input 'safe_name_0' is not a graph input,
  initializer, or output of a previous node."` (reproduced first-hand.)
- gates (a) `check_model(full_check=True)` and (b) `shape_inference(strict)`
  also reject sparse-fed op inputs (documented).
- `_score_v2.score_network_v2(sparse_probe) → None` (confirmed first-hand).
- Adding `value_info` to satisfy shape-infer bills the map as **dense memory
  (360 B)** → far above 90.
A name-collision hack (naming the sparse init `safe_name_0` so the first
counter slot aligns) could survive only gate (c); (a)/(b) still kill it, and it
is a grader-bug exploit contrary to the team's ethical-play stance. **Do not
pursue.**

## DEAD: reindex without a map (Gather/Slice/GridSample)
Any Gather/Slice that rearranges rows/cols materializes a counted
`[1,10,30,30]` F32 intermediate = **36,000 B** (two axes → 72,000). GridSample
grid `[1,30,30,2]` = 1,800 (old LIVE 1,520). All ≫ 90.

## DEAD: crop-then-process
Any op touching the free input to crop emits a counted intermediate ≥
`[1,3,3,3]` = 108 B (tetra_code, old staged 258) or `[1,10,3,3]` = 360 B
(einsum-final 456). memory+params all > 90.

## DEAD: cost≤1 zero-param family (the prompt's stale premise)
Identity → visible **0/266**; Transpose(H,W) perm `[0,1,3,2]` → **0/266**. A
3×3→6×6 mirror-FOLD cannot be an attr-only axis permutation of a
`[1,10,30,30]→[1,10,30,30]` graph. Output is locked to `[1,10,30,30]`
(`encode_grid` pads to 30×30; harness compares via `np.array_equal`), so no
`[1,10,6,6]`-output shrink of the map is legal either (a Pad back to 30×30 =
36,000 B counted).

## Remaining hypothesis (only crack, currently closed)
The dense `[30,3]=90` is information-theoretically forced: describing which of
3 latent rows (or zero) each of the 30 output rows samples needs 30×3 stored
values, and the only sub-dense encoding (sparse nnz) is unscoreable. **No
scoreable sub-90 route known. Verdict: WALL at 90.**
