# Dead route: color-index-collapse "SWEEP to ~720" (false floor)

- **Live bar:** 3752 (manifest authoritative, owner Boladi, sha256
  `df3c70f1cc72907e6d92570be3d511d58249916e836bdf8b35bdf24a87ceedac`,
  updated 2026-07-10T22:11:58Z). Re-measured this session: `score_v2` = 3752
  (mem 3661 + params 91); `profile_real_cost --arcgen 266` static=3752
  profile=3752 (no fiction). Live == staged `task208_f16seedscale_cost3752`
  (byte-identical). `ng_safety` = WARN (1× `WARN_X86_QUANTIZED_OP_FAMILY`
  for QLinearConv, 0 failures).
- **Attempted route (prompt Direction + Fast probe):** "color-index collapse
  sweep — perform spatial logic on one index channel, use Gather/Equal only at
  the boundary; delete charged carrier/canvas/template." Target `floor_est ~720`,
  class SWEEP, claimed gap 5.21x.
- **Why the target is a FALSE FLOOR (the 720 is not grounded):**
  - `720 == 3752 / 5.21` to within 0.2 B. The "floor_est ~720" is a mechanical
    `live_cost / class_gap_multiplier` estimate, **not** a forced-component
    floor. It has no relationship to this graph's actual charged surfaces.
  - The grader output contract is **[1,10,30,30] F32 one-hot** (don't-do log:
    21×21 / no-pad output probes fail live 0/266 on raw shape mismatch). The
    final op is `Where(mask30[1,1,30,30], paint_onehot[1,10,1,1], input[1,10,30,30])`.
    The condition `mask30` **must** be `[1,1,30,30]` to broadcast against the
    10-channel output ⇒ **900 B BOOL, structurally forced**. That single forced
    tensor (900) already exceeds the entire 720 target by 180 B. No correct
    graph for this rule can price at 720.
- **Why the charged component the attack targets is ALREADY GONE:** inspected the
  live 3752 graph (47 nodes, `inspect_graph.py` + `largest_intermediate.py`).
  The ONLY `[1,10,·,·]` intermediates are `counts10` and `paint_onehot_f32`,
  both `[1,10,1,1]` = 40 B **per-color scalar reductions** (ReduceSum over the
  spatial dims / a Where on those counts). Every spatial tensor is single-channel:
  `ch0_inner17`[1,1,17,17] F32=1156, `mask30`[1,1,30,30] BOOL=900,
  `mask19_u8`+`mask19`[1,1,19,19]=722, `bg_u8`[1,1,17,17]=289,
  `bg_score17`[1,1,16,16]=256, `kernel`/`paint` 7×7 ≤49. There is **no per-color
  spatial carrier / canvas / template to delete** — the color-index collapse the
  Direction asks for is 100% already realized in the live graph. Per the
  2026-07-10 workflow rule ("if the charged component is already gone, return
  DROP with the refreshed floor note"), this is a DROP, not a rung to shave.
- **Forced-component floor (refreshed, at 3752):** total intermediate 3,655 B is
  dominated by four surfaces, each floored by a hard constraint:
  1. `mask30` BOOL 900 — must be [1,1,30,30] to broadcast the [1,10,30,30]
     one-hot output; already at the 1 B/cell floor.
  2. `ch0_inner17` F32 1156 — the empty-region detection read. `Slice` preserves
     the F32 input dtype (library #5); casting the full input to U8 first = ~9000
     (loses); no slice+cast fusion op; QLinearConv needs U8/F32 input anyway.
  3. `mask19_u8`+`mask19` = 722 — the 2-placement render pair. QLinearConv must
     emit U8; final `Where` needs BOOL. The 2 placements are a data-dependent
     rank-4 union (ALWAYS 2), so a separable row/col render is impossible
     (see `separable_convt_rowcol_union.md`: 15/266 visible + 5794 cost).
  4. `bg_u8` 289 + `bg_score17` 256 — detection U8 + score map; score-grid trims
     are provably lossy (see `score_grid_row_col_trim.md`: hidden-bad, anchors
     span full 16×16).
- **Genuinely-different assemblies BUILT and MEASURED (2026-07-11 second pass,
  `build_task208_w2_altassembly_probes.py` + `largest_intermediate.py`):**
  | assembly / dominant intermediate | bytes | vs live 1156 read |
  |---|---|---|
  | Direction's "one index channel" — `ArgMax(input,axis=1)` INT64 [1,1,30,30] | **7,200** | +6,044 |
  | alt "one index channel" — Einsum weighted collapse F32 [1,1,30,30] | **3,600** | +2,444 |
  | live detection read — `Slice(ch0)` F32 [1,1,17,17] (+U8 289) | **1,156** | baseline |
  | "cast whole input to U8 first" — U8 [1,10,30,30] | **9,000** | +7,844 |
  The Direction's collapse is not merely already-done — **re-introducing any
  full-grid index plane is a REGRESSION** (3,600–7,200 vs the 1,156 channel-0-only
  read). ONNX `ArgMax` has no dtype attr → INT64 (task064 lesson, 2nd confirmation).
  The live graph reads ONLY the background channel; it never pays a 10→1 collapse.
- **Two more independent from-scratch formulations, both priced to a wall:**
  1. Index-plane + Gather/Equal boundary (the Direction's fast probe) — the
     output boundary still forces the 900 B mask30; and the index plane itself is
     3,600 B (F32) / 7,200 B (INT64), MEASURED above. No path < 720; strictly
     worse than live. WALL.
  2. Separable row/col render to kill the 722 render pair — already dead
     (`separable_convt_rowcol_union.md`): pairing loss ⇒ 15/266 visible, and it
     costs MORE (5794). WALL.
  (Third lever, score-grid trim, dead in `score_grid_row_col_trim.md`.)
- **Forced floor written with numbers (cost_floor_reference.md methodology, line 235
  "price every op family against the decision core"):** the [1,10,30,30] Where-overlay
  output forces `mask30` BOOL 900 (cost_floor_reference §"canvas=900 B floor" line 165
  + "Where-overlay tax" line 200 + "Where-condition broadcast pin" task064 line 206).
  Detection read F32 1156 (measured minimal; every alternative 3,600/7,200/9,000
  measured worse). Render pair 722 + detection U8/score 545 (QLinearConv U8-in/U8-out +
  BOOL-for-Where, correctness-floored). 900+1156+722+545 = 3,323 forced + overhead
  (kernels 123 / counts 80 / scalars) ≈ 3,655 intermediate + 91 params = **3,752**.
- **Exact "gate":** no candidate built — the fast probe is priced-walled before
  build. The output-tail forced mask (900 B > 720 target) is the priced wall;
  building a losing ONNX would be "proving a stale sweep" (2026-07-10 rule).
- **One remaining hypothesis:** none that reaches 720 — the [1,10,30,30] one-hot
  output contract forbids it. The only sub-3752 motion left is sub-byte scalar
  dtype reparametrization (the 3759→3754→3752 progression was exactly this: F16
  scalar shaves, byte-equal). Those are captain/peephole micro-shaves, not a
  floor break. **WALL at 3752.**

---

## 2026-07-12 bo_ng_cld_worker1 — independent re-derivation + NEW gated lever (input-read shrink)

- **Bar re-verified fresh (unchanged):** live 3752 (manifest, Boladi, sha
  `df3c70f1cc72907e6d92570be3d511d58249916e836bdf8b35bdf24a87ceedac`,
  2026-07-10T22:11:58Z) == staged `task208_f16seedscale_cost3752` (byte-identical).
  This session: `visible` 266/266 PASS; `profile_real_cost --arcgen 266`
  static=3752 profile=3752 (**no fiction**); `ng_safety` verdict WARN
  (n_failures=0, n_warnings=1 = `WARN_X86_QUANTIZED_OP_FAMILY` for QLinearConv).
- **Rule confirmed from data (train0/1/2):** copy a rare-color HOLLOW rectangle
  frame into an EMPTY (all-color-0) interior block; the placed frame's interior
  is an all-color-0 block in the input (train0 tgt interior rows14-16/cols10-11 =
  all 0, verified). Target location is data-dependent (src→tgt offset differs
  every instance), so detection genuinely needs the spatial channel-0 scan —
  it CANNOT be replaced by arithmetic from the scalar row-moments.
- **Exact detection geometry (from the live graph):** `Slice(input, ch0, rows/cols
  2:19)` = [1,1,17,17] F32 (1156). `bg_score17 = QLinearConv(bg_u8, k=5x5,
  pads=[0,0,3,3])` → [1,1,16,16] (score anchors = grid rows/cols 2..17). A hollow
  frame has inner_h,inner_w ≥ 2, so an anchor at grid row/col 17 has an interior
  touching grid **row/col 18** — the read must reach row/col 18. The 17×17 read
  (rows/cols 2:19) is therefore geometrically tuned, not slack.
- **NEW measured lever — input-read shrink 17→16 (was never gated before; priors
  only gated SCORE-MAP trims, `score_grid_row_col_trim.md`):** builder
  `prototypes/build_task208_w1_read16_shrink.py` shrinks `ends_ch0_inner17`
  [1,1,19,19]→[1,1,18,18] (16×16 read) and re-pads `bg_score17` QLinearConv
  [0,0,3,3]→[0,0,4,4] (score stays 16×16). Static cost would drop **3752→~3581**
  (−165 B: `ch0_inner17` 1156→1024, `bg_u8` 289→256). **Result: `visible`
  **225/266 (84.59%) FAIL** — 41/266 placements have an interior touching grid
  row/col 18, dropped to zero-pad by the shrink. The single largest tensor's
  −132 B lever is thus MEASURED-dead, not merely argued. Per protocol
  (one visible miss = certain hidden 0) no truth/ab was run. DEAD.
- **Verdict stands: WALL at 3752.** Four forced surfaces (independently re-priced):
  `mask30` BOOL 900 (output-overlay broadcast pin), `ch0_inner17` F32 1156 (F32
  spatial read, now MEASURED-forced at 17×17), render pair `mask19_u8`+`mask19`
  722, detection `bg_u8`+`bg_score17` 545 = 3323 forced + ~330 near-forced small
  (7×7 paint-frame 98, [1,10,1,1] counts/paint 80, 5×5 kernel 25, ~20 F16
  scalars) + 91 params. All F32 scalars are op-dtype-forced (ReduceSum/Einsum
  follow the F32 input dtype; QLinearConv y_scale must be F32). No strict-win
  candidate reachable; only further byte-equal scalar reparametrization remains,
  and the live best is already its F16-shaved endpoint.

---

## 2026-07-12 bo_ng_cld_worker1 PASS-2 (forced 2nd pass) — library-mined levers + PER-AXIS read-minimality proof

Ran the runner's mandate to find a genuinely different assembly by mining
`INDEX.md`→`techniques.md`/`traps.md` + BoBrain provider. Two on-point findings:

- **#131 (task102) EXTERNALLY CORROBORATES this floor.** techniques.md §131 documents
  the *exact* task208 pattern — "a task that PRESERVES the input and paints a detected
  region (`Where(mask,color,input)` overlay)": its floor "is dominated by FORCED
  structural costs, NOT the detection logic: an F32 input-slice + the 30×30 `Where`
  condition mask (~900 B forced — the task341/#129 pattern: a `Where` with a free
  full-input else-branch needs a full HxW condition even when the logical grid is
  smaller)." This is an independent library statement that our two biggest tensors
  (`mask30` 900 + `ch0_inner17` F32 read) are the forced floor, not a shaveable head.
- **#133 (task267) per-axis read-clip lever — APPLIED and CLOSED.** #133: "clip the
  read to the generator's spawn box on EVERY axis; byte-equal iff clipped cells are
  corpus-guaranteed background; don't trust a single-axis clip — verify with the scan."
  - **Corpus bbox scan (266 arc-gen + train/test):** painted frames (`output!=input`)
    span rows **[1,19]** × cols **[1,19]**; max frame row=19 (20 cases), col=19 (23
    cases); min row=1 (15), col=1 (21). The detection reads the all-color-0 INTERIOR
    (frame minus border) → interior span = rows/cols **2..18 on BOTH axes**. The live
    read `Slice(ch0, rows/cols 2:19)` = [1,1,17,17] is **exactly** the interior bbox —
    **zero slack on any axis.** task208 has clutter filling the whole grid (no sparse
    spawn box), so #133's clip does not apply here.
  - **Gated per-axis verification (distinct from pass-1's symmetric clip):** col-only
    clip `build_task208_w1_read_colclip.py` (slice ends→[1,1,19,18], conv pads
    [0,0,3,3]→[0,0,3,4], read→[1,1,17,16]=1088, would save ~85 B) → **visible
    243/266 (91.35%) FAIL** — the 23 misses match the 23 corpus col-19-frame cases.
    (Pass-1 symmetric 17→16 was 225/266.) The 1156 F32 read is now MEASURED-forced
    on EACH axis independently, exactly as #133 prescribes.
- **Other genuinely-different assemblies re-ruled-out:** projection/separable detection
  = the 2-placement non-separable outer-product trap (spurious corners; `separable_convt_rowcol_union.md`
  15/266); coordinate-broadcast render = the historical target-only stamp, MEASURED 14,256
  (≫3752) because scalar-corner extraction (ArgMax/GatherND INT64) costs more than the
  spatial render QLinearConv; nibble-pack #83 cannot run a box-SUM detection on a packed
  plane (only boolean/row-gather consumers survive packing). detect+render cannot fuse
  (a conv emits a scalar score per anchor, not a spatial frame → 2 convs forced).
- **Verdict unchanged: WALL @ 3752.** Now backed by (a) external #131 corroboration of
  the forced (mask30 900 + F32 read) core, and (b) a #133 per-axis corpus scan +
  gated col-clip proving the 1156 F32 read minimal on both axes. No candidate built
  wins; only byte-equal scalar reparam remains and the live best is its F16 endpoint.

---

## 2026-07-12 REVIEWER SIGN-OFF (bo_ng_cld_reviewer1, Claude lane) — WALL independently reproduced → DROP

Re-ran the load-bearing checks by hand (not from pasted numbers), `$env:NEUROGOLF_PYTHON`,
all foreground:
- **Bar:** `ng_current_bar --task 208` = live 3752 (manifest authoritative, Boladi, sha
  `df3c70f1cc72907e6d92570be3d511d58249916e836bdf8b35bdf24a87ceedac`). Live onnx
  `profile_real_cost --arcgen 266` = static 3752 / profile 3752 (no fiction); `visible`
  266/266 PASS. (Note: `ng_current_bar` team_best recompute reads 3746 — flagged stale,
  a cost-model drift on the byte-identical file; the authoritative profile is 3752. Moot —
  no candidate approaches either number.)
- **Graph inspected directly** (shape-infer + charged-tensor sort): free-output
  `Where(mask30[1,1,30,30], paint_onehot[1,10,1,1], input[1,10,30,30])` overlay — the
  `[1,10,30,30]` F32 one-hot output is FREE (#67/#81). Charged surface = `ch0_inner17`
  F32 1156 (input-tax spatial read) + `mask30` BOOL 900 (output-broadcast pin) + render
  pair `mask19_u8`+`mask19` 722 + detection `bg_u8`+`bg_score17` 545 + ~330 near-forced;
  sum_mem 3661 + params 91 = 3752. Worker's decomposition is exact.
- **Load-bearing read-minimality proof REPRODUCED:** regenerated `build_task208_w1_read_colclip.py`
  → sha `53927c0a8482f811bbca037c2bc8960175b99f574280d3dd006ac71a1825b7ad` (== worker's
  claim); my `ng_gate visible` = **243/266 FAIL** (23 col-19 counterexamples). Combined
  with the pass-1 symmetric 225/266, the 1156 F32 read is minimal on each axis. A
  small-sample FAIL is decisive (concrete counterexamples), unlike a small-sample PASS.
- **floor_est ~900 is fantasy:** it is `live/4.2` (mechanical STRUCTURAL class-gap), not a
  forced-component floor. The single forced `mask30` (900) alone ≈ that estimate, with
  four more forced surfaces stacked on top. Exactly the prompt's own escape hatch
  ("if recon proves the floor estimate is fantasy, return DROP with priced evidence").
- **No un-tried lever below 3752 nameable:** ColorIndexCanvas collapse (3600 F32 / 7200
  INT64 regression), separable render (15/266, 5794), coordinate render (14,256),
  nibble-pack (no box-SUM on packed plane), both read-clips (225/266, 243/266),
  score-grid trim (hidden-bad) — all measured-dead with priced/gated evidence. Two
  dominant tensors are forced by the fixed grader output contract and the
  information-theoretic input-tax. Remaining motion is byte-equal scalar reparam, already
  at its F16 endpoint.
- **Reviewer verdict: DROP — confirmed at floor, no strict gain.** Honest, thorough
  worker packet (2 gated passes, nothing overfit staged). Do not re-assign task208 for a
  cost win; a re-open needs a NEW output-contract insight (something that removes the
  900 B mask or the F32 detection read), not another sweep/peephole.
