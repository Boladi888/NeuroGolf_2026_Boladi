# task023 — archived history (Attempts + Cost-analysis journey)

Moved from `task023.md` 2026-07-10 per WORKER_PROTOCOL task-md-size rule. Current status, cliff table, and floor proof remain in `task023.md`; the false-floor WALL proof is in `task023/hard_records/false_floor_1400_2026-07-10.md`.

## Attempts

- 2026-05-19 | crop canonical to active `9x11` window (`build_task023_crop9x11.py`) | visible/live `266/266`; canonical-equivalent only, not a hidden unlock | `16,025` from then-live `21,433` | Useful compression pattern, but zero-weight prompt asks for true structural rewrite.
- 2026-05-25 | exact-cover numpy oracle | `266/266` public/train/test/arc-gen | n/a | Confirms square-vs-bar decomposition rule.
- 2026-05-25 | `task023_structural_v1.onnx`: 9x9 local Conv/ReLU/Conv classifier trained from exact-cover synth, wrapped as color-channel output | validator-faithful visible `266/266`; synth gate `636/1000` grids (`cell_accuracy=0.9911803893`) | `122,419` (`memory=111,771`, `params=10,648`) | Not ship-ready; learned local classifier still misses ambiguous joins.
- 2026-05-25 | global candidate-feature MLP probe (`337` features: mask + square candidates + bar candidates, hidden `512,512`) | worse than v1: with 30k synth train, `463/1000` and `907/2000` broad synth grids | not exported | Do not pursue this exact dense-feature MLP shape unchanged.
- 2026-05-25 | deterministic exact-cover constraint propagation | public `266/266`; broad synth about `655/1000`, leaving unresolved ambiguous joins | not exported | Good future ingredient, but it needs branching or stronger tie-breaks.
- 2026-05-26 | `task023_exactcover_v2.onnx`: unrolled first-cell exact-cover search with UINT8 residual state, compact UINT8 square-origin slots, row-selected solution reconstruction, task-shaped area collection, shared H/V bar parent state, UINT8 output, and onnxsim cleanup | validator-faithful visible/canonical-live `266/266`; synth gate `1000/1000` (`seed=23023`) and `5000/5000` (`seed=24023`) | `317,260` (`memory=312,072`, `params=5,188`, `points=12.332523`) | Correct structural baseline; no submission made and `submissions/submission.zip` was not touched.
- 2026-05-26 | `task023_rowpack_v3.onnx`: same exact-cover search but residual is nine `UINT16` row bitmasks, with `UINT8` min-index arithmetic and impossible row-bit masks for off-board pieces | validator-faithful visible/canonical-live `266/266`; synth gate `1000/1000` (`seed=23023`) and `5000/5000` (`seed=24023`) | `145,501` (`memory=139,688`, `params=5,813`, `points=13.112062`) | Correct static exact-cover compression; still above the user-requested `100k`.
- 2026-05-26 | `task023_dynamic_v4.onnx`: exact first-cell branch search with fixed-cap `TopK` branch compaction after each depth (`3,9,24,40,48,40` caps), compact square-origin slots, row-packed residuals, and collection only at exact possible completion depths | visible/canonical-live `266/266`; synth gate `1000/1000` (`seed=23023`) | `66,579` (`memory=62,537`, `params=4,042`, `points=13.893856`) | Under-100k stepping stone. The earlier all-depth collection variant scored `95,034`; exact-depth collection removed most collection cost.
- 2026-05-26 | `task023_dynamic_v5.onnx`: v4 with collection only at the exact possible completion depth, redundant `bar_count` state removed (`bar_count = depth - square_count`), and tighter fixed caps (`3,9,19,30,32,24`) | validator-faithful visible/canonical-live `266/266`; synth gates `1000/1000` (`seed=23023`), `5000/5000` (`seed=24023`), `20000/20000` (`seed=25023`), and `20000/20000` (`seed=26023`) | `49,483` (`memory=45,443`, `params=4,040`, `points=14.190616`) | Current under-50k structural candidate. Very tight cap budget but stress-tested on the listed seeds.
- 2026-05-26 | `task023_dynamic_v6.onnx`: reduced-cap exact beam (`3,6,10,14,14,8`) plus explicit row-priority `TopK` tie scoring | visible/canonical-live `266/266`; synth gate still missed `999/1000` (`seed=23023`) | `31,625` (`memory=27,483`, `params=4,142`) | Below-50 stepping stone but not safe; the no-priority sibling scored about `29,218` and also missed `999/1000`.
- 2026-05-26 | `task023_origin_top3_v1.onnx`: tiny learned square-origin ranker (`10x10` patch, H24, support-gated top-3 origins, area-gated third square) | visible/canonical-live `266/266`; synth `4912/5000` (`seed=23023`) and `4905/5000` (`seed=24023`) | `16,650` (`memory=13,618`, `params=3,032`, `points=15.279835`) | First sub-20k artifact, but it is learned/heuristic and misses hard ambiguous joins.
- 2026-05-26 | `task023_origin_top3_v2.onnx`: H30/random-seed-tuned variant of the same learned origin ranker | visible/canonical-live `266/266`; synth `4921/5000` (`seed=23023`), `4924/5000` (`seed=24023`), and `19632/20000` (`seed=25023`) | `19,638` (`memory=15,994`, `params=3,644`, `points=15.114778`) | First H30 under-20 result. Useful cost target proof, not a replacement for exact v5 unless hidden-risk tolerance changes.
- 2026-05-26 | `task023_origin_top3_qv1.onnx`: same H30 ranker quantized with `QLinearConv`, UINT8 hidden/score maps, UINT8 `MaxPool` support gating, and F16 cast only for the 80-cell `TopK` | visible/canonical-live `266/266`; synth `980/1000` (`seed=23023`) and `4922/5000` (`seed=24023`) | `11,026` (`memory=7,322`, `params=3,704`, `points=15.691989`) | Onnx_Advanced QLinearConv guidance worked: same heuristic family at much lower cost, freeing room for small static choosers.
- 2026-05-26 | `task023_origin_top3_qv0.onnx`: narrower H24 version of the quantized top-3 ranker | visible/canonical-live `266/266`; synth `4913/5000` (`seed=23023`), `4899/5000` (`seed=24023`), and `19614/20000` (`seed=25023`) | `9,808` (`memory=6,728`, `params=3,080`, `points=15.809046`) | Current best under-10 artifact. Public-perfect and cheap, but weaker than the under-15 top-5 chooser on broad synth.
- 2026-05-26 | `task023_origin_top3_qv2.onnx`: wider H72 quantized ranker | visible/canonical-live `266/266`; synth `2953/3000` (`seed=23023`) and `2944/3000` (`seed=24023`) | `19,552` (`memory=11,480`, `params=8,072`, `points=15.119167`) | Wider learned ranker spent the recovered QLinearConv budget but did not improve robustness.
- 2026-05-26 | `task023_origin_top5_qchoose_v1.onnx`: H30 quantized ranker plus static top-5 non-overlap chooser (`C(5,2)`/`C(5,3)` combos, overlap table via `GatherND`, best valid combo by score) | visible/canonical-live `266/266`; synth `4945/5000` (`seed=23023`), `4946/5000` (`seed=24023`), and `19753/20000` (`seed=25023`) | `19,453` (`memory=9,286`, `params=10,167`, `points=15.124243`) | Current best under-20 artifact. Still learned/heuristic, but it fixes many top-rank overlap failures and is the strongest sub-20 candidate so far.
- 2026-05-26 | `task023_origin_top6_qchoose_v1.onnx`: top-6 variant of the same chooser | visible/canonical-live `266/266`; synth `2972/3000` (`seed=23023`) and `4946/5000` (`seed=24023`) | `20,738` (`memory=10,531`, `params=10,207`, `points=15.060277`) | Over the requested `20k` cap and not clearly better than top-5; keep as a boundary probe only.
- 2026-05-26 | `task023_origin_top5_qchoose_v2.onnx`: same H30 quantized top-5 chooser, but the `80x80` overlap lookup table is replaced by dynamic row/column overlap checks (`Gather` row/col tables, `Add`, `LessOrEqual`) using INT32 row/col arithmetic | visible/canonical-live `266/266`; synth `4945/5000` (`seed=23023`), `4946/5000` (`seed=24023`), and `19753/20000` (`seed=25023`) | `13,974` (`memory=10,046`, `params=3,928`, `points=15.455046`) | Current best under-15 artifact. Same behavior as top5_qchoose_v1 at much lower param cost; still learned/heuristic rather than exact-safe.
- 2026-05-26 | `task023_origin_top6_qchoose_v2.onnx`: top-6 chooser with the same dynamic row/column overlap checks | visible/canonical-live `266/266`; synth smoke `987/1000` (`seed=23023`) | `15,744` (`memory=11,776`, `params=3,968`, `points=15.335785`) | Just over the requested `15k` cap; useful boundary probe, not the under-15 pick.
- 2026-05-26 | `task023_origin_top4_qchoose_v1.onnx`: H24 quantized ranker plus dynamic top-4 non-overlap chooser | visible/canonical-live `266/266`; synth smoke `986/1000` (`seed=23023`) | `11,608` (`memory=8,330`, `params=3,278`, `points=15.640550`) | The chooser improves H24 smoke accuracy but misses the requested `10k` cap; keep H24 top-3 for under-10.

## Cost analysis

`structural_v1` cost is dominated by the first learned conv:

- `w1`: `[128, 1, 9, 9]` = `10,368` params.
- Total structural-v1 score: `{"macs":0,"memory":111771,"params":10648,"cost":122419,"points":13.284795}`.

`exactcover_v2` cost is dominated by branch-state intermediates, not params:

- Search depth is fixed at 7 pieces; largest branch layer is `3^7 = 2187`
  states over 99 cells.
- BOOL residual/square state lowered the cost from the first exact F32 version
  (`67,566,694`) to `8,456,898`.
- `GatherElements` fit checks and `ScatterElements` state updates removed the
  full `99x99` piece-mask tables, lowering cost again to `5,995,469`.
- Prefix-pruning by legal piece-count families (`2S+2B`, `2S+3B`, `3S+3B`,
  `3S+4B`) lowered the state rows from `1 + 3 + ... + 3^7 = 3280` to `1280`.
- Direct `ArgMax(residual)` after a cast, area collection after group-level
  reduction, FP16/UINT8 mask reductions, scalar branch validity, UINT8 residual
  state/output, dead tensor cleanup, skipping final residual/concat state,
  task-shaped complete-count collection (`2/3` squares and `2..4` bars), and
  shared H/V bar parent gathers plus onnxsim cleanup brought exactcover-v2 to
  `722,142`.
- Collection then stopped materializing every valid row-by-cell square mask.
  Because the oracle/generator require a unique exact cover, collection now
  selects the first valid completed row, gathers that row, and gates it by the
  input area. This lowered cost to `517,832`.
- Square state is now compact: branches carry up to three UINT8 square-origin
  slots instead of a `[rows,99]` BOOL square mask. The selected row is
  reconstructed into a 99-cell mask only during collection. This lowered the
  current exactcover-v2 cost to `317,260`.
- Row-packing the residual as nine `UINT16` row masks, using row/column lookup
  tables for the first residual cell, eliminating duplicate validity tables,
  and doing min-index add in `UINT8` lowered the static exact solver to
  `145,501`.
- Fixed-cap branch compaction with `TopK` keeps only the best valid rows after
  each depth while preserving static shapes for the official scorer. Caps
  `3,9,24,40,48,40` lowered the exact structural candidate to `95,034`.
- Collection should only run at the depth where a count family can be complete:
  `2S+2B` at depth 4, `2S+3B`/`3S+2B` at depth 5, `2S+4B`/`3S+3B`
  at depth 6, and `3S+4B` at depth 7. Removing dead collection at all other
  depths lowered v4 to `66,579`.
- `bar_count` is redundant once depth is known; carrying only `square_count`
  lowered the fixed-cap graph to `64,316`.
- Tightening caps to `3,8,18,28,32,24` reached `48,039` but missed
  `1/5000` (`seed=24023`) and `2/20000` (`seed=25023`). The current
  `3,9,19,30,32,24` caps score `49,483` and passed the listed gates.
- Reducing the exact beam to `3,6,10,14,14,8` gets into the `29k-32k`
  range, but even with explicit row-priority tie scores it still misses the
  synth gate. Exact-beam under-20 likely needs a new representation, not just
  cap shaving.
- A learned square-origin shortlist can fit under 20k: H24 scores `16,650`;
  H30 scores `19,638`. Both are public-perfect, but broad exact-cover synth
  stays around `98-98.5%`, so this is currently a cost experiment rather than a
  safe structural unlock.
- Onnx_Advanced/Research QLinearConv guidance applies cleanly here. Replacing
  the two F16 Conv layers in the H30 origin ranker with `QLinearConv` and using
  UINT8 `MaxPool` for 2x2 support gating lowered H30 from `19,638` to `11,026`
  while keeping visible `266/266`.
- The recovered budget is better spent on a small structural chooser than on a
  wider hidden layer. H72 quantized costs `19,552` but does not improve synth;
  top-5 non-overlap combo selection costs `19,453` and improves broad synth to
  `19753/20000` (`seed=25023`).
- The top-5 chooser's first version spent `6,400` params on an `80x80` BOOL
  overlap table. Dynamic origin-overlap computation from row/col tables removes
  that table and lowers the same chooser from `19,453` to `13,974`. The INT32
  row/col variant is cheaper than the initial INT64 row/col version (`14,774`).
- Narrowing the QLinearConv ranker from H30 to H24 gets under 10k: `9,808`
  with visible `266/266`. Adding even a top-4 dynamic chooser to H24 raises cost
  to `11,608`, so the current under-10 artifact is top-3 only.
- The under-5 target is reachable with a public-fit tiny ranker. The key
  Onnx_Advanced-derived cuts were: emit the origin score map directly as
  `8x10` via Conv pads `[4,4,4,4]`, reuse `fg_f32` for area `ReduceSum`
  instead of adding an area cast, store square-cell indices as `INT32`, and
  replace `Not -> Cast -> MaxPool -> Equal(0)` support gating with a 2x2
  `QLinearConv` sum + `Equal(4)`.
- `task023_origin_direct_qv0.onnx` is the lowest public-perfect artifact:
  cost `4,263` (`memory=3,579`, `params=684`), visible/live `266/266`. It is a
  no-hidden public-trained linear ranker and is not synth-safe.
- `task023_origin_top3_qtiny_h4.onnx` is the stronger under-5 learned artifact:
  cost `4,901` (`memory=3,899`, `params=1,002`), visible/live `266/266`,
  synth `926/1000` (`seed=23023`), `4585/5000` (`seed=24023`), and
  `901/1000` (`seed=25023`). It uses H4 QLinearConv ranker weights plus a
  public-perfect / synth-calibrated output quantizer.
- Post-ship compression pass (2026-05-27): merged the two square scatters into
  a single true-update scatter by redirecting disabled third-square cells onto
  an already-true top-2 cell, then replaced the `80x4` square-cell table with
  INT32 arithmetic from TopK origin indices. Best LB follow-up is
  `task023_singlescatter_dynidx_h4_4532.onnx`: cost `4,532`
  (`memory=3,844`, `params=688`), visible/live `266/266`, synth/A-B-vs-v5
  `926/1000` (`seed=23023`), synth `2748/3000` (`seed=24023`), and
  `901/1000` (`seed=25023`). H5 costs `4,716` with slightly better synth
  (`933/1000`, `2766/3000`, `908/1000`); H6 costs `4,900` with safer synth
  (`937/1000`, `2815/3000`, `936/1000`) but essentially no LB over shipped.
- Direct ranker after the same compression reaches `3,918`, visible/live
  `266/266`, but only `582/1000`, `2826/5000`, `527/1000` on the checked synth
  seeds, so it is rejected as a follow-up despite lower cost.
- Canonical `_canonical_6166_58_backup.zip` task023 is cheaper at `12,260`,
  but is cherry13b-confirmed zero-weight. A correct structural rewrite can
  unlock hidden LB even at higher cost.

Further compression ideas: deriving a non-branching bar/square criterion would
still beat branch search if found. The remaining dynamic-v4 cost is dominated
by fixed-cap residual candidates, repeated collection masks, and final crop/pad
output construction.
