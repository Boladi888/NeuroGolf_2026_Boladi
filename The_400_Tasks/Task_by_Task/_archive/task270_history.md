# task270 — archived attempt history

Moved from `Task_by_Task/task270.md` 2026-06-21 (W13) to keep the live card under the 20 KB budget. The live card retains the v14→v16 ladder + the W13 floor proof; everything below is the earlier provenance.

## Early attempts (rule decode + pre-v14 golf ladder)

| Date | Approach | Result | Cost | Session log |
|------|----------|--------|------|-------------|
| 2026-05-05 | Beta majority-outer-color rule | 256/266 (96.2% — 10 ambiguous-intersection fails) | n/a (numpy draft) | filed in recipe note |
| 2026-05-08 | Beta refined color-partition rule (verifier) | **266/266** numpy verifier | n/a | `Logs/sessions/2026-05-08_*.md` Beta-verify session |
| 2026-05-10 | Alpha-A: simplified `delta_3 > 0` predicate (canonical tie-break covers `delta_3 == 0` to outer1=7) | 266/266 numpy + ONNX | 81,207 | this card's session |
| 2026-05-10 | Alpha-A: submitted as `grape11_fig11_task270_anchor_cross` on **stale May-7 tracked submission.zip** (NOT actual fig11) | **5004.63** (publicScore; ~-592 vs true best — wrong base, not a task270 failure) | n/a | same |
| 2026-05-19 | crop15 / worker-7 idx200 sweep | STAGED task270_crop15_t_d25dcfce_on_5762_84 | cost 43145 -> 38657 (Δ-4488, +0.110 LB); verify 266/266; adversarial 7355/7355 (100%) | no init rewrites; model already works 15x15 internally |
| 2026-05-10 | Alpha-A: re-submitted as `lemon11_kiwi11_task270_anchor_cross` on **actual kiwi11 base** | **5596.38** (publicScore; **-0.54** vs kiwi11 5596.92 — existing external is cheaper) | 81,207 | same |
| 2026-06-04 | Re-ran promoted numpy reference | **252/266** on current bundled corpus; stale vs current LIVE/generator | n/a | do not use as source of truth |
| 2026-06-04 | Runtime-kernel 1D line rewrite: direct 4 channel slices, 1D `Einsum`/`CumSum` flags, two dynamic 3×3 Conv stamps, scalar canvas tail | `verify_against_live` **266/266** | 10,416 | superseded by sparse ScatterND |
| 2026-06-04 | Sparse ScatterND scalar canvas: free-input row/col/line `Einsum`s, 10 scalar writes, sentinel outside active region, BOOL `Equal` tail | `verify_against_live` **266/266**; `ng_gate visible` **266/266**; `ng_gate truth --full` **200000/200000**; `ng_gate ab` **2266/2266 byte-equal** | **6,550** | staged at `.hermes/board/staged_submits/task270_sparse_scatter_t_29288ef2_on_38430_cost6550/` |
| 2026-06-04 | U8 16×16 sparse canvas and rank-2 ScatterND indices (`build_task270_v3_u8_16scatter.py`) | `verify_against_live` **266/266** | 4,734 | saved canvas/index bytes but missed the under-4000 target |
| 2026-06-04 | Active 15-wide line logic with `CumSum` (`build_task270_v4_u8_15lines.py`) | `verify_against_live` **266/266** | 3,969 | first under-4000 exact candidate |
| 2026-06-04 | Active 15-wide line logic + triangular before/after masks (`build_task270_v5_tri15.py`; also `build_task270_v1.py`) | `verify_against_live` **266/266**; `ng_gate visible` **266/266**; `ng_gate ab` **2266/2266 byte-equal**; `ng_gate truth --full` **200000/200000** | **3,681** | primary staged at `.hermes/board/staged_submits/task270_tri15_t_807421ad_on_38430_cost3681/` |
| 2026-06-04 | Direct triangular side tests, no marker-line tensors (`build_task270_v6_direct_tri.py` / flat-index v8) | v6/v8 `verify_against_live` **266/266**; v8 `visible` **266/266** and `ab` **2266/2266 byte-equal**; v8 full truth timed out at 304s | 3,441 / **3,281** | cheaper but full-truth sweep too slow in this session |
| 2026-06-04 | 15×15 center-noop scatter probe (`build_task270_v10_center_noop.py`) | `verify_against_live` **266/266**; `visible` **266/266**; `ab` **2266/2266 byte-equal**; full truth timed out at 184s and 904s | **2,978** | LIVE/team-best; confirmed exact on Kaggle (lesson 2026-06-05) |
| 2026-06-12 | v11 no-upper-tri: remove `upper_tri` init (225 params); swap Einsum subscripts `zv→vz` (row-right) and `zr→rz` (col-down) to use `lower_tri` for both directions — pure algebraic identity since `upper_tri[z,v]=lower_tri[v,z]` | `visible` **266/266**; `ab` **2266/2266 byte-equal**; `truth --n 5000` **5000/5000**; `score_v2` **2753**; `profile_arcgen_266` **2753** | **2,753** | superseded by v14; SHA 5185b489... |
| 2026-06-12 | v12 int64-offsets: Cast F32→INT64 first, do Sub/Add in INT64; removes 8 F32 scalar intermediates | `visible` **266/266**; `ab` **2266/2266 byte-equal**; `truth --n 5000` **5000/5000** | **2,721** | superseded by v14 |
| 2026-06-12 | v13 noop-direct: remove 8 redundant row/col center-noop Where nodes (Where(flag,x,x)→x); saves 8×8=64B | `visible` **266/266**; `ab` **2266/2266 byte-equal**; `truth --n 5000` **5000/5000** | **2,657** | superseded by v14 |
