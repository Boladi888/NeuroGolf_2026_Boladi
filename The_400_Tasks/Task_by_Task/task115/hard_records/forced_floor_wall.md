# task115 forced-floor WALL (bo_ng_cld_worker3, batch48)

- **Live/team-best bar:** 389, SHA-256 `e959b06e6782df52dbb55e802bb0613d0825fb4f71bad262712975dcf8022357`,
  source `Team_Submission/Backend/manifest.csv` + workbook row updated 2026-07-02T02:35:15Z
  (owner Vamshi). Confirmed via `tools/ng_current_bar.py --task 115`.
- **Verdict: WALL. 389 is AT the forced floor** under the current cost model + [1,10,30,30]
  encoding. The prompt's `floor_est ~120` is an authoring artifact: it was written against a
  STALE 14k full-grid F32-band baseline (`live_costs.json` still shows 13,187). The LIVE graph
  is already Vamshi's from-scratch structural rewrite (11 nodes) that does the per-axis
  positional reduction the prompt asked for. There is no 14k canvas left to delete.

## Cost model (source-read: neurogolf_utils.py calculate_params / calculate_memory)
- `params` = total ELEMENT COUNT of every initializer (dtype-independent; F32→F16 saves 0 params).
- `memory` = Σ intermediate-tensor `elements × dtype_itemsize` (max of static & profiled shape),
  excluding graph `input`/`output`.
- 389 = params **190** + memory **199**.

## Forced params (190) — every item pinned
- `row_decay[30]` + `col_decay[30]` = **60**. Dual positional scoring bases; length 30 forced by
  the 30×30 input contraction (`Einsum nchw,h,w->nc`). Serve DOUBLE duty: color order (TopK) AND
  orientation (top_values magnitude ratio). Cannot shorten without slicing the input (adds a
  large intermediate).
- `linear_basis_h[2,30]` = **60**. Painting basis `[ones; ramp]`; width 30 forced by the 30-wide
  output canvas. Shared across all four paint contractions (counts once).
- `term_slot_h[5,4]`=20 + `ra/rb/ca/cb[5,2]`×4=40 = **60**. The 5-term single-Einsum painter
  (3 orientation-shared + 1 row-gated + 1 col-gated). Minimum to paint 4 ordered positions in
  EITHER orientation inside ONE Einsum (see dual-canvas refutation below).
- `orient_coeff[4]`=4, `k4`=1, `zero_f`=1, `depth10`=1, `oh_values[2]`=2, `one_h`=1 = **10** logic.

## Forced memory (199)
- `seq_h[1,10,4]` F16 = **80** — slot→color router; 10 canvas channels × 4 TopK slots. OneHot is
  the cheapest router (Equal costs more — proven, Formulation C).
- `score[1,10]` F32 = **40** — per-color score; F32 (from F32 input) × 10 colors, forced.
- `top_indices[1,4]` I64 = **32** — TopK indices are I64 by spec.
- `top_values[1,4]` F32 = **16**; rest (present/row/col/orient/concat) ≈ **31** small.

## Independent formulations tried (all ≥ 389)
1. **Dual-canvas direct painter** (`prototypes/bo_ng_cld_worker3_dualcanvas/`,
   SHA `18c2e2ae190b261ac541fd2fa4f260e85b8df6de108bc76c074048e763de884c`):
   two explicit `[1,10,30,30]` placement Einsums (row-canvas + col-canvas) blended by orientation.
   **score_v2 = 72,500** (mem 72,277) — **visible 266/266 PASS** (algorithm correct, cost only).
   ⇒ Proves the LIVE 5-term single-Einsum term-gated painter is FORCED; materializing a per-
   orientation canvas is the exact 30×30-canvas trap the prompt warns against.
2. **Equal-based router** (`prototypes/bo_ng_cld_worker3_eqrouter/`,
   SHA `73859290c63bf94f4b2534429b02aaff3d348221fa89b86df5cd89f1157df164`):
   replace OneHot with `Equal(top_indices, arange10)` + Cast, keep LIVE painter.
   **score_v2 = 474** (arange10 +10 params, extra bool `[1,10,4]` intermediate). ⇒ closes the
   prior worker's "shrink seq_h/OneHot/top_indices" hypothesis: OneHot is the cheapest router.
3. **Remove top_present via decay-scaling** (prior worker bo_ng_cdx2_worker2, hard_record
   `topvalue_present.md`): static 385 but **visible 221/266 (83%) FAIL** — F16 render sensitivity.
   ⇒ presence Greater/render is forced.

## Remaining hypothesis (thin)
The only untested micro-lever is re-deriving the painter with 4 terms instead of 5 (would save
~12 params → 377 if it exists). This is (a) an epsilon shave the prompt explicitly says not to
chase, (b) high-risk: the 5 terms = 3 shared + 1 row + 1 col already look minimal for two-
orientation placement. Not pursued. No sub-389 STRUCTURAL route exists; treat 389 as the floor.
