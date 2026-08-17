# task387 worker packet — bo_ng_cld_worker3 (Claude lane, forced 2nd pass)

Date: 2026-07-12
Verdict: **STRICT WIN — staged 2566** (−41 vs live 2607, −40 vs prior effective best 2606). Path-A byte-equal. The P[18,30]=540-param embed remains a proven wall; the win is param-only Einsum operand-folds orthogonal to it.

## Candidate (NEW STAGED BEST)
- Staged: `The_400_Tasks/Task_by_Task/task387/staged_candidates/task387_drop_coord2_allc_cost2566/task387.onnx` (+ `manifest.json`)
- Builder: `The_400_Tasks/Task_by_Task/task387/prototypes/build_drop_coord2_allc.py`
- Prototype: `The_400_Tasks/Task_by_Task/task387/prototypes/task387_drop_coord2_allc.onnx`
- **profile cost = 2566** (static==profile); score_v2 = `(0, 1906, 660)`; points 17.150
- candidate_sha256: `6a3f8962a08024e78a830376e3c693a6de53fb5c81f8691d6e7886792e17c8e4` (raw 7609 B)

## Bar + delta
- **Live team-best 2607** — `python tools/ng_current_bar.py --task 387 --json` → authoritative_source=manifest, owner=Vamshi, sha `f5ceadec…`, updated 2026-07-02T02:35:15Z, path `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`. Unchanged this session.
- Effective best before this pass = local staged `task387_absorb_batchW_cost2606` (2606). Target was strictly <2606.
- **Cost delta (candidate − live): 2566 − 2607 = −41 B.** vs effective best: 2566 − 2606 = −40 B.
- Stale sources ignored: main-repo Team_Best_Onnx 2455 (same sha), task_state 3516, live_costs.json 3842, library_db 100793.

## Named saving (`ng_graph_cost_diff` vs live Team_Best 2607)
```
-1  params  removed initializer  batch1    (float16 [1] absorbed into W [6,4]->[1,6,4]; inherited from 2606 parent)
-30 params  removed initializer  coord2_f  (float32 [30]; 2nd moments now r^2 via doubled coord_f)
-10 params  removed initializer  allc_f    (float32 [10] all-ones; c-axis summed implicitly)
RECONCILE: params -41 == cost_predictor -41 [OK]; memory +0 B [OK]
NET COST DELTA -41  -> candidate is CHEAPER
```

## The two shaves (genuinely different from pass-1: pass-1 attacked P/params-vs-memory; this attacks redundant operand inits)
1. **drop `coord2_f[30]` (−30 params).** `coord2_f = coord_f²`, used ONLY by the two second-moment Einsums. Replace it by contracting `coord_f` twice on the same axis:
   - `row_s2: 'bcrs,c,r->'(input,nonbg_f,coord2_f)` → `'bcrs,c,r,r->'(input,nonbg_f,coord_f,coord_f)`
   - `col_s2: 'bcrs,c,s->'` → `'bcrs,c,s,s->'`
   `coord_f` already exists (the `cross` node already doubles it) ⇒ no new init, no new intermediate. Byte-exact (r² exact in F32 for r≤29).
2. **drop `allc_f[10]` (−10 params).** All-ones channel weight is redundant — Einsum sums the `c` axis implicitly when the operand is removed:
   - `area_f: 'bcrs,c->'(input,allc_f)` → `'bcrs->'(input)`
   - `active_rsum_f: 'bcrs,c,r->'(input,allc_f,coord_f)` → `'bcrs,r->'(input,coord_f)`
   - `active_csum_f: 'bcrs,c,s->'` → `'bcrs,s->'`
   Byte-exact (allc_f≡1; area/active are small-integer sums ≤~13000, exact in F32 regardless of reduction order; single-operand `'bcrs->'` runs in ORT).

Memory unchanged (1906) — no new tensors; pure initializer deletions.

## Gates run (exact result lines)
```
ng_current_bar --task 387 --json            live_cost=2607 sha f5ceadec… (manifest, Vamshi)  UNCHANGED
score_v2                                     (0, 1906, 660) cost=2566
ng_gate visible --task 387                   [visible] 266/266 (100.00%)  PASS
profile_real_cost --arcgen 266               static=2566 profile=2566 (memory 1906 + params 660; 266 real inputs)
ng_gate truth  --task 387 --n 2000           [truth] vs ARC-GEN ground truth: 2000/2000 (100.000%)
ng_gate ab     --task 387 --n 2000           [ab] vs submission.zip: match 2266/2266 | divergences=0 | byte_equal=True (first_div=None)
ng_gate battery --task 387 (staged copy)     === battery task387: PASS ===  [score] cost=2566 (mem 1906+par 660) points=17.150
ng_graph_cost_diff vs live 2607              NET COST DELTA -41 (batch1 -1, coord2_f -30, allc_f -10; memory +0)  [OK]
ng_safety model --task 387 --json            verdict=PASS  n_failures=0  n_warnings=0  runtime_integer_denominator_count=0
```
(raw `--full`/n=200000 NOT run — banned p42; n=2000 is the ship ceiling. Byte-equality is also GUARANTEED analytically: both folds preserve exact-integer F32 arithmetic.)

## Contract fields
- candidate_class: **PATH_A_BYTE_EQUAL** (ab byte_equal=True vs live submission.zip)
- parent_source: `staged_candidates/task387_absorb_batchW_cost2606/task387.onnx` · parent_sha256: `474ac7a55706f67dd7928343ffb99cb50a0833a1daee060fb4c8d2e1d7e7ce53`
- candidate_sha256: `6a3f8962a08024e78a830376e3c693a6de53fb5c81f8691d6e7886792e17c8e4`
- proof_commands_run / proof_results: see Gates above (profile static==profile; byte_equal_to_live=true; truth 2000/2000; ab divergences=0)
- **safety** (raw `ng_safety.py` JSON):
```json
{"kind":"model","verdict":"PASS",
 "summary":{"n_failures":0,"n_warnings":0,"runtime_integer_denominator_count":0},
 "model":{"task":"task387","sha256":"6a3f8962a08024e78a830376e3c693a6de53fb5c81f8691d6e7886792e17c8e4",
          "size_bytes":7609,"failures":[],"warnings":[],
          "risk_op_flags":["Cast","Div"],"runtime_integer_denominators":[],
          "official_io_contract":{"data_inputs":["input"],"outputs":["output"],"ok":true},
          "load_after_sanitize":"ok","verdict":"PASS"}}
```
- x86_required: **false** · kaggle_probe_required: **false** · **status: LOCAL_GOLD**

## What I tried this pass (and what failed) — genuinely different formulations
| formulation (different from pass-1) | result |
|---|---|
| **coord2_f→doubled-coord_f fold** | **WIN −30** (staged) |
| **allc_f→implicit-c-sum fold** | **WIN −10** (staged) → 2566 |
| **narrow-dtype render** (INT8/UINT8/INT16 profiles + integer Einsum) | DEAD — ORT `Einsum` only implements INT32 & F16 (INT8/UINT8/INT16 = NOT_IMPLEMENTED, tested); INT32 is 4 B (worse than F16). Render forced to F16 (techniques.md line 3926 concurs). |
| **input-as-Einsum-operand** (reuse the free [1,10,30,30] input to save a factor) | DEAD — carrier-dtype tax (techniques §4203): reading the F32 input F32-locks ALL operands, doubling R/C/P bytes. The current no-input-operand render keeps everything F16. |
| further droppable init (`nonbg_f`, `channel_id_f`, `coord_f`, `coord16`, `e0`, `e5`, `W`, `oh16`, scalars) | none — each load-bearing. `nonbg_f`=[0,1,…,1] is a genuine ch0-exclusion (NOT all-ones, can't implicit-sum); deriving any other init materializes a charged intermediate > the param it saves. |
| P[18,30]=540 embed (pass-1 wall) | still walled — sparse-P re-verified DEAD (strict shape-inference rank-0 crash), no-P 3135, zero-block MEASURED 2858, Pad-after ~2790, materialize ~+559, rank<6 forced. |

Library/BoBrain mined as required: library_db (stale, best_cost 100793 — pre-dates the render), BoBrain provider context (leaderboard state), techniques.md free-output floor proofs (task011 operand-vs-node, task057 growth-op, task295 factors-are-memory, task398 degree-LB) — all independently confirm the P-embed wall; none offered a render-side lever. The operand-fold wins came from direct graph inspection, not the library.

## Remaining floor / WALL argument (why <2566 is not reachable by this route)
The output `[1,10,30,30]` one-hot is a **forced rank-6 separable render**. Params floor 660 = **P 540** + coord_f 30 + coord16 18 + nonbg_f 10 + channel_id_f 10 + e0 10 + e5 10 + W 24 + oh16 2 + scalars 6; memory floor 1906 = R,C[6,18] 432 + ~24 F16 profile pieces 864 + 16 BOOL masks 288 + T 80 + front-end F32 ~242. Forced-component argument:
- **P (540) is irreducible** (task011 operand-vs-node law): the Einsum must emit the 30-wide output; the size-30 axis is supplied EITHER by the shared identity-embed OPERAND P (540 element-params, 0 memory — optimal) OR by width-30 R/C tensors (≥720 memory-bytes) OR a widened pipeline (+1056). Element-params < memory-bytes ⇒ dense P wins; sparse-P discount blocked by strict shape-inference; render dtype can't go below F16 (Einsum INT8 no kernel).
- **rank 6 forced** (6 distinct row⊗col spatial layers; W rows 2&3 share color but distinct R/C).
- **R,C,profiles are DATA-DEPENDENT** (marker geometry varies) ⇒ charged F16 memory, not initializers (task295 law); polish no-op ⇒ minimal.
- **every non-P init is load-bearing** (verified exhaustively this session).

⇒ 2566 is the floor for this render family. A sub-2566 win requires breaking the P-embed (needs sparse-init strict-shape-inference support, or an integer Einsum kernel, or a lower-rank render — none available).

## Artifacts (all task-local)
- STAGED: `staged_candidates/task387_drop_coord2_allc_cost2566/{task387.onnx, manifest.json}`
- `prototypes/build_drop_coord2_allc.py` (the win), `prototypes/build_drop_coord2.py` (intermediate −30 rung), `prototypes/task387_drop_coord2*.onnx`, `prototypes/task387_2566_polished.onnx` (polish no-op), `prototypes/build_zeroblock_concat.py` + `task387_zeroblock.onnx` (pass-1 P-avoidance evidence, 2858)
- `hard_records/freeoutput_einsum_render_2607_wall.md` — ADDENDUM updated: WIN to 2566, operand-fold technique, P-wall stands at 2566
- Attempts row updated in `task387.md`

Lesson packet: none filed separately, but the transferable lever is documented in the hard-record addendum — **"all-ones / squared coordinate weights fed to an Einsum are droppable operands"**: an all-ones channel/spatial weight → drop the operand (Einsum sums that axis implicitly); a squared coordinate init → contract the base coordinate twice on the same index. Both are param-free, memory-free, byte-exact for integer/exact-F32 sums. General to any moment/reduction front-end. Consider promoting to `Onnx_Advanced` via a `.update/DONE_*.md` if a sorter pass runs.
