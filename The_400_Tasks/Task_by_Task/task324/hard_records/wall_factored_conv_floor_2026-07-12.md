# task324 — WALL @ 7898 (Sho factored-conv floor) — worker3 batch 2026-07-12

## Bar (live, authoritative)
- **live_cost = 7898** (manifest, owner **Sho**, `2026-07-09T01:09:17Z`).
- **live_sha256 = `164004214e841c7cd4caaf061d82077801a0db71d88c54d621468a718d7edf9b`**
- artifact: `Team_Best_Onnx/task324.onnx`; `profile_real_cost --task 324 --cand ... --arcgen 266` = **static=7898 profile=7898** (memory 7611 + params 287, 266 real inputs). **No dynamic-shape fiction** → no boring-static-sibling free win.
- No lower staged sidecar (`Staged_Boladi/` has no task324; `Staged_Vamshi/_shipped/…20260630…` is pre-Sho).

## VERDICT: WALL. No candidate strictly below 7898 found. Sho's 7898 is the formulation floor.

## What Sho's 7898 actually is (SUPERSEDES the stale "X-Conv is the floor" claim in task324.md)
Sho's graph (opset 20, 43 nodes, **287 params only**) does NOT use the 1,521-param 39×39 X-kernel. It renders the X-diagonal with a **factored / dilated conv**:
- `w1` UINT8 `[2,1,7,7]`: ch0 = `\` diagonal (identity), ch1 = `/` diagonal (anti) — 7-reach each → `s1 [1,2,20,20]` (800 B). QLinearConv pads `[3,3,3,3]`.
- `w2` UINT8 `[1,2,9,9]`, **dilations `[4,4]`** (33-reach, per-direction) → `s2 [1,1,20,20]`. pads `[16,16,16,16]`. Combined reach = 7 + 33 − 1 = **39** = corner-to-corner for 20×20.
- **260 params** for the diagonal render (w1 98 + w2 162) vs **1,521** for the single X-kernel. The 2 channels are REQUIRED to keep the two diagonal families separate (see dead route D1).

Color-collapse (node 0): 1×1 `Conv(input, ck)` with `ck`=F32[1..10] and **negative pads `[0,0,-10,-10]`** — collapses one-hot 10ch AND crops 30×30→top-left 20×20 in ONE op → `vf [1,1,20,20]` F32 (1600 B) → Cast → `v` U8 (400 B). Output tail: `cm20`(400) → QLinearConv `wid`(pad 20→30) → `cm30 [1,1,30,30]` U8 (900 B) → `Equal(cr[1..10], cm30)` → one-hot BOOL output (output tensor itself is free).

## Priced floor proof (from-scratch minimal design converges to 7898)
| Block | Charged | Forced-because |
|---|---|---|
| color rep `vf`1600+`v`400 | 2000 | No F32→U8 channel-reduce op exists. ArgMax→INT64=3200; F16 Conv illegal (weight dtype must == F32 input); QLinearConv needs [1,10,30,30] U8 cast=9000; Mul+ReduceSum touches [1,10,20,20]=4000 (and ReduceSum(U8) rejected). Conv-decode 1600 is the lean floor. |
| diagonal `rare`400+`s1`800+`s2`400+`line`400 | 2000 (+260p) | Rank-20 spread ⇒ conv. Factored 260p < single X 1521p < shear 3120p < Einsum-basis 15600p. 1-ch merge fails correctness (D1). |
| bg masks `ma`+`mb`+`mj` | 1200 | Plaid bg-class is per-cell; ma,mb both needed for recolor; mj=ma∪mb for `rare`. |
| recolor `pm0`+`pm`+`cm20` + reductions | 1200 + ~220 | 3-way select (bg-A / bg-B / original) = 2 nested Where (ORT has no fused 3-way). row∧col reductions both needed: plaid bg-class = f(row-stripe, col-stripe) is intrinsically 2D. |
| output `cm30` | 900 (+1p) | 30×30 output geometry forces a 30×30 U8 carrier; doing Equal at 20×20 first = [1,10,20,20]=4000 B. |
| misc `cnt`40+`ti`32, params ck10+cr10+scalars6 | ~98 | TopK color detection. |

Sum ≈ 7592 mem + 287 params = **~7879 ≈ 7898**. Every ≥400 B plane maps 1:1 to a required logical value; no fusion removes one.

## Dead routes (priced — do not re-explore)
- **D1 — 1-channel factored diagonal** (build `prototypes/build_task324_w3_diag1ch.py`, `task324_w3_diag1ch.onnx`): merge w1/w2 to 1 channel (X-shaped) to delete 400 B (s1 800→400). **`ng_gate visible` = 0/266 (0.00%) FAIL.** Cross-terms `w2[\]⊛s1[/]` + `w2[/]⊛s1[\]` create spurious diagonals. The 2-channel `s1` (800 B) is FORCED.
- **D2 — F16 color-collapse** (stale `build_task324_w3_f16colorconv.py`, targets Aditya node names): illegal — ONNX Conv requires weight dtype == input dtype (F32); an F16 output needs a Cast of the F32 input first = 18,000 B. DEAD.
- **D3 — QLinearConv/U8 color-collapse** (stale `build_task324_w3_qcolorconv.py`): needs `Cast(input→U8)` = [1,10,30,30] = 9,000 B > the 1,600 B it removes. DEAD.
- **D4 — shape-only / shear diagonal (the prompt's "ShapeOnlyRearrange" hint)**: circular roll is INCORRECT (r+c∈0..38, mod-20 merges r+c=k with r+c=k+20). Correct non-wrap shear = width-39 planes (780 B) + a [20,39]=780-param Gather index per shear, ×4 (2 families × shear+unshear) ≈ 3,120 params + 3,120 B — 3–4× worse than the 260-param factored conv. GridSample grid = 6,240 B F32. DEAD. **The prompt's shape-only framing is a misclassification: the rule requires a rank-20 diagonal spread AND per-cell color decisions (recolor by bg-class), neither of which is a shape rearrangement — the prompt's own kill-rule ("kill if correctness needs per-cell decisions") applies.**
- **D5 — Einsum/free-output diagonal**: literal `[20,20,39]` incidence basis = 15,600 params/family; ORT Einsum can't consume BOOL marker planes. DEAD.
- **D6 — cheaper 2-stage diagonal factorization (26-param win attempt)** (build `prototypes/build_task324_w3_diag96.py`, `task324_w3_diag96.onnx`): param-optimal split over `(k1 odd, k2, d)` with `(k1−1)+(k2−1)d=38, k1≥d` gives `w1[2,1,9,9]`(162p) + `w2[1,2,6,6]` dil6 pad15 (72p) = **234p, −26 vs Sho's 260** → would be 7872. The effective composed 2D kernel is **byte-identical** to Sho's (verified: solid centered 39×39 X). **BUT `ng_gate visible` = 163/266 FAIL.** Root cause: `k2=6` is even → stage2 taps at ±3,±9,±15 with **NO center tap (offset 0)**. Because `s1` is materialized/truncated at 20×20, a corner marker's stage1 spread is clipped to `s1[15..19]`; a diagonal cell 2 away (e.g. (17,17) for marker (19,19)) reads `s1` at 17±{3,9,15}={14,20,8,26,2,32}, all missing [15,19] → DARK. Sho's odd `k2` has a center tap that reads `s1[17]` directly → lit. **⇒ `k2` MUST be odd (center tap = boundary correctness). With odd `k1`,odd `k2`: min is exactly 260 (7/9-d4 and 9/7-d5 tie). Sho's factored conv is the proven 260-param diagonal floor.**

## One remaining (rejected) hypothesis
The only sub-400 B slack is the ~220 B reduction block (`rha…cbv`, pa1/pb1 derivation). Dropping the row-OR-col redundancy (~60 B) BREAKS correctness: plaid bg-class = f(row-stripe, col-stripe) is intrinsically 2D (compress_v3 lesson: `on_base=(countB_row>0)AND(countB_col>0)`), so both axes are required. Even if a ~60 B shave existed it risks the 100%-truth for a marginal gain and would need a full `ab` gate to prove byte-equal. Not worth it — no clean win exists.

floor_est ~1500 in the prompt is **FANTASY** (it assumed a pure shape-rearrange). Realistic floor = ~7898.

---

## PASS 2 (2026-07-12, forced second pass) — library-mined + one more genuinely-different build. WALL HOLDS.

**Library floor confirmations (the two dominant blocks are documented floors):**
- **Color-collapse 1600 F32 = library-confirmed floor.** `techniques.md §8b` (line 714, task400/task165): "*the single 30×30 F32 canvas (3,600) is irreducible — an F16/U8 canvas needs a full-input cast (18,000/9,000) that dwarfs the saving, and ArgMax-over-channels emits INT64 (7,200).*" Our cropped 20×20 analogue = 1600 F32, same logic. Sho's negative-pad crop-collapse Conv IS the documented "DILATED crop-collapse" pattern (§8b line 732).
- **Diagonal render via QLinearConv = library-recommended.** `by_symptom.md §377` ("geometric ray/diagonal drawing from anchor blocks") → "*single QLinearConv with documented integer kernel weights … cap each footprint cell at 1*". Sho's factored version is the param-optimal instance.

**D7 — single 39×39 X-conv (genuinely-different assembly: no `s1` intermediate)** (build `prototypes/build_task324_w3_singleX.py`, `task324_w3_singleX.onnx`): compose Sho's `w1⊛w2` into one dense `xk[1,1,39,39]` X-kernel; deletes `s1` (−800 B) but pays the dense kernel (+1521 p). **visible 266/266 PASS** (correct-by-construction), **profile static==profile==8359** (mem 6811 + params 1548). `ng_graph_cost_diff` names it: `s1` −800 B, `xk` +1521 p, `w1`+`w2` −260 p → **net +461, REGRESSION**. Concretely refutes the stale "X-conv 1521p is the diagonal floor" claim AND proves the factored `s1` (800 B) beats the dense kernel (1261 p). No truth/ab — cost-loser, not a ship candidate.

**Other genuinely-different formulations priced-dead this pass (diagonal-block charged vs Sho's 1860):**
- Reshape/pad diagonal (0 conv params): ≥3160 (2 families × Pad[20,39] 780 + Expand[20,40] 800; reshape outputs ARE charged — scorer counts static value_info, verified in `_score_v2.py`/`neurogolf_utils.score_network`). +1300.
- Flip-reuse shared "\"-kernel (input-flip + 130 p): 3330 (8×400 B planes + 130 p). +1470.
- Channel-domain recolor (skip color-index carrier): recolor one-hot plane = [1,10,30,30] = 9000 B vs cm30 900. DEAD.
- Cumsum/matmul diagonal basis: 15,600 p/family. DEAD.

**Information-theoretic WALL argument:** the diagonal spread is a **translation-invariant** operator (identical X pattern rendered at every marker). Convolution is the canonical minimal representation of a translation-invariant map — it stores the pattern ONCE (260 factored params) and applies it at every position for free. Every non-conv assembly must either store per-position indices/intermediates (reshape 3160, shear 3120, matmul 15600) or re-derive per-family (flip-reuse 3330) — all strictly worse. The color-collapse (1600, §8b floor) + factored-conv diagonal (260 p + 1200 B, translation-invariance floor) + per-cell recolor (each plane = 1 required logical value) + 30×30 output carrier (900, geometry) leave **no charged surface removable without losing information**. **7898 is the floor for this rule. WALL confirmed (pass 2).**
