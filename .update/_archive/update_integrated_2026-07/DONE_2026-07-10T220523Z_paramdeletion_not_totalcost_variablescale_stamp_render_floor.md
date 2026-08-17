# Transferable insight — "delete the 47k-param hash table → floor_est ~900" is a FALSE FLOOR when the task is a variable-scale 2D template-stamp: params ≠ total cost, and the live best already deleted the param bank while total stays MEMORY-bound at the render floor

**Source**: task133 (`bo_ng_cld_worker3`, 2026-07-10). Audited LIVE **16,290** (Sho,
sha `535fec11…`, 2026-07-09); **DROP / floor re-confirmation**, not a new win.
Third sibling of the "SWEEP prompt targets a component the live best already dissolved" family:
- `DONE_2026-07-10T202012Z_argmax_indexplane_is_int64_costtrap_coloridx_collapse_falsefloor.md` (task064) — color-index collapse; ArgMax plane is INT64 cost-trap.
- `DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md` (task254) — free-output pad-tail; best already free-output.
- `DONE_2026-07-09T201927Z_ort_samesize_resize_shortcircuit_and_resize_beats_freeoutput.md` (task289) — Resize beats free-output for scaled block expansion (directly corroborates below).

Same trap, new collapse verb: here the "charged carrier" was a **memorized hash-MLP / dense classifier param bank** (47,234 params), and the floor_est (~900) was implicitly a **param-count** estimate.

## The trap it kills
Prompt Direction: *"hash-MLP / dense classifier decode: extract the generator rule and replace the learned/template graph with a tiny deterministic rule; delete the charged carrier/canvas/template; floor_est ~900."* The instinct: a memorized 267-row hash table (47k params) is the whole cost, so a symbolic rewrite ⇒ ~900. **Two independent reasons it fails:**

1. **The live best already deleted the param bank — params = 102 B, total = 16,290.** Sho's 16,290 graph is a from-scratch deterministic **bbox + template-stamp** solver: no hash-MLP, no dense classifier, no 267-row table. `inspect_graph` shows 39 tiny initializers, largest `W10` F32 [1,10,1,1] = 10 params; **total params 102 B**. The entire "delete the 47k-param table" win was ALREADY captured (that is the historical 799k→…→16,290 chain). The Direction's charged component is gone → DROP.

2. **floor_est ~900 was a PARAM estimate; the cost is now MEMORY-bound at the render floor.** Once the param table is gone, cost = rendering intermediates, which are FORCED because the task is a **data-dependent, variable-scale, 2-D template-stamp** (generator `task_57aa92db`: input has one full "template" creature + 2–3 partial sprites showing only a pivot + one body pixel at magnifier 1–4; output completes each partial sprite by stamping the template shape **scaled by its magnifier and recolored**). All three free-output killers are present → Conv/render floor, not a ~900 separable floor:
   - arbitrary 2-D creature shape (non-separable) → lib #109 forces canvas + render plane;
   - runtime-decoded selectors (template-color, target-colors, each position/scale/color computed from input) → lib #110 kills free-output regardless of rank;
   - variable placement count + **per-sprite scale (magnifier 1–4)** → lib #132 Conv/render floor.

## The mechanism (refreshed forced floor ≈ live 16,290)
`gridf` F32 [1,1,30,30] **3,600** (color collapse of F32 input; INPUT-DTYPE-LOCK; ArgMax-over-channel alt = INT64 7,200, cast-full-input = 9,000) + `rowany`+`colany` F32 **2,400** (per-color bbox occupancy; ReduceMax preserves F32) + `grid`+`gflat` U8 **1,800** + **4× `Resize` stamp `s0..s3` U8 3,600** (≤4 sprites, each a distinct (pos, scale∈1..4, color); one Resize = one box/one scale ⇒ 4 planes forced — cannot batch, magnifiers differ) + `outgrid` U8 **900** + detection/TopK/ROI chains **~3,850** ≈ **16,150 ≈ live 16,290**. `profile_real_cost --arcgen 266` = static=profile=16,290 (no GRADER-GAP).
Corroborates task289: **Resize is the champion stamp op for scaled replication** and free-output/ScatterND lose — the older ScatterND sparse-paint formulation of this exact task priced **21,394+**, strictly worse than Sho's Resize stamp.

**Measured (forced 2nd pass, `profile_real_cost --arcgen 266`, static=profile):** two genuinely-different assemblies both lose. (a) color-collapse via **ArgMax index-plane = 8,110** vs **Conv F32 = 4,520** — ArgMax has no output-dtype attr so the index plane is INT64 [1,1,30,30]=7,200 (same trap as the task064 sibling, now measured here). (b) stamp via **index-remap Gather = 16,260 for ONE sprite with fixed pivot and no detection** — the flat Gather index over 900 cells needs INT32 (values 0–899 > U8 255, traps §8c) → [1,1,30,30]=3,600/sprite = **4× a Resize plane (900)**. So the two most tempting rewrites are 1.8× and ~4×/sprite worse; Sho's Conv-collapse + Resize-stamp (#117) is the measured optimum.

## Rule of thumb
When a STRUCTURAL/false-floor SWEEP prompt says *"delete the memorized param table → floor_est ≈ small"*, first split the live cost into **params vs memory** (`profile_real_cost` prints `memory + params`; `inspect_graph` lists initializers):
- If the live best already shows **params ≈ 0–small** (bank already deleted), the Direction's charged component is gone → **DROP**; the floor_est was a param estimate that the owner already realized.
- A ~best/N floor_est is only credible if the task's output is truly free-output/separable. If the rule requires **rendering an arbitrary 2-D shape at data-dependent position with a data-dependent SCALE** (template-stamp / sprite-replication / magnifier), the render is a Conv/**Resize** floor (canvas + per-placement plane); memory dominates and the param-count floor_est is false. Prefer/expect **Resize stamps** over free-output or ScatterND for scaled 2-D replication (task289/task133).
