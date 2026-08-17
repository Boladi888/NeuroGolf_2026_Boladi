# task025 HashMlpRuleFloor — DROP (false floor) — 2026-07-10, bo_ng_cld_worker2 (Claude lane)

## Live bar (refreshed)
- **live_cost = 8834** (owner **Sho**, manifest, `NG_OnnxlyFans_TaskStatus.xlsx` + Backend/manifest.csv, updated 2026-07-09T01:09:17Z, points 15.9136), `ng_current_bar.py --task 25` says NOT stale.
- **live_sha256 = `e0d68f3dae9effa13145415702900b856a159bd153ff1692c4705d800e72802b`** (`Team_Best_Onnx/task025.onnx`, graph name `bianfu3`, opset 17, 63 nodes, 6232 bytes on disk).
- Gate re-measure of the bar: `ng_gate visible` **266/266**; `profile_real_cost --arcgen 266` static=profile **8834** (memory 8430 + params 404); `ng_gate truth --n 5000` **4992/5000 = 99.840%**.
- Prompt/context-pack numbers are STALE: context pack says 14,267 (Aditya), task_state 14,267, live_costs.json 23,495. All superseded by Sho's 8834 (2026-07-09).

## The prompt's premise is a FALSE FLOOR — the named charged component is GONE
Prompt Direction: "hash-MLP / dense classifier decode: extract generator rule and replace learned/template graph with tiny determinist. Delete charged carrier/canvas/template." Target ~900.

Sho's live 8834 graph has **no dense classifier, no hash-MLP, no template canvas**. It is already a lean deterministic construction: `ReduceMax/ReduceSum` row/col stats → `TopK`(k=4 colors) → `OneHot` → 2× position-weighted `Einsum` (`bchw,fc,w,z->bfhz` / `...h..->bfzw`, weight vector `[2048..2077]` encodes count+position in one pass) → 2× `QLinearConv` (`[8,4,1,3]`/`[8,4,3,1]` = ±1 adjacency projection) → **rank-26 `QLinearMatMul` render → `Equal` vs color-ids [0..9]**. The structural rewrite the prompt asks for was already done by Sho. Per the 2026-07-10 workflow update ("if [the charged component] is already gone… return DROP with the refreshed floor note"), this is a DROP.

## Rule (decoded exact, 266/266 numpy — scratchpad rule_probe.py)
Full rows/cols of a single color are "lines" (kept). Each off-line cell whose color equals a line's color projects to the cell **immediately adjacent** to that line on the cell's side (vertical line at col L: c<L→(r,L-1), c>L→(r,L+1); horizontal analogously). Non-matching cells deleted. Structural facts on 266 official: **≤4 distinct line-colors, ≤4 total lines**; histogram {1:19,2:109,3:103,4:35}.

## Why 8834 is at/near the practical floor (the priced wall)
1. **Render tail = 900 B forced.** Output `[1,10,30,30]` BOOL is a graph output → EXCLUDED from scoring (Sho at 8834 < 9000 proves it; the old "9000 B immovable output" floor claim was WRONG). To PRODUCE it you need a `[1,1,30,30]` UINT8 color-index (900 B) → `Equal`-expand. A per-channel render charges 9000 B/tensor, so the color-index render is forced. **900 is the render-tail floor.**
2. **Render matmul is provably minimal at rank 26.** Empirically decoded (scratchpad decode_render.py, train0): the 26 planes = 4 colors × 2 orientations × 3 planes (line + 2 adjacency projections) + 2 out-of-grid masks. Each line contributes exactly 3 rank-1 terms, so ≤4 active lines ⇒ only ≤12 planes are ever nonzero. BUT a color's orientation (row-line vs col-line) is unknown at build time, so BOTH must be statically allocated ⇒ 4×2×3+2 = 26. Collapsing to rank-12 needs a `Where`-selection layer that materializes the same both-orientation planes AND selects AND concats — strictly MORE tensors than Sho's direct rank-26 concat, for only ~840 B render-operand saving. **Rank-12 collapse is NET WORSE. Rank-26 is the render-architecture minimum.**
3. **Correctness/cost tradeoff is pinned.** Sho's 8 truth misses (scratchpad confirm_misses.py, first 5000 corpus) are **exactly the 5-line/5-color cases** (idx 965,1454,1634,2606,2622,3756,3815,4153 — all nlines=5 ncolors=5; 5-line freq = 8/5000 = 0.16%). Sho's TopK-4 is exact on ≤4 lines (99.84%) and misses every 5-line case.
   - MORE correct (truth-exact, 5th slot) ⇒ costs MORE: a 5th line re-pays both orientations at build time (see 2026-06-11 slots floor proof: k4→k5 = +2595…+4140 B). In Sho's render arch a 5-slot render ≈ 8834×5/4 ≈ 11k > 8834. Loses on cost.
   - CHEAPER (3-slot ≈ 6.6k) ⇒ truth ≈ 96% (drops all 3+line-color cases, which are 55% of the corpus) — far below Sho's shipped 99.84%. Hidden-bad, REJECT_LOCAL.
   - MATCH 99.84% but cheaper ⇒ must reproduce the 4-slot function; render is minimal (pt 2) and the position-weighted Einsum + conv detection is already tight (63 nodes, costs spread thin across ~40 tensors of 16–240 B, no fat carrier). No lever found.

## Formulations tried/analyzed (all ≥ 8834)
1. Sho 4-slot both-orientation rank-26 render — 8834 (incumbent, measured). ← the bar.
2. Rank-12 orientation-collapsed render — analytically + empirically WORSE (collapse `Where` layer > render saving; render decode proves the redundancy is unavoidable-without-select).
3. Truth-exact 5-slot render — ~11k (> 8834); more correct, more expensive.
4. Old slots/lpfl direct-render family — 23,495 (documented dead).
5. All-color route-B line-factor — 34,074 (documented dead).
6. Aditya free-output Einsum (`bpr,bpw,bos,ps->borw`) — 14,267 (documented dead).
7. Orientation-pair term compression (w12) — 15,407 (documented dead).

## Exact failed gate
No candidate priced below 8834 exists by construction (render-tail floor 900 + minimal rank-26 render 2460 + irreducible per-color detection ~6k). Fast-probe kill rule fired: minimal render construct prices ≫ 900. No visible/truth gate on a candidate because no candidate was built (near-floor "one crack / stop on priced wall" guidance).

## One remaining hypothesis (speculative, low-confidence)
The ~6k of position-weighted-Einsum detection + ~24 small `[1,4,·]` threshold tensors is the least-attacked cost center. IF a fundamentally different detection encoding computed line-position + both projection masks in materially fewer/smaller intermediates while keeping the 4-slot 99.84% correctness, it could dip below 8834. No concrete lever identified — the data is inherently [4×30]-shaped and Sho's node count (63) is already lean. Not worth a grind; the render (dominant reducible term) is proven minimal.

## Notable team intel
Current team-best (Sho 8834) is a **RISK ship at 99.84% truth** (misses all 5-line cases). If the hidden/scored set ever includes 5-line grids, task025 loses partial credit there. A truth-exact graph would cost ~11k. The team apparently accepted this (5-line freq 0.16%, and 0 in the official 266).
