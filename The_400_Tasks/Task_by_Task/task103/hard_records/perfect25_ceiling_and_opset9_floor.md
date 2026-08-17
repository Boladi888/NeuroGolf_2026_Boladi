# task103 — WALL record: perfect-25 cost-1 impossible; opset-9 structural floor = 51

- **Live bar at time of work:** cost **51** (manifest authoritative, owner Vamshi,
  sha256 `c7118d8de216ca7b4a2e984ec885d282f2faa99d66da3dafb70579055235931d`,
  updated 2026-07-02). Confirmed via `tools/ng_current_bar.py --task 103 --json`.
- **Worker:** bo_ng_cld_worker3 (Claude lane), batch p53, 2026-07-07.
- **Verdict:** WALL. No candidate strictly below 51 exists. Do not re-explore the
  perfect-25 / cost-1 family, nor the "drop the casts" shave.

## What the task actually is (ground truth over ALL 223 corpus examples)
- Input: always 3×3, colors **strictly {0, 2}** (no other color appears in 6 train
  + 2 test + 215 arc-gen). Output: always **1×1**, value **1** (19×) or **7** (204×).
- Rule: **horizontal mirror symmetry of the red (color-2) plane** — output 1 iff
  `red[r,0] == red[r,2]` for **all** rows r∈{0,1,2}; else 7. Verified 223/223.
- All 3 row-comparisons are load-bearing: any 2-row subset misclassifies
  (rows (0,2)→199/223, (0,1)→191, (1,2)→190). Single row →~133/223.

## Why the prompt's "perfect-25 cost-1 identity/transpose" premise is FALSE
- Output is a **1×1 class label** (color 1 or 7), NOT a shape-preserving transform
  of the 3×3 input. No no-payload direct-output op (Identity/Transpose/flip/rotate)
  changes shape 3×3→1×1, and none computes the symmetry decision. Dead on inspection.
- The rule note in the prompt already said the local pre-scan found no whole-grid
  identity/transpose/flip/rotate match. Confirmed: cost-1 is unreachable here.

## Why the "drop the 6 redundant Cast→BOOL" shave (target cost 45) is DEAD
- The 6 `Cast(F32→BOOL)` nodes are **load-bearing, not redundant**: at **opset 9**
  `Equal` does NOT accept float inputs (ORT: `INVALID_GRAPH: Type 'tensor(float)'
  of input of operator (Equal) is invalid`). The casts exist to make Equal legal.
- Bumping to opset ≥11 (where Equal accepts float) makes **Slice** take
  starts/ends/axes as **input tensors** → 6 slices × new INT64 initializers, far
  exceeding the 6-byte cast saving. Net loss.

## Forced-floor argument (cost = memory + params; verified via `_score_v2.score_network_v2`)
- **Params floor = 12.** 10-channel one-hot output ⇒ Conv weight `[10,1,1,1]` = 10
  params (irreducible; sparse zeros still counted). Two-sided decision (color 1 vs 7)
  from one signed scalar needs ±1 constants (Where pos_one/neg_one) = 2. Alternatives
  (two weight vectors = 20; ScatterND base+indices ≥ 12) all ≥ 12.
- **Memory floor = 39 (opset 9).** Cheapest comparison primitive = single-cell F32
  slices (4B). 6 cells(24) + 6 bool Cast(6) + 3 Equal(3) + 2 And(2) + Where signed
  F32(4) = 39. Output [1,10,30,30] not billed (activation-memory rule).
- **Total = 39 + 12 = 51 = LIVE.**

## Independent reformulations tried, all priced ABOVE 51 (score_network_v2 = (macs, memory, params))
| Formulation | memory | params | cost | vs 51 |
|---|---|---|---|---|
| LIVE (single-cell, bool Cast, Equal, And) | 39 | 12 | **51** | — |
| B: Sub→Cast(bool)→Or (opset-9 legal, avoids Equal) | 45 | 12 | 57 | +6 ❌ |
| C: col strips [1,1,3,1] + Cast + Equal + slice-scalars + And | 42 | 12 | 54 | +3 ❌ |
| strip + INT8 ReduceMin reduce | ~36+ overhead | 12 | >51 | ❌ |
| whole red-plane [1,1,3,3] cast then slice | 60 | 12 | 72 | ❌ |
- Sub produces 4B F32 per row (> Equal-on-bool 1B); strips are 12B (> 4B single cell)
  and force an extra bool-reduction. Single-cell + bool-Equal + And is the optimum.

## Remaining hypothesis
None credible. The graph is at its opset-9 structural floor. A sub-51 route would
require either (a) a param model that outputs 10 one-hot channels with <10 stored
values (none exists via Conv/Mul/Gather without adding index inits), or (b) a
comparison structure cheaper than single-cell bool-Equal (none found; all alts lose).
Leave task103 unchanged.
