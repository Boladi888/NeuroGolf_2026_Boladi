# task005 dead routes — compact-bbox rewrite & validation shave (2026-07-08)

**Worker:** bo_ng_cld_worker2 (Claude lane). **Live bar:** cost **2852**
(profile, `ng_current_bar --task 5`; manifest authoritative, owner Aditya,
refreshed 2026-06-22). **Live sha256:** `d38cdf047af75e26d3bcf919b8b9fe8e70314233612430b87de422317f003d8b`
(= `Team_Best_Onnx/task005.onnx`, 13417 B, 84-node F32→BOOL graph).

> ⚠️ The local `task005.md` / `task_state.csv` / `live_costs.json` claim LIVE
> = 16,738 (motif-ray FP16 graph). **That is STALE.** Aditya rewrote task005 to
> a 2852 graph (Einsum/ArgMax/Gather/GridSample/QLinearConv, hash-table motif
> lookup). The old 13998/14988 staged candidates and all the 16738-era floor
> analyses are superseded — ignore them. Analyze `Team_Best_Onnx/task005.onnx`.

## The live 2852 graph (where the bytes are)
`profile_real_cost --task 5 --arcgen 266` → **2852 = memory 2662 + params 190**
(fully static; static-sum == profile). Two tensors dominate:
- `_padded_v291_v7` [1,1,30,30] UINT8 = **900** — the output canvas.
- `v291_i8` [1,1,21,21] UINT8 = **441** — the pre-pad paint (conv result).
- next: GridSample `_v55_sampled_onehot_sel` [1,10,1,8] F32 = 320; then ≤81 each.

Algorithm: color-counts→ArgMax source color; row/col **moments**→anchor;
MatMul-**hash**→Gather packed-motif table→BitShift unpack = 3×3 motif shape;
**GridSample** input at 8 offsets→ArgMax = 8 marker colors; QLinearConv(anchor⊗
motif)=9×9 stamp, QLinearConv(stamp⊗ray-kernel)=21×21 paint; Pad(99)→30×30;
Equal(0..9)→one-hot output.

## CLASS: ambiguous (byte-equal-vs-LIVE is the gold gate)
`truth --full` on **LIVE itself = 198810/200000 (99.405%)** — NOT 100%. Per
traps §CLASS-CHECK this task is ambiguous-class: a successor must be **byte-equal
to LIVE** (matching its 1190 "misses"), not merely truth-correct. Any semantic
change that alters even one output is disqualified.

## DEAD ROUTE 1 — eliminate the 441 pre-pad (direct 30×30 conv)
**Hypothesis:** make the replication QLinearConv emit 30×30 directly (asymmetric
pads, kernel stays 7×7), drop the separate 21×21 tensor → −441 → ~2411.
**DEAD by proof, no build needed.** Grader (`neurogolf_utils.verify_subset`) does
exact `np.array_equal((output>0), expected_onehot)`. For a 21×21 grid the
expected out-of-grid region is **all-false** (encode only sets in-grid cells). A
conv emits **0** in its border → channel-0-True out-of-grid → mismatch. The
**99-border is forced**, which forces a 21×21 tensor distinct from the 30×30
one. **441 + 900 = 1341 paint bytes are forced.**

## DEAD ROUTE 2 — remove the source-color validation safeguard
**Hypothesis:** on all 266 visible inputs `source_color == selected_color` and
`hash_oob` is never true, so the validation subgraph (2nd-color + first-anchor +
hash-validate, nodes 3-36) is dead weight → prune → **2549 (−303)**.
Built: `prototypes/_task005_v50_no_validation.onnx`
(sha `442866c93a529bde2baae3da6f1d8a8a057c2be5ec1d8feff91b95a4a7e290cb`).
**Gates:** visible **266/266 PASS**; profile **2549** (< 2852 ✓);
`ab --full` = **1468 divergences / 200266 (byte_equal=False)**;
`truth --full` = **198532/200000 (99.266%)**.
**DEAD:** DUAL-TRUTH → cand 1468 misses vs LIVE 1190 misses = **net +278 WORSE**,
and not byte-equal. The validation fires on ~0.73% of the broad distribution and
is **net-positive for correctness → forced.** (ONNX has no cheap data-dependent
conditional — If = subgraph, banned by the scorer — so the always-recompute
duplication cannot be skipped byte-identically either.)

## Other checks (no win)
- **Stale value_info free-win:** none. Graph is clean (84 nodes, 83 value_info,
  0 stale, 0 init/vi overlap).
- **GridSample 320 → cheaper:** F32 is forced (GridSample outputs input dtype);
  any Gather/Einsum alternative needs a materialized color grid (≥3600) or is
  byte-identical in size. No win.
- **F16 moment/mean/Round chain:** precision-critical (a 0.5 flip moves the
  anchor 1 cell → not byte-equal). High risk vs the byte-equal gate, ~150 B EV.
  Not pursued.

## VERDICT: WALL at 2852 for pure cost-ratio compression
Forced ≈ paint 1341 + irreducible detection (moments→anchor, hash→motif-table,
GridSample→marker-colors, 2× stamp conv) + validation (proven net-positive). The
prompt's ~900 floor_est is unreachable — it ignores the forced 441 pre-pad and
all detection; paint alone forces 1341. Only a *more-correct-than-LIVE* graph
(fixing some of LIVE's 1190 misses) at ≤2852 could win, but the prompt scopes
this task as compression only ("do not expect solve-from-scratch unlock
surprises"), and the misses may be inherent tie-break ambiguity.

**One remaining hypothesis for a future batch (low confidence):** audit whether
LIVE's 1190 truth-misses are inherent ambiguity or a fixable heuristic gap in
the moment/hash anchor detection. If fixable AND expressible without adding net
bytes, a byte-*different* but more-correct successor could ship (DUAL-TRUTH:
cand_misses < LIVE_misses). This is a correctness project, not compression.

---

## Reviewer addendum (bo_ng_cld_reviewer2, 2026-07-08)

Independently re-ran all gates with `$NEUROGOLF_PYTHON` (did not trust pasted
numbers). **Confirmed:**
- LIVE profile = **2852** (memory 2662 + params 190); manifest-authoritative.
- v50 no_validation probe (sha `442866c9...a7e290cb`) = profile **2549**, visible
  266/266, but `ab` capped 8000 = **7 divergences / byte_equal=False** AND `truth`
  8000 = **7948 vs LIVE 7955** (7 MORE misses) -> DEAD on both paths. Worker right.
- The +1.15 structural prize (2852->900) is **DEAD**: stale premise, no F32 masks,
  Aditya already rewrote to a near-forced-floor 2852 graph. Worker right.

**Correction to "no stale value_info / graph is clean":** the worker checked only
for orphaned value_info *annotations*, not **dead nodes**. Reverse-reachability
from the graph output finds **2 fully-dead nodes** (outputs consumed by nothing
that reaches `output`):
- `Concat -> _v60_anchor_xy_2d` (F32[2], 8 B charged)
- `Cast '_h_top_u8_vec' -> _v61_top_bit_u8` (U8[1], 1 B charged)

Their inputs (`_v60_anchor_col_abs/_row_abs`, `_h_top_f`) stay live -> no cascade.
Stripping the 2 nodes -> profile **2843** (memory 2653 + params 190), **-9 B**.
Byte-equal proof: removed nodes are not in the output's reverse-reachability
closure, so removal is output-preserving by construction; confirmed empirically
`ab --full` vs LIVE = **200266/200266, divergences=0, byte_equal=True**; visible
266/266; truth 8000 = 7955 (identical fail set to LIVE).

Value: `points = 25 - ln(cost)` (fit exact against manifest) -> **+0.00316 pts**.
Staged byte-equal at `staged_candidates/task005_deadnode_strip_cost2843/`
(sha `839adc0b...d24093`). **VERDICT: SHIP as SAFE_BATCH_QUEUE** (discretionary
byte-equal micro-filler, zero LB risk) — NOT the structural prize, which is dead.
No further byte-equal compression identified; 900/441/320 are structurally forced.
