# DEAD ROUTE — task243 CarrierNarrowEqualTail (~900 forced-floor attack)

- **Batch/worker:** bo_ng_cld_worker1 (Claude lane), 2026-07-11
- **Live bar (ng_current_bar):** manifest **6985** (Vamshi), onnx sha `8c3e6ba54a75611d8ff629caa97ba1db40c576647b6465b1d3c41c7b45b09745`, updated 2026-07-02.
- **Effective-best ceiling (local staged):** **4718** — `staged_candidates/task243_regionfillrewrite_cost4718/task243.onnx`, sha `f97ee83e18141a5cb1fefdec94f36da64c005f5c0f3e6e2b1792a89517d4f8aa` (re-verified score_v2 = mem 3816 + par 902 = 4718, macs 0, points 16.5409).
- **Prompt target:** ~900 via "carrier-narrow sweep: collapse 10-channel/F32 spatial carrier to U8/color-index; one spatial predicate/color-index plane plus output tail."

## Verdict: DROP / WALL — no candidate strictly below 4718 by a floor route.

### 1. The charged component named in the Direction is ALREADY GONE (2026-07-10 workflow DROP condition)
The Direction attacks a "10-channel/F32 spatial carrier … before Conv/Reduce/Where" and "repeated Conv masks." That is the ORIGINAL 117k model. Both the manifest team-best (6985, Vamshi 2-Einsum) and the effective-best (4718, single-Einsum) are from-scratch Einsum rewrites: single-channel `[1,1,18,18]` carriers, **zero Conv nodes**, no 10-channel carrier. There is no charged carrier left to narrow.

### 2. The ~900 floor is a FALSE floor — fast probe on 3200 fresh ARC-GEN pairs (seed_offset=2026)
`prototypes/fast_probe_carrier_floor.py` (exact-floodfill rule == ARC-GEN truth 3200/3200):

| Question | Result | Meaning |
|---|---|---|
| Q1 fill-region real matrix rank | min 2, **p50 14**, mean 14.17, p90 17, max 18 | fill spans the grid; not low-rank |
| rank == 1 (separable / free-output-Equal-able) | **0 / 3200 (0.0%)** | a separable Equal-tail cannot render the fill |
| rank ≥ 3 | 3199 / 3200 (100%) | ~14 rank-1 terms needed per grid |
| filled 4-conn components / grid | p50 10, mean 10.30, max 28 | ~10 disjoint blobs — not one predicate plane |
| grids with blockers (≥ color 2) | 3200 / 3200 (100%), mean 44.5% of cells | blockers omnipresent |
| fill region borders a blocker | **3200 / 3200 (100%)** | per-round gating is load-bearing every grid |

The ~900 target = the output tail (30×30 BOOL Where condition = 900 B) **only**. It assumes the flood is free. The probe shows the flood is rank-~14 data-dependent connectivity across ~10 disjoint CCs — it can NOT be a free-output separable Equal (#109: a 2-D region complement is not separable-Equal). A full 2-D state carrier is forced.

### 3. The `allowed` gate carrier cannot be dropped
- H2 ungated flood (dilate seeds, ignore blockers) == ARC-GEN truth: **1 / 3200 (0.03%)**.
- H3 dilate-then-AND-allowed-once == ARC-GEN truth: **1 / 3200 (0.03%)**.
Per-round gating with a materialized `allowed` mask is mandatory (blockers 100%, fill-borders-blocker 100%).

### 4. Carrier-narrow to U8 is self-defeating here
The two F32 carriers (gate 1296, state 1296) are F32 *because they are Einsum operands*, and the Einsum is what folds all 45 flood rounds into ONE materialized `state` (the reason 4718 < 11,315). ORT Einsum has no U8/int8 kernel. Narrowing to U8 forces per-round bitwise dilation (~36 materialized intermediates) = the documented **11,315 U8-lane paradigm = 2.4× MORE expensive**. "Collapse F32 carrier to U8" LOSES for this graph.

### 5. Forced-component floor arithmetic (every byte information- or paradigm-forced)
| component | bytes | why forced |
|---|---:|---|
| output tail `fill_b30` [1,1,30,30] BOOL | 900 | non-uniform 30×30 Where cond; no broadcast, no sub-30 |
| `state` [1,1,18,18] F32 (flood result) | 1296 | rank-~14 non-separable; Einsum output dtype |
| `gate` [1,1,18,18] F32 (allowed) | 1296 | blockers load-bearing 100%; Einsum operand |
| `fill_b` [1,1,18,18] BOOL (cast bridge) | 324 | cheapest cast placement (before 18→30 pad) |
| params R[18,30]+M[18,18]+sel/pads | 902 | Einsum operands must be F32; slicing R = +1296 net loss |
| **sum** | **4718** | = current effective-best |

### Remaining hypothesis (one, low-confidence, NOT worth a re-crack)
Micro-fusion of the cast/params might extract ~100–300 B (→ ~4400–4600), but (a) that is a `rung − ε` shave the 2026-07-10 workflow explicitly excludes, (b) it perturbs the exactly-52-letter-budget Einsum, and (c) it is below a confident strict beat. No floor route to ~900 exists; the task is intrinsically iterative connectivity with omnipresent blockers.

---

## PASS 2 (forced second pass, 2026-07-11) — both structural levers falsified by gate/numerics

Cost is fixed by the two F32 carriers, NOT the round count (§37: round count is FREE inside one Einsum). So the only levers below 4718 are (A) narrow a carrier's dtype, or (B) remove a carrier. Both now falsified with fresh artifacts (not reasoning):

**Lever A — narrow carriers F32→F16 (halve to ~3422): DEAD (numeric overflow).** `prototypes/probe_f16_range.py` on 1500 ARC-GEN grids: the flood `state` holds **walk counts** (sums of 0/1 products over the band), whose value range is **[1, 5.97e19]**. That overflows F16 (max 65504) on **100%** of grids → inf, then `inf·0 = NaN` at the next gate → corrupts the whole plane. This is the mechanism behind the 2026-05-20 "catastrophic adversarial fail." The range 6e19 also exceeds int32 (2.1e9) and int64 (9.2e18) — among ORT-Einsum dtypes only a **floating exponent (F32)** holds it. F32 carriers are numerically forced.  (A MAX-based flood keeps values in {0,1} but MaxPool materializes per-round = the 11,315 paradigm; Einsum sums only.)

**Lever B — remove the `gate` (allowed) carrier (drop to 3412): DEAD (visible 0/265).** `prototypes/build_task243_dropgate.py` builds the single-carrier band-only flood.
- `score_v2` = mem 2520 + par 892 = **3412** (−1306 vs 4718). `ng_graph_cost_diff` names the delta exactly: `gate` float32 [1,1,18,18] removed (−1296) + `gate_sel` (−10) + one Einsum. So the cheaper structure is real and its saving is precisely the gate carrier.
- `ng_gate visible --task 243` → **0/265 (0.00%) FAIL** (ungated flood leaks through blockers; matches the fast-probe 0.03% truth rate). Per traps §VISIBLE-FAIL a visible zero is a certain hidden 0 → dead; no truth/ab run (hygiene).

**Net:** the sole structure removable below 4718 is the `gate` F32 carrier (1306 B), and it is correctness-forced (visible 0/265); the sole dtype narrowing (F16) NaNs on 100% of grids. 4718 is the floor — now proven by a gated cheaper-but-wrong sibling and a numeric dtype kill, not by argument alone.

Library cross-refs mined this pass: §37 (round count free; doubling/log-dilation provably inexact for masked flood), §50 (crossing-parity interior fill — applies only to CLOSED rectangular rings, not seed-reachability, so inapplicable here), §33/§34/dominance-flood (per-step ≥880 B → per-round paradigm, costlier).
