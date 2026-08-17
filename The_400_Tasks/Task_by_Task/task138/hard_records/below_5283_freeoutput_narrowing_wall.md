# task138 — WALL: strict-below-5283 for the free-output separable-Einsum family

**Recorded:** 2026-07-03 by `bo_ng_cld_worker3` (Claude lane), batch 42 / task138.

## Live bar at time of record
- Manifest team-best: **Vamshi 5403**, sha256 `c077e34a5e6c47f2818e70d5337d672fd4d8962b36cd2de8d1be32d0eb67c97a`,
  updated_utc 2026-07-02T02:35:15Z (`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`).
- Task-local staged win: **5283** (−120), `staged_candidates/task138_cropobjectdetectrewrite_cost5283/`,
  sha256 `7d0b8787c72a1ac5142039553e9d427e20d0d13d2bda580588a175e1afb4cc66`. Fully gated this session
  (visible 266/266 · profile==static 5283 · truth 5000/5000 · **ab byte_equal=True 5266/5266** ·
  output nan/inf scan 0/0 over 1500 real inputs · decode parity 0/1500 vs team-best).

## Route attempted and why it is dead
The named lever ("safe F16 gather-chain narrowing") is **already fully captured at 5283** — it narrows
the two chains whose gather-data provably never overflows F16 (`rowcode` max=3320, `colcode` max=6105,
0/3000 overflow). To go **below** 5283 in this family you must reduce a forced tensor. Enumerated:

1. **The pinv exp-ramp chains (`pinvx`/`pinvy`) — the ONLY remaining tensor-narrowing lever.**
   Narrowing them (the "4-chain" build, `prototypes/task138_f16gather.onnx`, score_v2 **5163**, −120 below
   5283) **emits `nan` on the OUTPUT for 500/500 real inputs** (re-confirmed this session; `inf` grids=0).
   Mechanism: `p_pos`/`p_neg` reach abs-max **662,499 ≫ F16 65,504`; casting the ramp to F16 before the
   Gather makes `inf`, and the downstream Einsum evaluates `inf*0 = nan`. **`truth%` cannot catch it**
   (the `(output>0)` decode maps `nan→false` and the nan lands in should-be-0 cells) — must scan the raw
   output for nan/inf. Dirty / Kaggle-decode-fragile. **DEAD.**

2. **The F32 exp-ramp island** (`rowcnt,colcnt,marker_counts,r_mask,c_mask,prow,pcol,rowcode,colcode`,
   each 120 B / 40 B). Every one is an **operand or output of an Einsum that contracts the F32 graph
   `input`** (nodes 0,1,22,71,73). ONNX `Einsum` requires all operands share a dtype; the graph input is
   fixed F32 and a F16 copy of the `[1,10,30,30]` one-hot input costs **18,000 B** — far above any saving.
   So these tensors are **structurally forced F32**; narrowing any of them makes the Einsum mixed-dtype
   (invalid) with no cast-free path. **DEAD.**

3. **INT32 gather-index chains** (`y_abs,y_idx,x_abs,x_idx`, 120 each) — Gather indices need INT32/INT64;
   no INT16 kernel; INT8 would need re-cast to INT32 before the Gather (net zero / negative). **DEAD.**

4. **rank-5 F16 basis** (`rowfeat_h`/`colfeat_h` 300 each = Concat of 5 spatial rows; `chfeat_h` 220 =
   11 feature-types × 10 ch) — reducing rank/feature-count changes the decoded output. Algorithmic, not a
   narrowing; out of scope for this "recapture" prompt. Not attempted.

## Verdict
**WALL below 5283 for the free-output separable-Einsum family.** The safe-narrowing lever is exhausted at
5283; the single remaining tensor lever (pinv ramp) is falsified by a hard `nan`-on-real-inputs result
(500/500). Any sub-5283 must come from an algorithmically different family (rank reduction or a
non-overflowing positional code), which the prompt explicitly de-scopes ("not a floor-proving task…
report the exact falsification and stop rather than a full siege").

## One remaining hypothesis (not pursued — high risk, explicitly discouraged)
Replace the exp-ramp positional code with a code whose full range stays < 65,504 so `pinvx`/`pinvy` narrow
cleanly (would save ~120 B → ~5163 clean). Blocked because: (a) the ramp's ~12-order dynamic range is what
makes the two separable magnitudes threshold-decode unambiguously across 30 positions — compressing it
risks position collisions in the final `>0` decode; (b) the don't-do log forbids touching any exp-ramp
tensor. Would require re-deriving p_pos/p_neg + a full 200k truth **and** a raw nan/inf scan.
