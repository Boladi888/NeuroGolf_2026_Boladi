# Hard record — 6-row conv-pad carrier is CHEAPER (5722) but VISIBLE-FAIL (bo_ng_cld_worker2, 2026-07-10 pass-2)

## Live bar / state
- Bar: LIVE Sho **6237**, sha `ba3c4625…` (`ng_current_bar.py`, manifest).
- Standing win: `staged_candidates/task157_cheapclamp_tail2ch_cost6008/` (6008, −229, byte-equal on runnable). Effective floor to beat = strict-below 6008.

## Route attempted (genuinely-different tail assembly, second pass)
Shrink the QLinearConv decode carrier from 10 rows to 6 rows and let the conv's own spatial `pads` fill rows 6-9:
- 6008 tail: `ch1_active_pre[1,1,6,15]` →Pad rows6→10 False→ `ch1_active_pad10[1,1,10,15]` (150B) →Where(blue2ch, base2ch_10x1)→ `acg2ch[1,2,10,15]` (300B) →QLinearConv pads`[0,0,20,15]`→ out.
- This route: `ch1_active_pre[1,1,6,15]` →Where(blue2ch, base2ch_6x1)→ `acg2ch6[1,2,6,15]` (180B) →QLinearConv pads`[0,0,24,15]`→ out. Drops the 150B pad AND shrinks the carrier 300→180B.
- Builder: `prototypes/build_task157_carrier6row_v1.py` → `prototypes/task157_carrier6row_v1.onnx` (sha via file).

## Gates (cheapest-first; stopped at the visible wall)
- `profile_real_cost --arcgen 266`: **static=5722 profile=5722 (memory 5422 + params 300)** — −286 vs 6008, −515 vs the 6237 bar. The ~270B reduction is REAL in isolation.
- `ng_gate visible --task 157`: **0/265 (0.00%) FAIL.** Per the VISIBLE-FAIL axiom (traps §VISIBLE-FAIL) a visible miss = certain hidden 0 → truth/ab NOT run.

## Why it is dead — the active grid must be a proper one-hot (conv padding can't render it)
The QLinearConv (x_zp=0, dec_w[k]=[2(k+1),−1], dec_b[k]=−k(k+2)) decodes the internal color code `c` per cell:
`y_k = clip_u8(2(k+1)·c − c² + dec_b[k])`. At a **padded** cell the dequantised input is 0 (input=x_zp), so
`y_k = clip(dec_b[k]) = [0,−3,−8,…]→[0,0,…]` = **all-zero across channels**. A per-tensor y_zero_point cannot encode
a one-hot, so conv padding *provably* renders black as all-zero, never as channel-0=1.

Empirically (all 265 officials, raw output tensor): rows 6-9 in LIVE/6008 = `[1,0,0,…]` (channel-0 one-hot);
rows 10-29 (already conv-padded) = `[0,0,…]` (all-zero). They differ on 1060 rows/board. Feeding a 6-row carrier makes
rows 6-9 all-zero like rows 10-29 → **visible 0/265**. The grader's verify path rejects the all-zero encoding inside the
10×15 active grid (rows 0-9); the conv's all-zero spatial padding is valid ONLY outside the active grid (rows 10-29,
cols 15-29). Therefore the full 10-row explicit render is a hard CORRECTNESS requirement, not merely a byte-equality nicety.

## Consequence — the decode tail is at its forced floor
- **150B `ch1_active_pad10`** (10-row BOOL plane) is FORCED: rows 6-9 must be explicitly rendered channel-0=1; the conv cannot pad them.
- **300B `acg2ch`** (2-channel {c, c²} carrier over the full 10 rows) is FORCED: (a) 3 colors {0,1,2} decoded by a *linear* conv need both `c` and `c²`; (b) the carrier must span all 10 active rows (this record). Padding the bool first (150B) then Where→300B is cheaper than Where→180B then Pad→300B (bool pad 150 < uint8 carrier 180), so the current arrangement is already minimal.

## Verdict: DEAD (visible-fail). 6008 stands as the truth-safe floor.
Nothing staged. This is a gated counterexample: the cheapest INCORRECT tail is 5722; the cheapest CORRECT tail is the 6008 lineage.

## One remaining hypothesis (low confidence, not pursued)
If a future grading stack is confirmed to argmax-decode the active grid AND treat all-zero as color-0 (it does NOT today —
visible 0/265), the 6-row carrier would reopen −286. Re-test the visible gate before ever reviving; do not assume.
