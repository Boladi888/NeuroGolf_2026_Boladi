# task322 sub-80 direct-output routes — WALL (floor = 80, forced)

Live bar: **80** bytes (memory 0 + params 80), sha256
`f41420db40ac99402c50068f91ff678c8cd63798b51fc59abba376ef80835ab4`
from `python tools/ng_current_bar.py --task 322 --json`
(`authoritative_source=manifest`, owner Vamshi). Confirmed live `score_v2 = (macs 0, memory 0, params 80)`.

## Rule (decoded, verified 266/266)
3×3 grid, at most ONE nonzero (color) per column (verified: max nonzero/col = 1 over all 266).
Gravity fill: each column's color falls down and fills to the bottom. In one-hot [1,10,30,30]:
- color channel c: out row r > 0  iff  color c appears at some input row ≤ r  (cumulative **OR**).
- channel 0 (bg): out row r > 0  iff  input rows 0..r are **ALL** background  (cumulative **AND**).
- off the top-left 3×3 (rows/cols ≥3): output must be ≤0.

## Gate contract (why the floor is what it is)
`session.run_raw` thresholds the raw output at `>0` → 1.0/0.0, then `visible`/`truth`
do a **full-tensor `np.array_equal`** against `encode_grid(expected)`. So the graph output
(post >0) must be exactly the expected one-hot on the 3×3 AND exactly 0 everywhere else.
Off-grid suppression is FREE: off-grid input is all-zero, so a Conv there outputs `bias`;
`bias ≤ 0` ⇒ off-grid ≤0. Kernel height is therefore spent ONLY on packing the three
independent output rows, not on suppression.

## Forced-floor argument (memory=0 ⇒ single terminal op ⇒ depthwise Conv ⇒ K≥7)
1. **memory must be 0.** `calculate_memory` sums `num_elements×itemsize` over every
   intermediate (input/output excluded). Any full-canvas intermediate = [1,10,30,30]·4 = 36,000 B;
   a spatially-sliced [1,10,3,3] intermediate = 360 B (F32) or 90 B (1-byte dtype) — already >80,
   and reaching 3×3 needs a Slice (adds index initializers + a second intermediate). So a sub-80
   graph must have ZERO counted intermediates ⇒ exactly ONE node whose output is `output`.
2. **The one op must differentiate channels.** ch0 needs cumulative-AND, colors need cumulative-OR.
   A single 0-param pooling op (MaxPool/CumSum/ReduceX) applies the SAME reduction to every channel,
   so it cannot make ch0 behave differently from colors. (MaxPool gives colors right but ch0 = OR,
   which stays ON at filled cells → two channels >0 → array_equal fail. Verified reasoning; matches
   worker1's `maxpool_k3` visible 0/266.) Per-channel behavior ⇒ per-channel parameters ⇒ Conv.
3. **Depthwise (group=10) is the cheapest per-channel Conv.** kernel [10,1,K,1] = 10K weights;
   group<10 only enlarges the kernel ([10,10,K,1]=100K). Bias length must equal out_channels=10
   (sub-length bias = traps §8h VISIBLE-UNSTABLE; recent-verdict #9). Bias is load-bearing:
   worker1 `conv_nobias70` = visible 0/266 (ch0 AND needs the negative threshold).
4. **K ≥ 7 is forced.** With one shared per-channel bias, ch0's cumulative-AND needs each output
   row (0,1,2) to apply a DIFFERENTLY-SCALED linear map (row1 threshold b∈(−2w,−w], row2
   b∈(−3w,−2w] are disjoint for a shared b), so the three rows cannot share taps. An exhaustive
   LP feasibility scan (`prototypes/search_geometry_lp.py`) over **K∈[3,7], stride∈[1,5],
   dilation∈[1,4], every pad, ALL 30 output rows constrained OFF for rows≥3** finds the depthwise
   Conv feasible for BOTH ch0 and color channels at **exactly one** geometry:
   `K=7, stride=3, pad=6, dilation=1` — the live graph. Every K<7 is INFEASIBLE.

⇒ cost = 10·7 (weights) + 10 (bias) = **80**, uniquely realized by the live graph. **WALL.**

## Perfect-25 / cost≤1 rumor slot — DISPROVEN
Prompt hypothesis: a zero-param Identity/Transpose/flip/rotate. Over all 266 examples:
identity 0/266, transpose 0/266, hflip 0/266, vflip 0/266; gravity-fill 266/266. The task is a
genuine data-dependent per-column fill, NOT a whole-grid isometry. No cost≤1 direct-output op exists.

## Exact failed gates (this batch + worker1)
- LP proves no sub-80 single-Conv geometry exists → **no candidate built to gate** (per kill rule).
- worker1 `conv_nobias70` (70): `[visible] 0/266 (0.00%) FAIL`.
- worker1 `maxpool_k3_legal` (0): `[visible] 0/266 (0.00%) FAIL`.
- worker1 `maxpool0`: ORT rejects Conv-like bottom pad on Pool.
- worker1 `u8_cumsum_reject`: ORT rejects UINT8 CumSum.

## Remaining hypothesis (only conceivable escape)
Some non-Conv single terminal op with <80 constant elements that expresses BOTH per-channel
cumulative-AND and cumulative-OR with off-grid suppression. None known in the ORT opset that
prices below the depthwise Conv; treat as closed unless a new op-cost verdict lands.
