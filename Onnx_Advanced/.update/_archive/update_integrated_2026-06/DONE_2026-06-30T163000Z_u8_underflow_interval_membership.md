# UINT8-underflow interval membership: `low<=r<=high` in ONE Sub + ONE compare (was 3 planes)

## What I discovered
A per-cell **bounded-interval** membership test — "is coordinate `r` within the per-column
interval `[low[c], high[c]]`?" — collapses from THREE full N×N planes to ONE Sub + ONE compare,
using UINT8 wraparound. This is the interval-test cousin of the `u8_diff_reducemin_membership`
drop (DONE_2026-06-30T152747Z): same U8-underflow lever, different pattern (bounded interval vs
equality projection).

The standard pattern (and what task379 LIVE/v3 used) is two comparisons + an And:
```
ge   = GreaterOrEqual(row_vals, low)   # BOOL  N×N
le   = LessOrEqual(row_vals, high)     # BOOL  N×N
in_b = And(ge, le)                     # BOOL  N×N        -> 3 planes
```
where `row_vals` is the broadcast coordinate `[1,1,N,1]` and `low`/`high` are per-column bounds
`[1,1,1,N]`. Three `[1,1,20,20]` planes = 1200 B on task379.

The underflow rewrite — `r∈[low,high] ⟺ (r-low) mod 256 <= (high-low)`:
```
span  = Sub(high, low)            # UINT8  [1,1,1,N]  tiny (20 B)   = high-low
t     = Sub(row_vals, low)        # UINT8  N×N        (400 B)        wraps mod 256
in_b  = LessOrEqual(t, span)      # BOOL   N×N        (400 B)
```
Two N×N planes + one tiny line instead of three N×N planes. Byte-identical output.

## Why it's correct (the wraparound does both bounds at once)
- `low <= r <= high`  ⇒  `0 <= r-low <= high-low = span`  ⇒  `t <= span` TRUE.
- `r < low`           ⇒  `r-low` underflows to `256+(r-low)` ≥ ~237  ⇒  `> span` (span tiny) ⇒ FALSE.
- `r > high`          ⇒  `t = r-low > high-low = span`                                    ⇒ FALSE.

The single `<= span` test rejects BOTH the below-low side (via underflow to a huge value) and the
above-high side (directly), because span is small. **Safety condition:** `span = high-low` must stay
small relative to 256 (true for any grid coordinate ≤ 30), AND the "poison/empty" encoding must also
fail the test. task379 poisons red-free columns with `low=30, high=0` → `span = 0-30 = 226`,
`t = r(1..20)-30 = 227..246 > 226` ⇒ all FALSE (no ray) — verify your empty-interval sentinel lands
outside `[0,span]` the same way.

## Requirement / trap
UINT8 `Sub` has no kernel below **opset 14** (ort_reality #41: I8/U8 Mul/Add/Sub arrived at 14;
INT32 is the floor below). Bumping opset 13→14 is free for this graph (the sanitizer and grader
accept 14; AB stayed byte-equal). `LessOrEqual` on UINT8 is fine at 14.

## Where / gain
- task379 (Filling-Regions, ray render). `The_400_Tasks/Task_by_Task/task379/prototypes/build_task379_v4_ray_underflow.py`.
- Staged: `The_400_Tasks/Task_by_Task/task379/staged_candidates/task379_v4_ray_underflow_cost8047/`.
- **8427 → 8047 (−380 B), byte-equal to LIVE** (visible 266/266; truth 200000/200000; AB 200266/200266
  divergences=0 byte_equal=True; profile 8047, 0% gap). The v3 floor proof had treated the 3-plane
  ray interval as forced; this rewrite broke it. After it, the rest of the graph is a genuine floor
  (forced-component argument in task379.md Cost analysis).

## What's missing from the library
The library documents the equality-membership underflow (sibling drop) but has no entry for the
**bounded-interval** case: fold `Ge + Le + And` (3 planes) into `Sub + LessOrEqual(span)` (2 planes)
via U8 wraparound. Common in ray/interval/region-fill renders where each column/row has a
data-dependent `[low,high]` span.

## Sorter note: Onnx_Advanced/op_cost_patterns.md — add as new technique "u8_underflow_interval_membership" (bounded interval `low<=r<=high` → `LessOrEqual(Sub(r,low), Sub(high,low))`; 2 planes vs Ge+Le+And's 3; requires opset 14 U8 Sub; safety: span≪256 and empty/poison sentinel must fall outside [0,span]). Measured −380 B on task379 (8427→8047, byte-equal). Sibling of u8_diff_reducemin_membership (equality projection). Exemplar: build_task379_v4_ray_underflow.py.
## Sorter note: Onnx_Advanced/by_symptom.md — add route: "per-row/col bounded-interval membership (is r in [low,high]?) → u8_underflow_interval_membership (not GreaterOrEqual+LessOrEqual+And)".
## Sorter note: Onnx_Advanced/WORKER_PROTOCOL.md "Ten most-recent verdicts" — replace oldest line with: "#NNN u8-underflow interval membership: `low<=r<=high` = `LessOrEqual(Sub(r,low), Sub(high,low))`, 2 planes not Ge+Le+And's 3; opset14 U8 Sub; span must be ≪256 and empty-sentinel must fall outside [0,span]. task379 −380 B byte-equal (8427→8047)."
