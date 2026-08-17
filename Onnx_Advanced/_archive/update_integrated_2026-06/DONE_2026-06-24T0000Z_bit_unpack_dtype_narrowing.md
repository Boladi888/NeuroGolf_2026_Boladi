# Bit-unpack dtype-narrowing: run the per-cell bit-expand in the SMALLEST dtype, not the packed word's dtype

## What I discovered

When a graph carries data as packed integer **row-words** (the bitboard / packed-row family,
techniques #45/#47) and must finally re-spatialize them into a per-cell BOOL plane, the standard
exit is:

```
target_col [1,R,1] U32  AND  pow_mask [1,1,W] U32  ->  bits [1,R,W] U32   (Greater>0)-> BOOL plane
```

That per-cell `BitwiseAnd` output plane is charged at **4 bytes/cell** because the AND inherits
the U32 word dtype. For task077 that plane is `[1,20,21] U32 = 1,680 B` — the single dominant
charged tensor (23.8% of the whole graph), and it had been written into the floor proof as
"structurally required, cannot be skipped" (Worker-10, 2026-06-22).

**It is NOT required.** A bit-TEST only needs 1 byte/cell. Split each packed word into ceil(W/8)
**UINT8 lanes** *before* the per-cell expand:

```
lane_k = Cast(BitShift(word, 8*k, RIGHT), UINT8)         # [1,R] U8, holds bits 8k..8k+7
lane_col_k [1,R,1] U8  AND  pow8 [1,1,8] U8  ->  [1,R,8] U8   (Greater>0)-> [1,R,8] BOOL
target_bool = Concat(lane_b0, lane_b1, ..., axis=-1)     # [1,R,W] BOOL
```

Now every per-cell plane is 1 byte/cell. For task077 (W=21 -> lanes of 8/8/5 bits) the unpack
dropped from `1,680 (U32 AND)` to `420 (sub-AND U8) + 420 (Greater BOOL) + 420 (Concat BOOL) +
~300 (lane shift/cast)` — **net -637 on the whole-graph cost (7,161 -> 6,524)**, all gates still
byte-equal (visible 266/266, truth-full 200000/200000, ab-full 200266/200266, static==profile).

## Where I encountered it

- task077 (W5, 2026-06-24). Builder: `The_400_Tasks/Task_by_Task/task077/prototypes/build_task077_u8lane_unpack_probe.py`.
- Staged candidate: `staged_submits/task077_v3_u8lane_unpack_cost6524/` (cost 6,524, SHA BF353D10...).
- The prior floor proof in `Task_by_Task/task077.md` had the wrong "U32 unpack 1,680 forced" line;
  I corrected stage-2 of the FLOOR PROOF and the Worker-10 don't-do entry there.

## Why it matters

This corrects a stated floor that was off by 9% of the whole graph cost. The error pattern is
general: an agent prices a "forced" charged plane at the dtype of the *upstream* tensor, when the
*operation that produces the plane* (a bit-test, a comparison, a mask) actually emits a value that
fits in a smaller dtype. ORT charges `max(static,runtime)` per tensor by its declared dtype, so
narrowing the dtype before the rank-full per-cell expansion is a real, repeatable saving.

General rule for the library: **when a per-cell plane is the dominant charged tensor and it is
produced by a bit-test / threshold / mask, check whether the test result fits in U8 or BOOL (1 B).
If the source is a wide integer, split it into U8 lanes BEFORE expanding, then Concat the BOOL
sub-planes.** The lane split costs only the (cheap, [1,R]-shaped) BitShift+Cast per lane plus the
Concat; the win is `(B_word - 1) * R * W` bytes on the expanded plane.

## What's missing from the library

- `op_cost_patterns.md` and `techniques.md` (#45/#47 packed-row / bitboard, #58b bit-expansion)
  describe the U32 unpack but do not mention the U8-lane dtype-narrowing on the *exit* expand.
  `ort_reality.md` line ~97 even states "per-cell unpack broadcast is >=2,500B INT32 ... priced
  DEAD" for task338 — that conclusion should be re-checked for the U8-lane narrowing (task338 is
  a [25,25] per-cell rule; W=25 -> 4 U8 lanes, the same trick may revive a route priced dead there).
- A short cross-ref under #58b / the packed-row §: "the EXIT expand should run in the smallest
  dtype holding the bit, not the packed word's dtype; split to U8 lanes + Concat BOOL."

## Verification gates (all passed, candidate at 7161->6524)

```
score_v2 static        : (0, 6424, 100)  -> 6524
ng_gate visible --t 77 : 266/266 PASS
ng_gate truth  --full  : 200000/200000 (100.000%)
ng_gate ab     --full  : 200266/200266 byte_equal=True (divergences=0)
profile_real_cost 266  : static==profile==6524
```
