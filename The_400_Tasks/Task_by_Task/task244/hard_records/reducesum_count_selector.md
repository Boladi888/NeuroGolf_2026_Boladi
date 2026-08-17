# task244 hard record - ReduceSum-count selector (drop the ArgMin sentinel)

Live/team bar: cost 85, profile_cost 85, sha256 db55e34c9cd94f6da4514281bdd2a377d1b4783c59a83e8d62f09353a60f175e
(`ng_current_bar.py --task 244 --json`, authoritative source `manifest`, owner Vamshi, updated 2026-07-02T02:35:15Z).

Live graph = 6-node RoiAlign: `Einsum(deg-11 width fingerprint r11) -> Greater(thresholds[8]) -> Cast(U8) -> ArgMin(first-unpassed) -> Gather(roi_table[8,1,4]) -> RoiAlign`.
Static score `(macs=0, memory=44, params=41)` -> 85. params = roi_table 32 + thresholds 8 + batch_idx 1. memory = r11(4)+passed bool[8](8)+passed_u8[8](8)+bucket_i64 scalar(8)+rois f32[1,4](16).

Attempted route: the 8th threshold is a `1e30` sentinel that exists ONLY so `ArgMin(first-unpassed)` finds a `0` for the top bucket. Replace `ArgMin` with `bucket = ReduceSum(r11 > thresholds)` counting exceeded separators, which needs only **7** real thresholds (drop the sentinel). Bucketing is mathematically identical (same threshold boundaries), so truth stays exact.

Prototype: `The_400_Tasks/Task_by_Task/task244/prototypes/task244_reducesum7_roialign.onnx`,
sha256 e106e570162f28371f1a4ee831aea74936a9ca7c10a2525e78de0b6b1f5aa412.
Nodes: `Einsum -> Greater(thresholds7) -> Cast(INT32) -> ReduceSum -> Gather -> RoiAlign`.

Gate results:
- `ng_gate.py visible --task 244` -> `[visible] 266/266 (100.00%) PASS` (bucketing is exactly equivalent to live -- route is algorithmically valid).
- static score `(macs=0, memory=59, params=40)` -> **cost 99 > 85 live. LOSES.**

Why dead:
1. `ReduceSum` rejects UINT8 (`Type 'tensor(uint8)' of input parameter (passed_u8) of operator (ReduceSum) ... is invalid`). A u8-count variant (`task244_reducesum7_u8_roialign.onnx`, keeps the width-K intermediate at 1 B/elem) fails ORT load for the same reason.
2. `ReduceSum` only accepts INT32/INT64/FLOAT+, i.e. >=4 B/elem. Casting the width-7 comparison plane to INT32 makes `passed_i32[7]` = 28 B, vs the live `passed_u8[8]` = 8 B. That +20 B of memory dwarfs the -1 param saved by dropping the sentinel. Net cost 99 (memory 59, params 40).

Floor consequence: the 8-way bucket selector floor is `Greater(bool[8]) -> Cast(u8[8]) -> ArgMin(int64)` as live. Any first-hit argmin/argmax selector needs 7 real separators + 1 boundary sentinel = 8 comparison points; the only selector that needs 7 (count-exceeded) forces a >=4 B/elem reduce input and loses. The u8 cast and int64 argmin output are both forced by ONNX op type contracts.

One remaining hypothesis: a sub-85 win must **delete a forced structural component** (the roi_table[8,1,4]=32 geometry table, or the rois[1,4] f32=16 RoiAlign input), not shave the selector. No cheaper op family than RoiAlign is known for a width-dependent flipped affine resample; treat 85 as the structural WALL until such an op or a table-free ROI generator appears.
