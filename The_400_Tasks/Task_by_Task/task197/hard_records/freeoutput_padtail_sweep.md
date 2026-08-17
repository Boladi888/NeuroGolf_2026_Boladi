# DEAD ROUTE — free-output Pad-tail / color-index collapse (task197)

- **Live bar when tested:** 277 (manifest-authoritative, owner Sho, SHA256
  `4f8eb859bab9ded1783047fd8fa0dc8ae202b4fe0aa3afe9a9e53f9124f45132`, updated
  2026-07-09T01:09:17Z). Live = 263 memory + 14 params. Profile == static == 277.
- **Batch/worker:** batch56 `bo_ng_cld_worker1` (Claude), 2026-07-09.

## What the prompt asked
"FreeOutputPadTailSweep": delete the charged 10-channel prepad/output tensor;
emit one label/mask channel and make Equal/Pad the graph output. floor_est ~110.

## Why it is dead (measured, not argued)
The current best is ALREADY a free-output Gather that materializes **no spatial
plane** — its only large charge is the `[30]` INT32 Gather index (120 B, 45.6%).
Every "free-output Equal/Pad" reformulation is strictly worse:

| Route | Mechanism | Cost | Gate |
|---|---|---:|---|
| free-output Equal/Pad plane | build `[1,1,30,30]` label plane, `Equal(iota)` free out | **7488** | score_v2 (7463 mem + 25) |
| gather10 + output Pad | `[10]` index, Gather→`[1,10,30,10]`, free Pad | **12175** | score_v2 / profile (prior 2026-06-21) |
| 2-column Where broadcast | Slice col0 + Gather colf1, broadcast-select | **≥36000** by construction | inner Where `[1,10,30,30]` = 36000 B |
| independent Gather rebuild | from-scratch column-copy (this worker) | **277 (TIE)** | visible 267/267, profile 277 |

`freeoutput_equal_v1` charged tensors: `colplane_full [1,1,30,30]` F32 3600 +
`outplane [1,1,30,30]` 3600 = the free-output Equal FORCES a full spatial label
plane the Gather route never pays. A cropped `[1,1,14,10]` plane still needs a
`[1,1,30,30]` (≥900 B U8) pre-Equal OR a `[1,10,14,10]` (≥1400 B) pre-Pad — both
already > 277.

## Forced-component floor (why < 277 is unreachable in this formulation)
Output = data-dependent column permutation-with-duplication of the F32
`[1,10,30,30]` input. Cheapest legal realization = `Gather(input, idx, axis=3)`:
- Free `[1,10,30,30]` output ⇒ index MUST be `[30]`; Gather needs INT32/INT64 ⇒
  `[30]` INT32 = **120 B forced** (`[10]`→12000 B pre-Pad; U8/INT16 index illegal).
- Pattern-row colour read is a channel-reducing Conv over F32 input ⇒ F32 output
  (Conv can't emit U8/F16 on F32 input — lib #5); min `[1,1,1,10]` = **40 B forced**.
- Cheapest cast path: U8 selectors (10 B ea) + single `[30]` U8 pad-stage (30) +
  one Cast to `[30]` INT32 (120). Cast-before-pad is +10 B (worse).
- Selector chain (same/active/same_u8/f1/idx_ab/idx10) ≈ 79 B, each forced:
  Where needs BOOL, ArgMin rejects BOOL (needs U8), active mask needed for
  width-8, two Wheres for the 3-way select {col0, f1, zero}.
- 14 params all essential; idx_pads(2) can't drop to Pad-10 attrs (opset-10
  `Equal(float/uint8)` illegal — prior worker 2026-06-22); col0 read is free via
  a 2nd Conv (no init) — the 278→277 param trim.

**floor_est ~110 is below the single forced 120 B Gather index ⇒ unreachable.**
277 is at the floor of the leanest (Gather-index) family, NOT a formulation
artifact above it.

## One remaining hypothesis (only thing that could ever beat 277)
A representation that avoids the INT32 Gather index entirely — i.e. a legal ONNX
op that performs a data-dependent axis-3 permutation-with-duplication into a
FREE `[1,10,30,30]` output while charging < 120 B. None exists in current opset
(Gather/GatherElements/Scatter all need ≥120 B index; MatMul/Where/Equal all
charge a ≥300 B column slice or ≥900 B plane). Revisit only if a new op or a
sub-120 B index-encoding trick appears.
