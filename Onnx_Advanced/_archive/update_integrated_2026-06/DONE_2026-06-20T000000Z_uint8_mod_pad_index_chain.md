# UINT8 row-index chain: Mod+Pad savings when indices fit in U8

## What was discovered

In task110, the fill path uses:
- `selected_rows_mod = Mod(arange29 INT32[29], selected_period INT32[])` -> INT32[29] = 116B  
- `selected_rows30 = Pad(selected_rows_mod, [0,1], sentinel_i32=9)` -> INT32[30] = 120B  
- `padded_labels = Gather(selected_compact_pad, selected_rows30, axis=2)` -> U8[30,30]

The values in `selected_rows_mod` range from 0..8 (Mod period 9) and `selected_rows30` adds a 9 at the end. All fit in UINT8.

**The optimization**: insert `Cast(arange29->U8)` [29B] + `Cast(period->U8)` [1B], change Mod to produce UINT8[29]=29B and Pad to produce UINT8[30]=30B, then `Cast(rows30_u8->INT32)` [120B] before Gather. Result: remove 236B (116+120), add 180B (29+1+29+30+120), net **-56B memory saved**.

Minus 1 new param (the UINT8 sentinel constant for Pad, value=9) = **-26B total cost**.

## ORT support verification (all opset 13)

- `Mod(UINT8, UINT8) -> UINT8`: WORKS
- `Pad(UINT8, INT64_pads, UINT8_const) -> UINT8`: WORKS
- `Cast(INT32 -> UINT8)` scalar: WORKS
- `Cast(UINT8 -> INT32)`: WORKS
- `Gather(UINT8_data, INT32_indices, axis=N)`: WORKS (INT32 and INT64 both accepted)
- `Mod(UINT8, INT32)` MIXED DTYPE: FAILS (type mismatch)

## Key constraint

`Gather` requires INT32 or INT64 indices (NOT UINT8). So after the UINT8 Pad, you MUST Cast back to INT32/INT64 before Gather. This adds a charged intermediate. The savings come from the intermediate BEFORE the Cast (the rows_mod and rows30 tensors).

The same optimization applied to `selected_2d_cols_mod` (Mod(arange29, 2D_period) -> INT32[29]=116B) costs +30B WORSE because the Cast back to INT32 for Gather[axis=3] costs exactly as much as the UINT8 version saves.

## Where encountered

- Task: task110  
- Staged: `staged_submits/task110_u8idx_cost7869/`  
- SHA256: `ad2fb92155c62185b6a4643aef7f9aff8cbaf281172c8bf4f81f6eff151436cc`
- Cost: 7895 -> 7869 (-26)

## Why it matters

When a Mod result feeds a Pad then a Gather, and the values are small (< 256), the UINT8 chain saves ~57B per instance minus the cost of a Cast-to-INT32 at the end (120B for INT32[30]). Net positive when the pre-Cast chain has >1 intermediate (rows_mod AND rows30 here). Only the ROW path was profitable; the COL path had just 1 intermediate before Gather so was neutral/worse.

## What's missing from the library

This could be added to `techniques.md` as a new technique about "U8 index chain: when Gather needs INT32 indices but Mod produces small values, insert U8 intermediate before Cast." Or extend `cost_floor_reference.md` with the rule: `Mod(arange, small_period)` -> INT32 intermediates can be U8 when values fit, but must Cast to INT32 before Gather.
