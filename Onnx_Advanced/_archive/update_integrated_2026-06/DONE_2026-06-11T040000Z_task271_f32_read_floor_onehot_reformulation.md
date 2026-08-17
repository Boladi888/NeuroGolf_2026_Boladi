# Two transferable findings from task271 (crop9x9 sliding-window box-select)

Worker-9, 2026-06-11. Build scripts: `The_400_Tasks/Task_by_Task/task271/prototypes/build_v4.py`
(best, F16) and `build_v6.py` (F32 ref). Best prototype `task271_v4.onnx` cost 1942,
byte-equal to LIVE on 200,267 pairs. **NOT staged** — team-best (Aditya) is 1311, so a
1942 byte-equal candidate is 0-EV.

## Finding 1 (cost law — likely missing or under-stated in cost_floor_reference.md)
**The "F32 input read floor": reading K channels of the [1,10,30,30] F32 input via Slice
costs K × (H_crop·W_crop·4) bytes, irreducibly, because Slice preserves dtype.** You cannot
get a channel into F16/U8 without FIRST landing it as an F32 Slice output (which is charged)
and THEN paying an extra Cast intermediate. So dtype tricks (F16/U8/ConvInteger) shrink the
*downstream* maps but never the initial per-channel read. For task271 the rule provably needs
2 channels (blue for score+extract, bg/cyan for solidity), giving a hard 648 B floor on a 9×9
crop — this alone is most of the gap to a sub-650 target. Practical rule: **before promising a
"~650" target on a crop task, count `n_channels_read × crop_area × 4` first; that's your floor
and casts only add to it.** (Confirmed: F16 pipeline netted only −238 vs all-F32 because the
two crop casts cost +324; ConvInteger output is INT32 = no win over F16's 2-byte maps.)

## Finding 2 (technique — output one-hot reconstruction beats 10-channel gather)
For "extract one object/block and place it" tasks where the **output palette is closed**
(here strictly {0,1,8}, guaranteed by reading the ARC-GEN generator), do NOT carry the full
10-channel one-hot through the graph or do a 10-channel Gather (that forces a ~900 B
[1,10,3,k] intermediate or a 3240 B [1,10,9,9] crop). Instead:
 - Gather/extract only the ONE discriminating channel (blue) at the winning location → [1,1,3,3].
 - Reconstruct the output one-hot by Concat of that channel + its complement + zero blocks:
   ch1 = blue_block, ch8 = `Not(blue_block)` (valid because a *solid* box has no bg, so every
   cell is color 1 or 8), all other channels = zero initializers. Emit the [1,10,3,3] block as
   **BOOL** (decoder thresholds `>0`, one-hot is 0/1) → 90 B not 360 B, then Pad.
This took task271 from the LIVE 7145 → 1942 (the 10-channel `input_cropped` 3240 B and the
900 B gather both vanish). The closed-palette + solid-region invariant (no bg in the answer
block) is what makes `ch8 = NOT ch1` exact — verify it in the generator before relying on it.

## Don't-do confirmations (already written into task271.md, repeating for the sorter)
- Dynamic Slice with runtime `starts` for a fixed-size crop → `dim_param` output → validator
  reject. Use two static-length Gathers (index length statically known = 3) instead.
- Blue/single-discriminator-only ArgMax (dropping the solidity channel) failed 5/267 on this
  task: sliding windows straddling two objects out-score the true object. If your scan can be
  fooled by inter-object straddle, you need the occupancy/solidity channel — it's not optional.
