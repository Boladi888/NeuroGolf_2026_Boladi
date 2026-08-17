# BitsetCross architecture floor proof + dead-end inventory (task047)

Session worker: worker-047-1 (2026-06-12 to 2026-06-13)

## Insight 1: Floor proof for BitwiseAnd cross-line architecture (task047)

**What was discovered:**
The BitwiseAnd free-output cross-line encoding (`BitwiseAnd(rowbits[1,1,30,1], colbits[1,10,1,30]) -> output[1,10,30,30]`) has a hard memory floor of 736 B and params floor of 267. Combined cost floor = 1,003 for task047. A full exhaustive analysis was conducted over 2 sessions.

**Memory floor breakdown (736 B):**
- 16 B: 4 F32 coordinate scalars from free-input Einsums — irreducible (each [1] F32)
- 120 B: 4 BOOL equality masks m7r/m8r/m7c/m8c [1,1,30,1] or [1,1,1,30] — irreducible
- 60 B: 2 row bitset Where intermediates (row_not7, rowbits [1,1,30,1] U8 each)
- 240 B: 8 column bitset Where lane intermediates ([1,1,1,30] U8 × 8)
- 300 B: `colbits` [1,10,1,30] U8 — FUNDAMENTAL FLOOR (see proof below)
- FREE: `output` [1,10,30,30] U8 = final BitwiseAnd writes the free output tensor

**Why colbits 300 B is the absolute floor:**
For BitwiseAnd to write the free output [1,10,30,30], it needs one operand at full width (30 in the W dim) and one at full channel depth (10 channels). The minimum-size 10-channel W-dimension operand is [1,10,1,30] = 300 B. Any rearrangement (e.g., transposed [1,10,30,1] × [1,1,1,30]) keeps total at 300+30=330 B. Shrinking colbits to [1,10,1,9] and using Pad doesn't help: the BitwiseAnd intermediate [1,10,30,9] = 2700 B is far worse.

**Params floor breakdown (267):**
- 20: sel7/sel8 F32 [10] — mandatory for free-input Einsum coordinate reads
- 30: ar30 F32 [30] — coordinate ramp for Einsum
- 60: posr F32 [1,1,30,1] + posc F32 [1,1,1,30] — position ramps for Equal
- 120: row_base21 U8 [1,1,30,1] (30) + col_base1 U8 [1,1,1,30] (30) + col_base2 U8 [1,1,1,30] (30) + col_base8 U8 [1,1,1,30] (30) — IRREDUCIBLE (see scalar trap below)
- 30: zero_col U8 [1,1,1,30] — needed for 6 zero-channel Concat inputs
- 7: U8 scalar initializers (z, six, twenty4, thirty2, fifty6, sixty4, seventy)

**Where the staged build lives:** `staged_submits/task047_bitset_and_cost1003/task047.onnx`
**Builder:** `The_400_Tasks/Task_by_Task/task047/prototypes/build_task047_bitset_and.py`

## Insight 2: Scalar base vector trap — boundary masking is load-bearing

**What was discovered:**
Replacing boundary-enforcing base vectors (row_base21[1,1,30,1], col_base1/2/8[1,1,1,30]) with U8 scalars silently breaks boundary masking.

**Specifically:** row_base21 = [21,21,...,21(×9), 0,0,...,0(×21)]. The zeros at positions 9-29 ensure Where chains only set bits for the 9 active grid positions. When replaced with scalar `21`, Where(cond, val, scalar) broadcasts to [1,1,30,1] shape with 21 everywhere — including the 21 padding positions — producing spurious nonzero bits in grid positions that should be background.

**Verified cost was 887** (memory 736, params 151) but officials = 0/3 (completely broken).
**Prototype:** `The_400_Tasks/Task_by_Task/task047/prototypes/build_task047_scalar_bases.py`

This is a general trap for any Where-chain architecture: if a base vector encodes both a value AND a boundary condition (zeros outside the active area), it cannot be replaced by a scalar or uniform-value init.

## Insight 3: BitwiseAnd intermediate size miscalculation trap

**What was discovered:**
When estimating the cost of `BitwiseAnd([1,1,30,1], [1,10,1,9]) -> [1,10,30,9]`, the intermediate is 1×10×30×9 = 2700 B, NOT 270 B. The mistaken estimate (270 B) came from a factor-of-10 error (forgot the channel dimension).

**General rule:** For BitwiseAnd broadcasting, the output shape is the broadcast of ALL dimensions together. `[1,1,30,1]` broadcast with `[1,10,1,9]` gives `[1,10,30,9]` not `[1,1,30,9]`. Always write out the full broadcast shape explicitly before estimating memory.

**Prototype built to test:** `The_400_Tasks/Task_by_Task/task047/prototypes/build_task047_col9.py` (cost 3,186 vs expected ~756).

## Insight 4: Add-broadcast impossibility for asymmetric crossing conditions

**What was discovered:**
For a rule where exactly two specific cells (r8,c7) and (r7,c8) must get a special color, and no other cell in those rows/cols should, Add-broadcast free-output cannot work.

**Proof:** For a separable function output[r][c] = f(r) + g(c) to have a special channel active at (r8,c7) and (r7,c8) but NOT at (r8,c8) or (r7,c7):
- At (r8,c7): f(r8) + g(c7) = target (nonzero)
- At (r7,c8): f(r7) + g(c8) = target (nonzero)  
- At (r8,c8): f(r8) + g(c8) = 0 (must be zero)
- At (r7,c7): f(r7) + g(c7) = 0 (must be zero)

From lines 1 and 3: g(c7) > g(c8) (since f(r8)+g(c7)>0 and f(r8)+g(c8)=0 requires g(c7)>g(c8)).
From lines 2 and 4: g(c8) > g(c7) (since f(r7)+g(c8)>0 and f(r7)+g(c7)=0 requires g(c8)>g(c7)).
Contradiction: g(c7)>g(c8) and g(c8)>g(c7) cannot both hold.

This applies to any non-negative separable combination (Add, multiply, etc.) — the crossing condition is fundamentally not separable for the "exactly two crossing cells" pattern.

BitwiseAnd works because it is NOT separable in the same way: bit patterns can be independently nonzero at (r8,c7) and (r7,c8) via different bits, while being zero at (r8,c8) and (r7,c7) because the bit overlap only occurs at the crossing positions.

## Sorter note: traps.md — add as new section on scalar-base boundary masking trap (Insight 2)
## Sorter note: traps.md — add to GRADER-GAP or intermediate-size section for BitwiseAnd broadcast shape estimation (Insight 3)
## Sorter note: techniques.md — extend §67 BitwiseAnd or add new section on crossing-cell impossibility for Add-broadcast (Insight 4)
## Sorter note: cost_floor_reference.md — add task047 floor measurement: 1,003 = memory 736 + params 267 (Insight 1)
