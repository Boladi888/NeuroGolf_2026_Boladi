# task316 worker2 scratch

## Rule
Input is always a 10x10 grid embedded at the top-left of the NeuroGolf 30x30 canvas. Nonzero cells are read by column from left to right. The current best assumes the generator has at most one nonzero cell per column, so each column can be reduced to a single color scalar. The first nine occupied columns are emitted as a 3x3 boustrophedon tail:
row 0 left-to-right, row 1 right-to-left, row 2 left-to-right. Empty tail cells are color 0.

Output is a 3x3 one-hot/BOOL block padded to `[1,10,30,30]`.

## Falsification Cases
- Two nonzero cells in the same column: would break the current column-scalar TopK formulation, so a replacement must match current best or the generator invariant.
- More than nine occupied columns: only the first nine left-to-right columns are emitted.
- Fewer than nine occupied columns: remaining 3x3 tail cells must decode as color 0, not no-color.

## Current Charged Bar
Team best is `Team_Best_Onnx/task316.onnx`, cost 426 = memory 388 + params 38.
Largest charged intermediates:
- `col_scalar30_f` F32 `[30]`: 120 B
- `color10_bool` BOOL `[1,10,3,3]`: 90 B
- `top_indices` INT64 `[9]`: 72 B
- `col_scalar30_i8` INT8 `[30]`: 30 B
- `sort_key_f_for_topk` F16 `[10]`: 20 B
- `top_values_f16` F16 `[9]`: 18 B

## Candidate Families
- Family A: index/sort-key shave. Keep the `Einsum -> TopK -> Gather` skeleton, but remove or narrow charged index/value tensors. Target charged structure: `top_indices` and `top_values_f16`.
- Family B: no-TopK fixed gather. Since occupied columns are sorted left-to-right by current TopK over a binary key with sorted ties, try replacing TopK with rank/where or a fixed permutation. Estimate likely worse because any per-column rank/mask plane costs at least `[10]` and multiple passes.

## Intended Deletion
First attempt is to delete the 18 B `top_values_f16` output from `TopK` if a single-output `TopK` is accepted by checker, ORT, and scorer. If ONNX requires both outputs, this family is dead as a staging route.
