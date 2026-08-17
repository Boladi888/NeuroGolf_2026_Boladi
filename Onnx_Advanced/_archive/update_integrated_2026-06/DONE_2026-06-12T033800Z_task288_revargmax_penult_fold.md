# Technique: Reversed ArgMax for "last-active-row" detection (task288)

**Date:** 2026-06-12T03:38Z  
**Worker:** worker-288  
**Cost delta:** v4 1905 → v5 1786 (-119)

## Pattern

When you need to find the INDEX of the last 1 in a binary [N] F32 vector (e.g., the last background row in col 0 of ch0), instead of:

```
Slice(input, shift by 1 down) → Pad(bottom with 0) → Sub(1, ...) → Mul(row_bg, no_below) → Reshape→[N] → ArgMax
```

Use:

```
Reshape(col_vec, [N])       →  [N] F32
Slice(..., starts=[N-1], ends=[-N-2], axes=[0], steps=[-1])  →  reversed [N] F32 (static value_info needed)
ArgMax(reversed)            →  [1] INT64  (position of first max in reversed = last max in original)
Sub(N-1, argmax)            →  [1] INT64  (original index)
```

## Memory savings

Old 5-node chain (for N=9): row_below_slice(32) + row_below_pad(36) + no_below(36) + penult_row(36) + penult_vec(36) = 176B removed.
New 3-node chain: row_bg_flat(36) + reversed_row(36) + rev_argmax(8) = 80B added.
Net: -96B memory.

Plus params removed: row_pad_bottom[8]=8, row_shift_starts[4]=4, row_shift_ends[4]=4 = 16 params.
Params added: eight_i[1], rev_end_i[1], rev_step_i[1] = 3 params.
Net: -13 params.

## Caveats

- The reversed Slice has provably constant output shape [N]; add static value_info annotation (trap §8b).
- `axes=[0]` can reuse `zero_i` if it already exists as [1]=0.
- `ends=[-N-2]` (e.g., -10 for N=9) must be smaller than -(N-1) to include all elements.
- ArgMax on reversed binary vector: if all zeros (edge case not in this task), ArgMax returns 0 → original index N-1 (wrong). Validate against task guarantees.

## Dedup reminder

`shape9=[9]` === `nine_i=[9]` and `shape10=[10]` === `ten_i=[10]` are identical initializers. Using nine_i/ten_i for both Reshape and Concat saves 2 params with no logic change.

## Candidate §-number

§148 (reversed ArgMax for last-active-element detection)
