# Concat-vs-Pad param savings for zero-prepend shift (task199, 2026-06-12)

## What I discovered

When you need to shift a 1D vector DOWN by 1 (prepend a single False/zero element and drop the last element), the standard approach uses `Slice(0, N-1, axis) + Pad([pad_row1])`. But `pad_row1` as an 8-element INT64 initializer (for a 4D tensor) costs 8 params. 

Replacing it with `Concat([false_row_init, Slice(0, N-1, axis)], axis=height_axis)` saves params:
- `pad_row1 = [0,0,1,0, 0,0,0,0]` INT64 [8] = **8 params**
- `false_row_init = [[[[False]]]]` BOOL [1,1,1,1] = **1 param** (single element)

Net: **-7 params**, 0 memory change. For cost = memory + params, this saves 7 bytes directly.

## Where I encountered it

task199 ("ParityFill"), v3→v4.

- Builder: `The_400_Tasks/Task_by_Task/task199/prototypes/build_task199_v4_concat_shift.py`
- Staged: `staged_submits/task199_parityfill_cost3172/`
- v3 cost 3179 → v4 cost 3172 (all gates green: 266/266 visible, 200k/200k truth, 200266/200266 ab byte-equal, profile==static==3172)

The shift was needed to move a row indicator vector down by 1 row (dot at row r → indicate row r+1 as the "next" row). The corpus guarantees the dot row is always ≤ H-2, so the last element (row N-1) is always False, making the Slice+Concat semantically equivalent to Slice+Pad.

## Why it matters

The `Pad` initializer for a 4D tensor must be [2*ndim] = 8 INT64 elements. Most values are 0 (pad nothing). The cost of this init is 8 params regardless of how many zeros it contains. 

The Concat approach uses the `axis` as an attribute (free) and a single BOOL [1,1,1,1] init with the zero/false value (1 param). The `false_row_init` has shape [1,1,1,1] = 1 element. Concat([1,1,1,1], [1,1,N-1,1], axis=2) = [1,1,N,1] ✓

Key constraint: this only works when:
1. The prepended value is FALSE/ZERO (a scalar constant, not data-dependent)
2. The dropped element (end) is structurally guaranteed to be zero/False anyway (corpus invariant)
3. The vector is 4D (cost of Pad init is 8; if it were lower-rank the savings would be smaller)

## Generalization

Any shift-with-zero-prepend of a 4D tensor (common in row/col shift operations):
- Old: Slice(data, 0, N-1, axis) + Pad(..., pads=[0,0,X,0,0,0,0,0]) where X=1
- New: Slice(data, 0, N-1, axis) + Concat([zero_init, slice_result], axis=X)

Savings scale with ndim: a 4D tensor's Pad init costs 8 params; a [1,1,1,1] BOOL init costs 1 param. Net: -7 per replaced shift Pad.

For cases where both axis-start and axis-end prepend/append are needed in the same Pad node, this trick can't easily replace it (the Concat would only handle one side). Best for pure-prepend or pure-append shifts.

## What's missing from the library

Searching techniques.md: I didn't find this specific insight documented. The general "Pad init is 2*ndim params" principle is implicit in cost_floor_reference.md but the specific Concat replacement trick isn't cataloged.

## Sorter note: techniques.md — add new section "Concat-vs-Pad for zero-prepend shifts" (tag: param_shrink, Pad, Concat, shift) after the current §41 negpad entry or near the param-shrink catalog entries
