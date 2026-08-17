# Bool-chain AND-reorder saves 10B memory (task234, worker-234)

## What I discovered

When a CS classifier produces `is_filled = AND(eq_area, has)` and then derives both `is_anchor = AND(is_filled, notch0)` AND `is_mover = AND(has, NOT(is_filled), notch0)`, there is a redundant chain. Reordering to factor out `has_notch = AND(has, notch0)` FIRST saves one BOOL [1,10] intermediate:

**Old chain** (5 intermediates after eq_area + has):
- `is_filled = AND(eq_area, has)` [10B]
- `is_anchor = AND(is_filled, notch0)` [10B]
- `nfill = NOT(is_filled)` [10B]
- `mv0 = AND(has, nfill)` [10B]
- `is_mover = AND(mv0, notch0)` [10B]
Total: 50B

**New chain** (4 intermediates):
- `has_notch = AND(has, notch0)` [10B] — shared
- `is_anchor = AND(eq_area, has_notch)` [10B]
- `not_eq = NOT(eq_area)` [10B] — direct NOT of eq_area
- `is_mover = AND(not_eq, has_notch)` [10B]
Total: 40B

**Saving: 10B**. Logic equivalent:
- `is_anchor = AND(AND(eq_area, has), notch0) = AND(eq_area, AND(has, notch0)) = AND(eq_area, has_notch)` ✓
- `is_mover = AND(has, NOT(AND(eq_area,has)), notch0) = AND(NOT(eq_area), AND(has,notch0)) = AND(not_eq, has_notch)` ✓

## Where I encountered it

- task234, `staged_submits/task234_cs_reorder_cost2447/`
- Builder: `The_400_Tasks/Task_by_Task/task234/prototypes/build_task234_worker234_cs_reorder.py`
- Reduced cost 2457 → 2447 (memory 2394→2384, params 63 unchanged).

## Why it matters

When the classifier needs BOTH `is_anchor` and `is_mover`, factoring out the shared `has AND notch0` term reduces the intermediate count by 1 BOOL tensor. Applies to any graph where a "positive class" and "negative class" both need the same `has AND mask` guard.

## What's missing from the library

This micro-optimization is a structural rewrite of the BOOL op chain pattern. It's not specifically documented in techniques.md. The general pattern: when building `X = AND(A, AND(B, C))` and `Y = AND(NOT(A), AND(B, C))`, compute `BC = AND(B, C)` first to share it.
