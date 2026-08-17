# INT64 ArgMax output: skip Cast→U8 when Equal works on INT64

**Source:** task192 v21, worker-192, 2026-06-12.

## What I discovered

When the ArgMax output (always INT64) is used only in an `Equal` comparison against a 1D index
vector (e.g., `idx8_colors = [1..9]`), you can skip the intermediate `Cast(INT64→U8)` and
use `Equal(INT64, INT64_index_vector)` directly.

Savings: eliminates 1B memory (the U8 scalar intermediate). The `idx8_colors` initializer
changes from U8 to INT64 but the **param count is unchanged** (params = element count, dtype-agnostic
per `neurogolf_utils.calculate_params`).

`Equal` on INT64 is valid at opset 13+. Broadcasting `[1] == [N]` → `[N] BOOL` works correctly.

## Where

- Builder: `The_400_Tasks/Task_by_Task/task192/prototypes/build_task192_v21_int64equal.py`
- Staged candidate: `staged_submits/task192_carrierdel_cost3030/`
- Δ cost: 3031 → 3030 (-1B memory, same 123 params)

## Why it matters

The v20 color-scale path was: `ArgMax → INT64[1](8B) → Cast→U8[1](1B) → Equal(U8, U8[9]) → ...`
The v21 path is: `ArgMax → INT64[1](8B) → Equal(INT64, INT64[9]) → ...`

This saves the 1B `main_idx_u8` intermediate with no other changes.

Gates on staged candidate:
- visible: 265/265
- truth full: 200000/200000
- AB full: 200265/200265 byte-equal, 0 divergences
- score/profile: 3030 static, 3030 real-input profile

## What's missing from the library

The techniques.md color-scale path sections (e.g. §42, §85) and the ArgMax pattern notes do
not mention this Cast-skip. It's a small but free gain on any graph with the pattern:
`ArgMax → Cast(INT64→U8) → Equal(U8, U8_index_vec)` where the index fits in INT64 and opset ≥ 13.

## Sorter note: add to scoring_internals.md or ort_reality.md — note that Cast(INT64→UINT8) can
be skipped when the INT64 value is used only in an Equal comparison with an INT64 index vector.
Also: params = element count (dtype-agnostic), so U8[9] vs INT64[9] = same 9 params.
