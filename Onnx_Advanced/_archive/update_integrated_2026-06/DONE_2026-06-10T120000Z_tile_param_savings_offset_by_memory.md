# Tile-for-param-reduction: memory overhead offsets (and exceeds) param savings for basis tensors

## What I discovered

In task007, the direct-trig Einsum has `basis[2,30]` = 60 element params (F32). The basis has period-3
structure (first 7 values periodic, positions 7-29 are all zeros). The intuition is: store `basis_core[2,3]`
= 6 params, Tile to [2,30] at runtime, save 54 params.

**Result: the probe FAILS both correctness AND cost.**

**Correctness failure**: The zeros at positions 7-29 in the basis are NOT redundant padding. They FORCE
the output to be zero outside the active 7x7 window. A tiled basis (nonzero at positions 7-29) causes
output cells at r,c >= 7 to receive nonzero scores from input cells at h,w < 7. Visible gate: 0/266 FAIL.

**Cost failure**: The Tile output `[2,30]F32` is an intermediate = 240 bytes memory. Params savings
from 60→6 = 54. Net change: `+240 (memory) - 54 (params) = +186` cost. Total: 266 vs 78. Much worse.

**The crossover law**: For a basis tensor of shape `[T, N]` F32:
- Params if stored: T×N elements  
- Memory if Tile'd from `[T, P]` (P = period): T×N × 4 bytes
- Break-even: T×N×4 < T×(N-P) → 4N < N-P → impossible for P < N (memory is always proportionately more
  expensive per saved element, since memory is BYTES and params are ELEMENTS)

This means: **for F32 tensors, Tile-for-param-reduction is NEVER worthwhile** because the memory
overhead in bytes (4× element count) always exceeds the params saving in elements.

However, for BOOL intermediates or very short periods: `[T, N]` BOOL costs T×N bytes as intermediate.
Break-even: T×N < T×(N-P) → impossible. Still never worthwhile.

**The only case where Tile saves cost**: when the basis IS the graph output (free) — i.e., Tile IS the
final node. But that only applies to trivially-structured tasks where the output IS a tiling.

## Where I encountered it

- task007 v13 `tile_probe`: `The_400_Tasks/Task_by_Task/task007/prototypes/build_task007_v13_tile_probe.py`
- Tested: score_v2 cost=266, `ng_gate visible` 0/266 FAIL
- Task007.md Don't-do log updated 2026-06-10

## Why it matters

This kills a common intuitive "factoring" reflex. When a basis tensor has period-P structure, the natural
instinct is to store the period-P core and Tile at runtime. But the scoring formula is `memory(bytes) +
params(elements)`, not `bytes + bytes`. Tile intermediates pay in BYTES; initializers save in ELEMENTS.
Since 1 byte = 1 memory cost but 1 element = 1 param cost regardless of dtype, a F32 element saved =
4 bytes of memory penalty. The trade is always losing for F32.

The correct exploitation of periodic structure in a basis is: **store the full basis including the zeros** (as
an initializer), OR use the period structure to reduce the COMPUTATION (e.g., fewer Einsum contractions),
NOT the storage.

## What's missing from the library

§130 mentions that 78 is the analytic minimum but doesn't explain WHY Tile factorization fails (both the
correctness reason and the cost reason). The cost formula asymmetry (memory in bytes vs params in elements)
is documented in `core.md §params vs bytes` but not connected to the Tile-for-param trade-off. The new
pattern worth adding:

**"Tile-for-param reduction is always losing for non-graph-output intermediates"** (because memory is
bytes but params are elements, so F32 intermediate memory costs 4× the saved params).

## Sorter note: techniques.md §130 — extend with explanation of WHY Tile factorization fails (both correctness and cost reasons for this specific pattern); also add a new callout to cost_floor_reference.md about the memory-bytes vs params-elements asymmetry for Tile intermediates.
