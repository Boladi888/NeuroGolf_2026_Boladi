# Growth-op charge law: Pad/Concat/Tile inputs are always charged — free-output only works at the LAST op

**Task:** task057 (SpriteHtileFloorBreakAttempt sessions 2026-06-22 and 2026-06-24)  
**Worker:** w02 (Hermes, 2026-06-24 session confirming prior opus 2026-06-22 closure)

## What was discovered

The scorer skips EXACTLY two tensors: the ones named literally `'input'` and `'output'` in the graph.
Every other tensor (`graph.value_info`) is charged at `max(static, runtime)` bytes (core.md §53).

**The growth-op charge law (task057 proof):**
Any op that GROWS a smaller intermediate up to the final output size — `Pad(X)`, `Concat(A,B,…)`,
`Tile(X)` — makes that op's INPUT(s) a named `value_info` tensor, which IS charged. There is NO
"free-output growth chain": only the single tensor immediately named `'output'` is skipped, not its
predecessor chain.

Concrete measurements (task057, output = [1,10,30,30] F32):

| Growth strategy | Input tensor charged | Net vs baseline |
|---|---|---|
| Einsum→[1,10,3,6] (27 params), then Pad to [1,10,30,30] | onehot6 BOOL 180 B | +180 mem − 153 params = **+27** |
| Two-slab Concat (each [1,10,30,3]) → [1,10,30,6] | each slab 3,600 B | **+7,200+** |
| Tile on 3-wide block → 30-wide | Tile input charged | no win |
| Aditya's 4-operand Einsum writing [1,10,30,30] directly | NONE — it IS `output` | baseline 588 |

**Implication:** the only way to get "zero growth-op charge" is a single Einsum/op whose SOLE output IS
the named graph output tensor. Any two-step "build smaller → grow to full" strategy charges the smaller
intermediate, making it always ≥ the growth-op's input size. Trading params for a growth-op ONLY wins
if `param_save > intermediate_bytes`, which almost never holds for large outputs like [1,10,30,30].

## Where encountered

- task057: the prompt's "single remaining lever" — can Concat/Tile/Pad collapse `colmap[3,30]` = 90 params?
- Priced in `The_400_Tasks/Task_by_Task/task057.md` §HTILE FREE-OUTPUT LEVER (2026-06-22)
- No build was warranted; closed by arithmetic before any code

## Why it matters

This closes a class of speculative "shrink the params by using growth ops" ideas that appear in task
prompts. Any prompt offering "smaller maps + Pad/Concat/Tile to output size" as a lever can be quickly
rejected if `intermediate_bytes ≥ param_save`.

**Quick check formula:** let `P_save` = params eliminated by using smaller maps, `I_bytes` = bytes of
the intermediate created by the growth-op input. If `I_bytes ≥ P_save` (in bytes), the route is a loss.
For BOOL intermediates: `I_bytes = numel(intermediate)`. For F32: `I_bytes = 4 * numel(intermediate)`.

## What's missing from the library

The current `scoring_internals.md` notes that `'input'`/`'output'` are skipped (line ~58, ~177) and
`techniques.md` has free-output family (#67, #78, #91) but no explicit statement of the growth-op
charge law as a **negative** — i.e., that free-output ONLY exempts the final tensor, not its
predecessors. Workers keep rediscovering this independently (task057 had 4 sessions hit it).

## Sorter note: scoring_internals.md — add as a "growth-op charge law" callout in the free-output exemption section (near line 58 / the `'input'`/`'output'` skipped row): "ONLY the tensor literally named `'output'` is free — any Pad/Concat/Tile predecessor is a charged value_info. Trading params for a growth-op wins only if `param_save > intermediate_bytes`."

## Sorter note: techniques.md §67 (free-output Pad) or §91 (Einsum-final vs Pad-last crossover) — add a "GROWTH-OP PREDECESSOR TRAP" callout: two-step build-smaller→grow strategies always charge the smaller intermediate; this only wins when `param_save > intermediate_bytes` (rarely true for [1,10,30,30]-scale outputs). Measured task057 arithmetic as the calibration example.
