# task171: border-frame detection — bias is mandatory for padding-cell suppression

## What I discovered

For tasks where a Conv detects spatial borders (interior vs border vs padding cells), the BIAS is mathematically mandatory when the rule requires distinguishing interior cells from padding cells adjacent to the active area.

## The specific contradiction proven

For interior detection (ch0 output) using a Conv kernel with positive off-center weights:
- Interior cells (self=1, all 8 neighbors active): need positive output
- Padding cells (self=0, up to 3 active neighbors in window): need non-positive output

ANY kernel that distinguishes interior (8 active neighbors) from border (3-7 active neighbors) requires positive off-center weights (to count active neighbors). But these same positive off-center weights cause PADDING CELLS adjacent to the active area to also get positive output (since they have 1-3 active cells in their 3x3 window).

The bias B[0] = -2.33 suppresses this: even with 3 active cells in the window (worst case for padding), the total = 3 × max_weight + B[0] < 0.

## Generalization

For ANY border/interior spatial detection task where:
1. You need positive output for cells with ALL neighbors active (interior)
2. You need non-positive output for cells in the padding region adjacent to the active area

→ The bias is MANDATORY. Zero-bias kernels are provably impossible for this use case.

This is true because: interior detection requires counting neighbors (positive off-center weights), but padding cells adjacent to the active area will inevitably have some active cells in their neighborhood.

## The floor it proves

For task171:
- Single Conv [10,10,3,3] + bias [10] = 900 + 10 = 910 params is the floor
- Any multi-step approach costs more (the ch0_slice intermediate is ≥324B min, making total >910)
- Zero-bias approach (900 params) fails visibly (0/54 on train examples)

## What's missing from the library

The traps.md §FREE-OUTPUT SAFETY BATTERY mentions the boundary-leak issue, but doesn't specifically address the "zero-bias is impossible for active-region border detectors" case. This is a narrow but important corollary:

**COROLLARY**: For `Conv` kernels that detect cells by counting active neighbors, the bias is mandatory when the task has variable-size active regions (the padding cells adjacent to the active boundary will have active neighbors). No structural zero-bias kernel can correctly suppress them while also correctly identifying interior vs border cells.

## Where I encountered it

- Task171, worker w07 session 2026-06-24
- Build script: `The_400_Tasks/Task_by_Task/task171/prototypes/build_v1_nobias.py`
- Failed prototype: `The_400_Tasks/Task_by_Task/task171/prototypes/task171_v1_nobias.onnx`
- Documented in: `The_400_Tasks/Task_by_Task/task171.md` (Don't-do #1, Floor proof section)
