# Task 374 - Creative ONNX Formulation Search Below 1365

You are optimizing one NeuroGolf ONNX task from a self-contained webprompt pack.
You do not have the original repo unless the user separately gives it to you.
Read `02_task_context/ONNX_ADVANCED_MINI_LIBRARY.md`; it replaces the local
Onnx_Advanced library for this task.

Metric:

```text
cost = memory_bytes + params
points = max(1.0, 25.0 - ln(cost))
```

The best useful result is an exact candidate below cost 1365. If you cannot
create an ONNX file directly, return builder code or an implementation-ready
graph construction plan that a local verifier can run.

## Core Rule

The input is a 10x10 grid embedded in NeuroGolf's 30x30/10-channel format.
Color-5 cells form non-overlapping horizontal and vertical line segments.

Compute each row's maximum contiguous horizontal 5-run and each column's maximum
contiguous vertical 5-run. Rank distinct positive lengths globally:

- longest -> output color 1;
- second-longest -> output color 4;
- all remaining positive lengths -> output color 2;
- background -> output color 0.

For each color-5 cell, use the column rank when the cell has a vertical
color-5 neighbor. Otherwise use the row rank.

## Creative Mandate

Do not start by polishing the existing qscale-share graph. First propose at
least three independent formulation families, estimate their charged tensors,
and decide which one has a credible path below 1365.

Good families to consider:

- rule-first direct renderer that avoids materialized 30x30 labels;
- different final-output convention or decoder primitive;
- compact row/column descriptor formulation;
- separable or final-op-only expansion;
- alternate way to encode max, second, and remaining segment classes;
- qscale-share family polish only after the broader pass.

The included qscale-share candidate is a reference rung, not the assignment. The
included 1150-1250 wall only proves that one-channel linear ConvInteger packing
cannot separate four classes. It does not prove all sub-1365 paths impossible.

## Guardrails

- DO NOT submit. The captain ships.
- Do not stage ties. A ready candidate must be strictly below 1365 unless a
  newer source gives a lower effective best.
- Do not use Loop, Scan, If, NonZero, Unique, custom domains, symbolic output
  shapes, sparse initializer tricks, metadata hiding, or packaging-only changes.
- Do not return "already optimal" unless you can state a forced-component proof
  that covers the formulation space you actually searched.
- Treat visible examples as smoke only. A rule-rebuild needs generated or
  adversarial validation when a local verifier is available.

## Required Work Product

Return one of these:

1. `ready` - exact ONNX candidate or builder code, expected cost below 1365,
   validation evidence, and why hidden behavior should hold.
2. `probe` - promising but unverified construction, exact tensors/ops, expected
   cost, and the first local validation command to run.
3. `discard` - no candidate, but with the families tested, cost estimates, and
   the next untried creative route.

Use the report format in `04_validation/VALIDATION_AND_RETURN_CONTRACT.md`.
