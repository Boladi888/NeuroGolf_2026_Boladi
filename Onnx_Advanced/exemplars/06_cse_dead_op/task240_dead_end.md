# task240 — documented dead-end (Pad-mode merge, fp16 regress)

| Field | Value |
|---|---|
| Task | task240 |
| Verdict | DEAD END — no working optimization found |
| Cost | unchanged after 3 attempted rewrites |

## Why this file exists

A documented dead-end is as valuable as a documented win. task240 had 3 optimization attempts fail, all documented here so future workers don't redo them.

## Failed approach 1: Two-Pad chain merge

task240 has `Pad(mode='reflect') → Pad(mode='constant')` chain. Tempting: collapse into one Pad with combined pads.

**Why it fails**: the two modes have different boundary semantics. Reflect-pad mirrors content; constant-pad inserts a fixed value. The combined behavior cannot be replicated by any single Pad mode. Specifically, the cells that get reflected by the first Pad include cells that are themselves filled by the second Pad's constant — the order matters and the combined output is mode-dependent.

**Rule**: NEVER merge Pad chains with different modes.

## Failed approach 2: No bool-first opportunities

task240's graph has:
- No add-chains (no consecutive Add nodes).
- No all-ones Mul patterns (`pass_mul_by_all_ones_drop` no-op).
- No Sub-Mul-Less patterns (`pass_equal_fold_distance_threshold` no-op).
- No Concat of bool-eligible inputs (`pass_concat_bool_inputs` no-op).

All bool-first family detectors no-op. The graph structure simply doesn't have the shapes any of these rules match.

## Failed approach 3: fp16 regression

`pass_fp16_selective` applied: regresses by +89,140 bytes.

**Why**: task240's graph has many Cast nodes already. Adding blanket F16 conversion via onnxconverter_common inserts MORE Cast nodes for type compatibility, and the additional Cast overhead exceeds the F16 byte savings on the intermediates.

## What COULD work (untested)

Speculative — workers welcome to try:
- Manual placement-aware F16 cast (not blanket) at specific narrow intermediates only.
- input → BOOL whole-graph IF downstream is bool-tolerant (worth checking the graph structure).
- Algorithm-family swap: rebuild from scratch with a different op chain.

## Worth recording

Cost: task240 currently at its canonical value. Without a working optimization, it stays.

## Generalizes to (anti-pattern)

`techniques.md` "Anti-patterns" section. Documented dead-ends to NOT redo:
- DON'T merge multi-mode Pad chains.
- DON'T trust `pass_fp16_selective` blanket conversion.

## Cross-ref

- `Algorithm_Library/rule_discovery_methodology.md` "Trap 5" — numerical-precision rules can break under bool reformulation.
- `techniques.md` anti-patterns section.
