# Source: ARC-AGI task families & priors

## ARC-AGI-1 dataset facts

- Repository: `https://github.com/fchollet/ARC-AGI` (Apache-2.0).
- 400 train + 400 evaluation tasks. Grids: integer matrices, values 0–9, max 30×30.
- Original paper: Chollet, *On the Measure of Intelligence*, arXiv:1911.01547, especially §III defining the Core Knowledge priors.

## Core priors (Chollet, ARC-AGI-2 paper §1.1, arXiv:2505.11831)

1. **Object persistence** — entities maintain identity across changes.
2. **Goal-directedness** — outputs respond to discrete causal rules.
3. **Number sense** — counting, sorting, comparing small integers.
4. **Basic geometry & topology** — symmetry, connectivity, neighbourhoods.

## Vision-ARC AND/OR/XOR cluster (J.01 relevant)

- Paper: "ARC Is a Vision Problem!", arXiv:2511.14761 (Nov 2025).
- §3 identifies a task-embedding cluster characterized by Boolean overlay logic between two halves of a grid (or grid-and-template). Sub-types include:
  - "Symmetric difference" (XOR of input halves).
  - "Common cells" (AND of two indicator masks).
  - "All-cells covered" (OR).
- This cluster is small (~5% of public tasks by the paper's count) but high-value because Conv solutions tend to be wasteful — bit-overlay logic is the "natural" representation.

## Hodel DSL primitives (J.13 / J.14 relevant)

- Repository: `https://github.com/michaelhodel/arc-dsl`. ~160 primitives covering ARC's effective operation space.
- Adjacency-rule families: `objects()`, `colorfilter()`, `fill()`, `gravitate()`, `connect()`, `neighbours()`. These are the candidates where Morton/Hilbert linearisation can pay off.
- Counting families: `numcolors()`, `count()`, `mostcommon()` — small dynamic range, no need for RNS.

## Procedural task taxonomy

- Wang et al., "A Neural Affinity Framework for Abstract Reasoning", arXiv:2512.07109 (Dec 2025). 97.5% rule-based classification accuracy on 400 ARC train tasks across taxonomy categories: geometric (rotation/reflection), color manipulation, fill/extend, count/compare, pattern detection.
- For NeuroGolf cluster targeting:
  - Geometric (~25%): Identity / Transpose / D4-symmetry orbits → cost 0 or near-0.
  - Color manipulation (~20%): Gather-based color permutation → ~5,000–40,000 cost.
  - Fill/extend (~15%): Conv-based or Morton-row-fill candidates.
  - Count/compare (~10%): single Mod or Conv 1×1 with reduce.
  - Pattern detection (~30%): mostly Conv 3×3 territory.

## Specific task IDs flagged in COVERED list

- 179, 241: solved by Transpose pattern (cost 0).
- 016, 337: solved by color-perm Gather pattern.
- 157, 177, 002, 337, 276, 309: flagged in `https://github.com/openai/codex/issues/19594` as connected to the leaderboard scoring discrepancy / evaluator bug (April 2026); now patched.

## ARC-AGI eval pipeline reference

- `https://arxiv.org/pdf/2505.11831` (ARC-AGI-2 paper) §2.1 describes the evaluation protocol that NeuroGolf inherits: exact-match grid equality after the model produces an output of the right size. NeuroGolf adds the `(>0)` thresholding wrap and the per-channel decoded representation.

## ARC-GEN-100K

- Bundled per-task with 250–1000 augmented variants. The "arc-gen" subset of each `taskNNN.json` is verified across these augmentations during scoring. **Implication for J.14**: a Morton-permutation solution that works on the canonical task instance must also work on rotated/reflected variants, which means the chosen permutation must commute with the augmentations applied during ARC-GEN. In practice this rules out fixed-orientation Morton paths for tasks whose augmentations include rotation; Hilbert curves with chosen rotations or per-orientation Morton tables may be needed.
