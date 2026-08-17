# exemplars/ — worked case studies for radical rewrites

Each exemplar is a case study of a technique that landed bytes on a real task. The format is consistent across files:

1. **Header** — task ID, family, cost before/after, bytes saved, LB impact, verify + adversarial result.
2. **Before** — the original op chain in `Op(in1, in2)` notation.
3. **After** — the rewritten op chain.
4. **Why it works** — the math / safety reasoning.
5. **Generalizes to** — pointer to the relevant `techniques.md` rule or `algorithm_patterns.md` section.
6. **Artifacts** — paths to staged candidate, build script, manifest.
7. **Reproducibility** — checklist for applying the same technique to another task.

## How to use these

Read EVERY exemplar in the categories relevant to your task family. Don't skim the headline. Some exemplars cover multiple techniques; the cross-references matter.

If you discover that an exemplar's technique applies to your task, you can copy the build script as a starting point. Adjust dimensions / indices to match your task's specifics. Re-verify with `tools/verify_against_live.py`.

## Categories

- `01_whole_graph_rewrites/` — algorithm-family swaps; from-scratch rebuilds. Highest leverage.
- `02_bool_first/` — the 8-rule bool-keeping family.
- `03_crop_and_f16/` — spatial crop-to-active-grid + F16 placement.
- `04_algebraic_identities/` — math-equivalent rewrites (Equal-fold, Add-chain, etc.).
- `05_static_to_dynamic/` — replace static lookup tables with dynamic compute at minimum rank.
- `06_cse_dead_op/` — common-subexpression elimination, dead-op pruning.
- `07_conv_reformulation/` — Conv → smaller-cost equivalent.
- `08_gather_lookup/` — lookup-table compression and Gather optimizations.
- `09_periodicity_detection/` — period detection + iterative fill (task110 lesson).

Some exemplars appear in multiple categories — file lives in the primary; cross-category READMEs cross-reference.

## Cross-references

When an exemplar appears in multiple categories, the primary file lives in one. Other categories' READMEs link to it. Examples:

- `task376_input_to_bool.md` — primary in `01_whole_graph_rewrites/`. Cross-ref from `02_bool_first/` (rule 8).
- `task119_diag_conv.md` — primary in `01_whole_graph_rewrites/`. Cross-ref from `07_conv_reformulation/`.
- `task071_ring_mask.md` — primary in `05_static_to_dynamic/`. Cross-ref from rank-reduction discussion in techniques.md §11.
