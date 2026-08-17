# Deep-read: `neurogolf-acking-multiple-tasks-part-3.ipynb`

> Beta-track deep-read, 2026-05-03 20-hr session. Source: `Research/primary_sources/neurogolf-acking-multiple-tasks-part-3.ipynb` (3 cells, 12 KB).

## Summary

Author: artemnazemtsev. Russian-language notebook. Title literally: "ACKing multiple tasks part 3" — "хакер ONNX для NeuroGolf" ("ONNX hacker for NeuroGolf").

This is the **canonical exploit notebook** that uses the three "positive-cost wrappers" tricks that we explicitly rejected on ethical grounds (see `Research/A2_cost_metric_exploits.md` and `03_PROJECT.md` "Ethical-play stance" architecture decision).

## The three exploits (cell 2)

The author's `hack_submission.py` applies three transformations to existing ONNX submissions:

### Hack 1 — Dynamic-shape decoration

For every output and value_info tensor, replace `dim_value=N` with `dim_param="dN"`. This breaks onnx-tool's memory profiling: dynamic-shape tensors don't get accounted for, so the validator's `calculate_memory()` returns 0 for the wrapped intermediate.

### Hack 2 — Initializer → Constant node conversion

Move every initializer into a `Constant` node inside the graph. The validator's `params` accumulator iterates over `graph.initializer` only, so constants-via-Constant-nodes aren't counted as `params`. **This is the most direct cost-side cheat** — same answer at runtime, lower validator-counted cost.

### Hack 3 — ConstantOfShape output wrapper

Wrap the actual output through:
```
internal = (real_output)
shape = Shape(internal)
zeros = ConstantOfShape(shape, value=0.0)
output = Add(internal, zeros)
```

This makes the OUTPUT appear to be the result of an `Add(_, ConstantOfShape)` chain. Same rationale as Hack 1 — confuses the cost-attribution.

## Effect

Per cell 1, the notebook applies these to artemnazemtsev's `norm-blending-v2/submission.zip`. The author's expected outcome (per the comments in cell 2): cost_before > cost_after on most tasks; some tasks fail the hack (where adding the wrappers breaks `verify_subset` or `check_network`) and revert to the original.

## SKIP_TASKS list (tasks where the hacks BREAK correctness)

```python
SKIP_TASKS = {47, 53, 99, 127, 258, 294, 331, 385}
# + 1..399 (initially, then explicitly removed via specific lists)
# + [100, 103, 104, 129, 167, 171, 175, 176, 186]
# + [47, 53, 99, 127, 258, 294, 331, 385]
# + [200, 248, 266, 274, 291, 292, 305, 325, 334]
# + [357, 373, 375, 376, 399]
```

Net SKIP_TASKS (after the for-loop dedup): 39 tasks where the author DOES NOT apply the hacks because they would break those specific models.

### Cross-reference vs our 50 SOLVED set

Tasks from this SKIP list that are in our 50 SOLVED:
- **task053** (our `fixed-shift_v_1`) — already noted in `03_PROJECT.md` as competitor-flagged.
- **task129** (our `const-dominant-color`) — already noted.
- **task266** (our `marker-corners`) — already noted.
- **task385** (our `shape-changing-pixel-perm`) — already noted.
- **task047** — NOT in our SOLVED; new info.
- **task099** — NOT in our SOLVED.
- **task127** — NOT in our SOLVED.
- **task258** (our `marker-corners`-related) — already noted via PR #21 D2 work.
- **task294** (our `grid-pattern-color-to-grid`) — NEW competitor flag (not previously noted).
- **task331** (our `grid-pattern-color-to-grid`) — NEW competitor flag.
- **task373** (our `color-replace`) — NEW competitor flag.

Three NEW competitor-flagged items added to our SOLVED watchlist:
- task294, task331, task373 — **add these to the silent-fail watchlist** in `03_PROJECT.md` Open Bugs.

Combined competitor-flagged-from-our-SOLVED total = 12 tasks: {16, 50, 53, 129, 266, 276, 294, 309, 331, 337, 373, 385}.

## Strategic value to NG

1. **No novel patterns** — pure exploit application, not a pattern source.
2. **Validates the exploit family**: confirms our characterization in `Research/A2_cost_metric_exploits.md` is accurate. The Hack 1/2/3 are exactly as we documented and rejected.
3. **Updates the silent-fail watchlist**: 3 new tasks (294, 331, 373) to flag.
4. **Confirms ethical-play ceiling estimate**: the author runs the hacks on existing high-LB submissions. If a clean-play artifact got LB ~3275 (our honeydew), the hacked version reportedly hits LB 5000+ (per references in other notebooks). Our ceiling without exploits remains in the 4000-5000 range max.

## Action items

1. **Update `03_PROJECT.md` Open Bugs**: add task294, task331, task373 to the "Four of our 50 SOLVED appear in competitors' skip lists" entry. Total now: 12 of 50 SOLVED.
2. **No backlog additions** (no novel patterns).
3. **Consider regression test** for our SOLVED tasks on the watchlist before any future submission. Specifically, run `silent_fail_check --min-arcgen-N 100` on tasks {294, 331, 373} when next submitting.

## Closed sub-questions

- **"Is acking-multiple-tasks-part-3 a pattern source?"** — No, it's an exploit applied to existing submissions.
- **"Does it have novel insights beyond what `Research/A2_cost_metric_exploits.md` documents?"** — Slight extension: explicit code for all three exploits chained, plus the SKIP_TASKS list.
- **"Should we adopt the exploit?"** — No (per existing ethical-play stance).

---

*Author: Beta-track 2026-05-03 deep-analysis session. Cross-reference: `Logs/sessions/2026-05-03_1830_beta-20hr-session.md`.*
