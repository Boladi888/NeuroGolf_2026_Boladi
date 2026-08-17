# per_family_starting_points.md — best-first techniques per task family

Quick-reference. When you've identified your task's family (from `The_400_Tasks/NeuroGolf_Task_Status.xlsx` or your task's `taskNNN.md`), use this to pick the first technique to try.

The four families:

| Family | N (of 401) | Median cost | Best-first technique |
|---|---:|---:|---|
| Pattern_Recognition | 170 | 27,584 | input→BOOL whole-graph (if downstream bool-tolerant); else crop-to-active-grid + F16 |
| Object_Based | 144 | 17,303 | bounding-box via ReduceMax(axes=[2,3]); projection-cache for wide intermediates |
| Color_and_Logical | 45 | 34,395 | bool_first family (And-fold, Or-cast, mul_mask_to_where) |
| Spatial_and_Geometric | 41 | 4,674 | shape-only (Slice + Gather + Pad + Concat) |

---

## Pattern_Recognition (170 tasks)

The biggest family. Includes "Filling Regions" (the dominant transformation), Tiling, Symmetry Completion.

### Best-first techniques

1. **Crop-to-active-grid + F16-output trick** when max active grid << 30. Stack: prepend Slice + rewrite range30/c29i/c30i + append Pad + drop final F32 Cast. Cost reductions of -50k to -200k common. (techniques.md §2 + §3.)

2. **input → BOOL whole-graph** when downstream is BOOL-tolerant. **BOOL is accepted by**: `Gather`, `Where` (condition only), `Pad` (opset 13+), `Concat`, `Reshape`/`Slice`/`Transpose`/`Squeeze`/`Unsqueeze`/`Identity`, `And`/`Or`/`Not`/`Xor`, `Equal(BOOL, BOOL)`, `ScatterND`. **BOOL is REJECTED by all Reduce ops** (Max/Min/Sum/Mean) — `Cast(BOOL, UINT8)` before `ReduceMax`/`ReduceMin` (1 byte at opset 13+) or `Cast(BOOL, F16/INT32)` before `ReduceSum`/`ReduceMean` (2/4 bytes). BOOL is also REJECTED by `Greater`/`Less` (order comparison) and `Conv`/`MatMul`/`Gemm`. The "whole-graph" framing applies only to shape/logical chains; cast before reductions. (techniques.md §1 rule 8.)

3. **bool-Concat keeping** if there's any Concat in the graph and all inputs are bool-eligible. (techniques.md §1 rule 7.) task299 example: 60% reduction on cost-2,678 near-floor task.

4. **Equal-fold variants** for Sub→Mul(self)→Less<1 or Sub→Abs→Less<0.5 chains. Use shared Cast across N parallel comparisons (techniques.md §5).

5. **Periodic-fill algorithm template** for tasks where the rule is "fill missing cells using inferred period" (algorithm_patterns.md §4). task110 exemplar.

### Algorithm patterns common in PR

- Periodic fill (task110 family) — period detection + iterative fill.
- Tile expansion — input is a small tile, output replicates it. Slice + Concat templates.
- Symmetry completion — half input, fill mirror. mirror-concat family (5 tasks at cost ~1,500).
- Pattern recognition with color compare — multiple colors, find the matching pattern via Equal-fold.

### Anti-patterns (PR-specific)

- Joint V+H ArgMax for periodic tasks. Always score per-direction.
- `Sub(1, ch0)` for foreground detection.

---

## Object_Based (144 tasks)

"Cropping", "Object Detection", "Color Mapping" transformations dominate.

### Best-first techniques

1. **Bounding-box extraction** via `ReduceMax(axes=[2])` + `ReduceMax(axes=[3])` (algorithm_patterns.md §2). For "find the object" or "extract the largest region" tasks.

2. **Projection-cache** (techniques.md §12) — when a wide `[1, C, H, W]` intermediate is only consumed by reductions/gathers. Compute reductions directly from input, skip the intermediate.

3. **ScatterND** for fill-region tasks where you compute target positions then fill them. task118 exemplar (-38,341 bytes via bool ScatterND).

4. **separable_mask_collapse** if there's a 2D mask init that varies along only one axis. Common for "edge mask" or "border mask" inits. task040 example (-21,960 bytes).

5. **nonzero_shortcut** for `Mul(input, gate) → ReduceSum` patterns (techniques.md §10). Saves ~28k bytes by dropping the mask construction.

### Algorithm patterns common in OB

- Bounding box → fill / recolor.
- Find-largest / find-smallest object via ReduceSum(axes=[2,3]) + ArgMax.
- Color of the object → Gather + ScatterND.
- "Find the unique color" — color histogram + ArgMax on count=1 entries.

### Anti-patterns

- Approximating CC analysis as bounding-rectangle without adversarial verification. task145 trap.
- Building a `[1, C, H, W]` wide intermediate before reducing. Use projection-cache.

---

## Color_and_Logical (45 tasks)

"Color Mapping" dominates. Smaller family, often clean predicate-logic tasks.

### Best-first techniques

1. **Color permutation** via single `Gather(axis=1)` (algorithm_patterns.md §1). For tasks where every cell color c → π(c). Cost ~10. Threshold: require ≥ 7 of 10 colors observed.

2. **bool_first family** end-to-end — Equal/Greater/Less producing bool, And/Or/Not chains, Where(bool_cond, ...) until output. (techniques.md §1 rules 1-7.)

3. **Equal-fold** for "is color X" tests — replace Sub→Mul(self)→Less<0.25 chain with single Equal. task233 saved -32,000 bytes.

4. **shared_cast_equal_fold** for N-way color comparisons (10 channels typical). One shared `Cast(value, INT32)` + N Equal nodes. task368 saved -7,200 bytes.

5. **or_cast_collapse** for mutex predicates `Greater(x, hi) + Less(x, lo)` with `hi >= lo`. task196 example.

### Algorithm patterns common in CL

- Color permutation (algorithm_patterns.md §1).
- "Color of the most common cell" — histogram + ArgMax.
- "Cells matching N criteria" — chained predicates via And.
- "Recolor based on neighborhood property" — Conv-based local check + color decision.

### Anti-patterns

- Pre-casting bool masks to F32 for arithmetic. Stay bool through Mul-of-masks (`bool_andfold`).
- Using `Cast(Less(Sqr(Sub(...)))` when integer-equivalent. Use Equal directly (Equal-fold).

---

## Spatial_and_Geometric (41 tasks)

Smallest family by count, smallest median cost. "Symmetry Completion", "Rotation", "Reflection", "Tiling".

### Best-first techniques

1. **Shape-only patterns** — Slice + Gather + Concat + Pad + Reshape + Transpose. Cost floor ~1,500. Beats LUT/Conv when applicable. (techniques.md §7.)

2. **Pad-attr → Pad-input (opset bump)** for tasks with opset-10 Pad attribute form. Combined with `Cast(bool→UINT8)` saves bytes (task319 example, -742).

3. **Tile → Reshape+Expand** when the task uses Tile with a small repeat factor. Expand doesn't materialize.

4. **Single-fixed-shape hardcoded transform** when all visible pairs have the same input/output shape. 4/400 tasks fit this. Cost = 0 (max 25 pts).

5. **slice_chain_to_multiaxis** when two consecutive Slices on different axes can fold into one.

### Algorithm patterns common in SG

- Rotation / reflection — fixed Gather indices on axis 2 or 3.
- Tiling — Slice + Concat with repetition.
- Padding manipulation — Pad with specific modes.
- Symmetry completion (algorithm_patterns.md §5).

### Anti-patterns

- Full-canvas Gather when a sub-grid Slice→Gather→Pad sandwich is smaller. Measure first (`tasks 87/140/380` lesson).
- Trying to merge Pad chains with different modes (e.g., reflect + constant). Cannot fold; documented dead-end on task240.

---

## Cross-family decision flow

When you start a new task:

1. **Look up the family** in `The_400_Tasks/NeuroGolf_Task_Status.xlsx` (Tasks sheet, `family` column) OR in this file's table.
2. **Open `task_techniques_index.md`** to see which technique class your specific task has been previously attempted in.
3. **Read the exemplars** in the corresponding `exemplars/<category>/` folder.
4. **Apply techniques.md** for the actual rewrite, using the family's best-first order from this file.

If your task is `Pattern_Recognition` + "Filling Regions" + cost > 100k:
- Start: crop-to-active-grid (techniques.md §2) — almost certainly applies.
- Stack: F16-output trick (techniques.md §3).
- Consider: input→BOOL whole-graph if no Conv (techniques.md §1 rule 8).
- Exemplar: task117 (the canonical crop+F16 stack).

If your task is `Object_Based` + cost > 50k:
- Start: bounding-box extraction (algorithm_patterns.md §2).
- Look at: task049 (least_color_solid_bbox -51,523 from-scratch).
- Consider: projection-cache (techniques.md §12).

If your task is `Color_and_Logical` + has many `Equal`/`Where` ops:
- Apply: bool_first family rewrites (techniques.md §1).
- Exemplar: task196 (or_cast_collapse + flip_where), task233 (Equal-fold).

If your task is `Spatial_and_Geometric` + cost < 5k:
- Likely already shape-only; check if smaller rewrite is possible.
- Consider: opset bump (techniques.md rule 6) for Pad-based tasks.
- Consider: dead-Relu / Slice-CSE checks (techniques.md §13, §16).
