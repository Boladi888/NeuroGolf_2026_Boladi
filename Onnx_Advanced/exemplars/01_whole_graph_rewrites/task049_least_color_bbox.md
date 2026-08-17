# task049 — from-scratch bounding-box rebuild (`least_color_solid_bbox`)

| Field | Value |
|---|---|
| Task | task049 |
| Family | Object_Based |
| Cost reduction | -51,523 (from-scratch) |
| Verify | 96% pass (some adversarial fail) |
| Pattern | Whole-graph rewrite — bounding-box extraction |
| Build script | `The_400_Tasks/Task_by_Task/task049/prototypes/` |

## Rule

Find the bounding box of the cells with the least-common non-bg color in the input. Output a solid rectangle filling that bbox.

## Algorithm

```python
# Step 1: per-channel histogram
hist = ReduceSum(input, axes=[2, 3], keepdims=False)   # [1, 10] — count per color
# Exclude ch0 (bg)
hist_no_bg = Slice(hist, starts=[1], ends=[10], axes=[1])   # [1, 9]

# Step 2: ArgMin to find least-common non-bg color
# (skip colors with count 0 — those don't appear)
nonzero_mask = Greater(hist_no_bg, 0)
masked_hist = Where(nonzero_mask, hist_no_bg, INT_MAX_const)
least_color_idx = ArgMin(masked_hist, axis=1)   # [1] scalar INT64 (0..8 → +1 for actual color)
least_color = Add(least_color_idx, 1)            # color value 1..9

# Step 3: extract the least-color channel
least_ch = Gather(input, least_color, axis=1)    # [1, 1, H, W]

# Step 4: bounding box of least_ch
row_present = ReduceMax(least_ch, axes=[3], keepdims=1)   # [1, 1, H, 1]
col_present = ReduceMax(least_ch, axes=[2], keepdims=1)   # [1, 1, 1, W]
bbox = Mul(row_present, col_present)                       # [1, 1, H, W]  ← outer product
# (use And of bools instead of Mul to stay bool-first — see bool-first rule 2)

# Step 5: place into output channel
output = Mul(bbox, OneHot(least_color, depth=10, axis=1))   # [1, 10, H, W]
```

## Why it works

The rule is decomposable into well-defined op-class steps. Each step has a minimal ONNX template. The cumulative cost (~50k) is dominated by the OneHot expansion at the end — without it, the model could be much smaller (just the bbox tensor, ~3,600 bytes F32 or 900 bytes BOOL).

## Why verify is 96% not 100%

The original task049 ONNX may have additional preconditions (tie-breaking when multiple colors have the same count, edge cases on single-cell objects) that this rebuild approximates. The 4% adversarial gap suggests hidden-bad risk — see `Algorithm_Library/rule_discovery_methodology.md` "Trap 3" (rectangular approximation of CC).

## Generalizes to

`Algorithm_Library/algorithm_patterns.md` §2 (bounding-box extraction). Apply to any Object_Based task whose rule is "find object → fill bbox → place at color X".

Variants:
- "Largest object" → ArgMax instead of ArgMin on hist.
- "Specific color" → skip the ArgMin, directly Gather(channel=K).
- "Outline" → bbox without fill (subtract eroded bbox from outer bbox).

## Reproducibility

Read `The_400_Tasks/Task_by_Task/task049/prototypes/build_task049_least_color_solid_bbox.py` for the full construction. The build script is annotated with the worker's design choices.

For applying to a new bbox-style task: copy the template, adjust:
1. The hist computation (which axes to reduce, which channels to slice).
2. The picker (ArgMin / ArgMax / fixed channel).
3. The output construction (solid fill / outline / placed shape).

Verify against LIVE on adversarial inputs BEFORE staging — bbox approximations are silent-fail-prone.
