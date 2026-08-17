# Research Card F4: Direct-detection-via-multi-QLinearConv patterns

## Deliverable

Produce a single **copyable Markdown research note**. Do not write files.
Target length: ~120 lines. Suggested filename: `direct_detection_multi_qlinearconv.md`.

Survey + design note. Include short ONNX/numpy snippets only for the canonical detector layouts.

## Goal

Survey and formalize a class of grid-transformation rules where the output is produced by running **multiple parallel QLinearConv detectors**, each detecting a different sub-pattern, and combining their outputs via paint-mask selection. The hallmark is that each detector is a structural QConv kernel (the "rule library" is the set of kernels), not a learned weight. Provide ONNX-static patterns under opset 18 with int8/uint8 quantization.

## Background

In some grid-transformation problems, the rule is best expressed as "if pattern A appears, paint with color X; if pattern B appears, paint with color Y; ..." where each pattern is a small (typically 3×3 or 5×5) template. Implementing this as a single MLP would require learning all templates jointly; implementing as multiple parallel QLinearConvs allows each detector to be hand-crafted with explicit weights and quantization parameters.

Distinct from:
- Single Conv + threshold (one template only)
- Mask-then-paint (no per-cell template matching)
- Learned CNN (weights are trained, not structural)

## Required research questions

### Q1. Inventory of structural-detection-via-Conv patterns
Survey published or canonical patterns:
- **Template library**: N parallel QLinearConvs, each detecting a different small template
- **Multi-class detection**: Conv with C output channels, one per class
- **Hierarchical detection**: coarse detector first, fine detectors gated by coarse
- **Per-color detection**: one detector per color in the palette

For each: structural setup, intended use, and computational cost.

### Q2. QLinearConv kernel design
For structural (non-learned) QLinearConv kernels:
- Choosing weights to detect a specific binary template (e.g., L-shape, T-shape, cross)
- Setting bias so positive detection has score ≥ K and negative ≤ 0
- Choosing zero-points and scales to keep score in valid int8 / uint8 range
- Multi-input-channel kernels (detection across multiple input planes)

Discuss the int-arithmetic edge cases (overflow, saturation).

### Q3. Combination strategies
Once N parallel detectors output their per-cell scores, how to combine them into a single paint output:
- **ArgMax across detectors** (paint with color of best-scoring detector)
- **Max with threshold** (paint only if max score exceeds T; else background)
- **Priority chain** (detector 1 wins if positive, else detector 2, ...)
- **Sum + lookup** (sum scores, lookup combined score in a table)

Compare ONNX-op cost and behavioral semantics.

### Q4. Tiling / receptive-field tradeoffs
For an N×N template on a 30×30 grid:
- Stride-1 detection (output is 30×30, full receptive field per cell)
- Stride-N detection (output is 30/N × 30/N, lower memory)
- Dilated detection (sparse kernel, larger effective receptive field)
- Multi-scale detection (multiple kernels at different scales)

Tabulate memory + compute for kernel sizes 3, 5, 7.

### Q5. Quantization tactics for multi-detector setups
When running N detectors in parallel under int8 / uint8:
- Sharing zero-points / scales across detectors (vs per-detector)
- Avoiding accumulated quantization error in argmax / threshold steps
- When to upcast back to float (and at what cost)
- Choosing scale × bias such that int outputs map cleanly to paint colors

### Q6. Detector pruning
Some detectors will never fire on the visible training pairs. Survey:
- Identifying always-zero detectors by running them on training inputs
- Pruning their Conv weights to reduce model size
- Folding pruned detectors into combinatoric mask reductions

### Q7. Sample task: NeuroGolf task367 v2 + task077 rawwhere
Real examples. task367 v2 is "structural direct-25 QLinearConv (no MPH)" and task077 rawwhere is "rawwhere with hull-8 detection". Sketch:
- For task367 v2: what 25 parallel detectors might look like, their combination strategy
- For task077: what hull-8 detection means as a Conv kernel, and how "rawwhere" produces direct paint output

## Required answer order

1. Inventory of multi-detector patterns (Q1)
2. QLinearConv kernel design for structural detection (Q2)
3. Combination strategies (Q3)
4. Tiling / receptive-field tradeoffs (Q4)
5. Quantization tactics (Q5)
6. Detector pruning (Q6)
7. Worked examples for task367 v2 + task077 rawwhere (Q7)
8. Decision tree: given rule shape, which detector setup
9. Verification checklist

## Required schemas

For Q1 (pattern inventory):

| Pattern | # detectors | Kernel shape | Combination | Memory cost | Use case |

For Q4 (tradeoff table):

| Kernel size | Stride | Output shape | Memory | FLOPs | Notes |

For the decision tree:

| Rule structure | # detectors | Kernel design | Combination strategy | Quantization tactic |

## Code snippet policy

- Include short ONNX-Python or numpy snippets only for ONE canonical detector layout.
- No full graph builders.
- Demonstrate one QLinearConv kernel design for a 3×3 template (e.g., L-shape).

## Source policy

- Cite ONNX operator spec (`QLinearConv`, `ArgMax`, `Cast`) by version.
- Cite ORT 1.24 supported QLinearConv configurations.
- Cite any published structural-CNN / template-library detection references.
- Mark inferences where the technique is the researcher's synthesis rather than published.

## Verification checklist

- At least 4 distinct multi-detector patterns surveyed.
- QLinearConv kernel design covered with explicit int weights example.
- Combination strategies covered for ≥ 3 options.
- Tradeoff table populated for kernel sizes 3, 5, 7.
- Quantization tactic explicit.
- task367 v2 + task077 worked examples sketched.
- All examples compatible with ONNX opset 18 + ORT 1.24.
