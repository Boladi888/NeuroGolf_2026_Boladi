# task119 — diagonal-Conv reformulation (from-scratch)

| Field | Value |
|---|---|
| Task | task119 |
| Family | Pattern_Recognition |
| Cost reduction | -80,529 |
| Estimated LB | +1.03 |
| Verify | 266/266 |
| Pattern | Whole-graph Conv reformulation |
| Build script | `The_400_Tasks/Task_by_Task/task119/prototypes/build_task119_diag_conv.py` |

## Rule

task119's algorithm depended on detecting diagonal patterns in the input grid. The original ONNX used a chain of standard 3×3 Convs + per-cell threshold/AND ops. The from-scratch rewrite replaced this with a **diagonal-convolution kernel** that captures the same diagonal-pattern detection in fewer ops.

## Key reformulation

```
# Before: chain of 3×3 Convs + per-cell ANDs to detect diagonal patterns
# Each Conv: [1,10,30,30] F32 intermediate = 36,000 bytes
# 6+ Conv intermediates × 36,000 = ~216,000 bytes plus the ANDs

# After: single diagonal Conv with specially-crafted kernel
# Diagonal kernel: 1.0 on the diagonal positions, 0 elsewhere
diag_kernel = np.zeros((1, 10, K, K), dtype=np.float32)
for i in range(K):
    diag_kernel[0, :, i, i] = 1.0    # main diagonal
# (or anti-diagonal, or both, depending on the pattern)

# Single Conv with this kernel captures diagonal sums directly
diag_response = Conv(input, diag_kernel)   # [1, 1, H', W'] F32 — much smaller intermediate
```

The single Conv replaces the chain of Convs+ANDs because the diagonal pattern can be captured by a single convolution kernel whose weights are non-zero only on the diagonal positions.

## Why it works

When the rule is "match a specific spatial pattern" (diagonal, cross, plus-shape, L-shape, etc.), a Conv with a hand-crafted kernel of the EXACT shape achieves the same detection in one op. The previous implementation built up the pattern detection from primitives (3×3 Convs + boolean combinations); the rewrite expressed the pattern directly in kernel weights.

Cost reduction:
- Eliminates intermediate F32 tensors between the Conv chain steps.
- Eliminates the AND ops + their intermediates.
- Single Conv output is smaller (just the response tensor, not per-channel partial sums).

## Generalizes to

`Algorithm_Library/algorithm_patterns.md` §9 (Whole-grid summary via single Conv) — same idea applied to a full-grid summary via 2^(i*W+j) weights.

For diagonal/shape detection specifically:
- Diagonal patterns: kernel with 1.0 on the diagonal.
- Cross patterns: kernel with 1.0 on the horizontal + vertical midlines.
- Plus patterns: kernel with 1.0 on the four orthogonal arms.
- Custom shapes: kernel weights = 1.0 at the shape's cell positions.

## Cross-ref

- `07_conv_reformulation/task374_conv_run_combined.md` — fused-Conv approach.
- `07_conv_reformulation/task323_conv_bool.md` — Conv→bool reformulation.
- `Algorithm_Library/algorithm_patterns.md` §9 (single-Conv whole-grid summary).

## Reproducibility

Read `The_400_Tasks/Task_by_Task/task119/prototypes/build_task119_diag_conv.py`.

To apply to a new shape-detection task:
1. Identify the specific spatial pattern your task needs to detect.
2. Craft a Conv kernel with non-zero weights at the pattern's cell positions, 0 elsewhere.
3. Choose padding to match the desired output shape.
4. Single Conv replaces the chain.
5. Verify byte-exact.

For non-diagonal patterns, this template applies broadly. The trick is recognizing that the rule has a specific spatial template that maps directly to convolution weights.
