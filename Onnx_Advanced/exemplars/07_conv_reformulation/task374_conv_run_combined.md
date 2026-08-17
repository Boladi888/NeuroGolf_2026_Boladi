# task374 — `conv_run_combined` (fused Conv + downstream)

| Field | Value |
|---|---|
| Task | task374 |
| Cost reduction | -11,728 |
| Pattern | Conv reformulation — fuse downstream into Conv weights |

## Rule

task374's original ONNX has `Conv → downstream_ops`. The downstream ops can be folded into the Conv weights themselves, eliminating the post-Conv intermediates.

## Technique

When a Conv is followed by a deterministic linear combination of channels (e.g., Mul by a per-channel constant + Sum), the per-channel constants can be MULTIPLIED INTO the Conv weights:

```
# Before
y = Conv(x, W) → [1, C_out, H, W]
z = Mul(y, alpha [1, C_out, 1, 1]) → [1, C_out, H, W]
result = ReduceSum(z, axes=[1], keepdims=0) → [1, H, W]

# After: fold alpha into W
W' = W * alpha (per-channel)
y' = Conv(x, W') → [1, C_out, H, W]   # but the per-channel multiplication is now baked in
result = ReduceSum(y', axes=[1], keepdims=0) → [1, H, W]
```

The Mul node and its `alpha` initializer are gone. The intermediate between Conv and Mul is gone.

## Why it works

Linearity. `Mul(Conv(x, W), alpha) = Conv(x, W * alpha)` when alpha broadcasts per-channel. Mathematically identical; one fewer op + one fewer intermediate.

## Generalizes to

Apply when:
- Conv is followed by a `Mul(per-channel scalar)`.
- That Mul is followed by a reduction or further linear op.
- The downstream linear ops can be expressed as Conv weight modifications.

Common downstream foldings:
- `Mul(per-channel α)` → multiply weights by α.
- `Add(per-channel β)` → bias the Conv (Conv has a bias input).
- `BatchNorm` (with constant scale/bias) → fold into weights + bias.

## Caveat: precision and overflow

Folding alpha into W changes the magnitude of the weights. If the original alpha was in `[0.01, 0.1]` range, the new W' may be small. Watch for F16 quantization issues at the smaller magnitudes; F32 weights are safer for folding.

## Reproducibility

Manual application:
1. Find Conv → Mul → downstream chains.
2. Verify Mul's scalar is per-channel and constant.
3. Compute `W' = W * alpha_broadcast` in numpy.
4. Update Conv's weight initializer.
5. Drop the Mul node and its alpha init.
6. Verify byte-exact (small numerical drift possible — accept ε if downstream thresholds at `> 0.0`).

## Cross-ref

- `01_whole_graph_rewrites/task119_diag_conv.md` — different reformulation (custom Conv kernel).
- `07_conv_reformulation/task323_conv_bool.md` — Conv → bool reformulation.
- `Algorithm_Library/algorithm_patterns.md` §9.
