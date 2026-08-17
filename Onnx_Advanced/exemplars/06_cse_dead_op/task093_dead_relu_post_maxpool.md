# task093 — dead Relu after pad-zero MaxPool

| Field | Value |
|---|---|
| Task | task093 |
| Pattern | `dead_relu_post_maxpool` (techniques.md §16) |
| Cost reduction | -3,136 (4 dead Relus dropped) |
| Detector | `tools/spread_patterns/pass_dead_relu_post_maxpool.py` |

## Before

4 Relu nodes (indices 11, 13, 15, 17) each downstream of `Sub(MaxPool(X, pad-symmetric, stride=1), X)`:

```
Relu(Sub(MaxPool(X, kernel=K, pads=sym, stride=1), X))
```

Each Relu is provably an identity for these specific inputs.

## After

```
Sub(MaxPool(X, kernel=K, pads=sym, stride=1), X)   # Relu removed; identity
```

## Why it works (the math)

MaxPool with stride=1 and symmetric pads (`pad_start[d] + pad_end[d] == kernel[d] - 1`) guarantees that the output window at position `[i, j]` includes the input position `[i, j]`. Specifically:

- Output `[i, j]` is computed from input window starting at `[i - pad_start, j - pad_start]` and spanning `kernel` rows/cols.
- Since `0 ≤ pad_start ≤ kernel - 1`, the window includes input position `[i, j]`.
- Therefore `MaxPool(X)[i, j] ≥ X[i, j]` pointwise.
- `Sub(MaxPool(X), X) ≥ 0` pointwise.
- `Relu(≥0)` is identity.

The 4 Relus are dead code.

## Safety conditions (detector verifies)

- MaxPool has `strides == [1, 1, ...]` (or strides absent — default).
- MaxPool has `kernel_shape` with all values ≥ 1.
- MaxPool has `pads` satisfying `pad_start[d] + pad_end[d] == kernel[d] - 1` for each spatial dim d. (This is the "preserves output shape" condition.)
- Sub's second input IS the same tensor that MaxPool consumed as its input.

If any condition fails, the Relu may not be identity, and the detector skips.

## Generalizes to

`techniques.md` §16. Apply to any `Relu(Sub(MaxPool(X, sym-pad, stride=1), X))` chain.

Same logic with reversed sign: `Relu(Sub(X, MinPool(X, sym-pad, stride=1)))` — also identity, but ONNX has no MinPool op directly (must negate MaxPool of negated input).

## Out of scope

- Asymmetric padding (pad_start[d] + pad_end[d] != kernel[d] - 1): output shape changes, output window may not contain input position.
- Non-unit stride: output positions may not include input positions.
- Wider variants like `Relu(Sub(Op(X), Y))` where Y != X: generic non-negativity isn't provable.

## Reproducibility

Run `tools/spread_scan.py --tasks taskNNN`. Detector automated. Worth running on any graph with MaxPool + Sub + Relu chains.

## Cross-ref

- `06_cse_dead_op/task302_drop_relu_shift.md` — different drop-Relu variant.
- `techniques.md` §16.
