# task133 — Equal-fold with ReduceSum variant

| Field | Value |
|---|---|
| Task | task133 |
| Pattern | Equal-fold (ReduceSum variant) — techniques.md §5 |
| Cost reduction | -55,636 |
| Estimated LB | +0.4238 |

## Before

```
diff   = Sub(scalar_v13, v1)                # F32
sq     = Mul(diff, diff)                    # squared diff F32
sumsq  = ReduceSum(sq, axes=[k])            # reduce over size-1 axis k → still scalar-ish
result = Less(sumsq, 0.25)                  # threshold
```

Four ops + three F32 intermediates. The ReduceSum is over a size-1 axis (i.e., it just removes a trivial dim, equivalent to Squeeze).

## After

```
result = Squeeze(Equal(scalar_v13, v1), axes=[k])
```

One Equal + one Squeeze. Drops 4 ops and the threshold init (the 0.25 constant is eliminated).

## Why it works

For integer-valued v13 and v1 (true for one-hot basis vectors and color codes):

- `(v13 - v1)² < 0.25` iff `|v13 - v1| < 0.5` iff `v13 == v1` (since integer differences are 0 or ≥ 1).

The ReduceSum over a size-1 axis is just Squeeze — it removes a trivial dim. After dropping the Sub-Mul-Less chain, you can replace the ReduceSum with Squeeze directly.

## Why this variant is a separate detector

The basic `equal_fold_distance_threshold` detector handles the POINTWISE form (task233). When the chain has a `ReduceSum(axes=[k])` over a size-1 axis between Mul and Less, the detector needs an additional check to recognize the trivial reduction.

Current `pass_equal_fold_distance.py` handles only the pointwise case. The ReduceSum variant requires manual rewrite or a future pattern extension.

## Generalizes to

`techniques.md` §5 (Equal-fold). Specifically the ReduceSum-over-size-1-axis variant.

Apply when:
- Sub → Mul(self) → ReduceSum(axes=[k], keepdims=0) → Less<1 chain.
- The reduced axis is size-1 (so the reduction is just dim removal).
- Both Sub inputs are integer-valued.

## Reproducibility

Manual application:
1. Find Less nodes with threshold in (0, 1).
2. Trace back: Less → ReduceSum → Mul(self) → Sub.
3. Verify the ReduceSum axis is size-1 in static shape inference.
4. Replace with `Squeeze(Equal(sub_input_a, sub_input_b), axes=[k])`.

If the reduce axis is NOT size-1, the rewrite changes math — don't apply.

## Cross-ref

- `02_bool_first/task233_equal_fold.md` — pointwise version.
- `02_bool_first/task368_shared_cast_equal_fold.md` — N-way shared-Cast for opset-10.
- `techniques.md` §5 — full Equal-fold discussion including variants.
