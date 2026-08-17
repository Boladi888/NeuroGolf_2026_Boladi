# task233 — `equal_fold_distance_threshold` (Sub→Mul(self)→Less<1 → Equal)

| Field | Value |
|---|---|
| Task | task233 |
| Family | Color_and_Logical |
| Cost reduction | -32,000 |
| Estimated LB | +0.5510 |
| Pattern name | `equal_fold_distance_threshold` (techniques.md §5) |
| Detector | `tools/spread_patterns/pass_equal_fold_distance.py` |

## Before

```
diff   = Sub(v41, v12)              # both inputs F32 with exact-integer values
sq     = Mul(diff, diff)            # squared distance (Mul of self)
result = Less(sq, 0.25)             # threshold: True iff |diff| < 0.5
```

Three nodes producing two intermediate F32 tensors. The math computes "is v41 within 0.5 of v12" — but since both are exact integers, this is equivalent to exact equality.

## After

```
result = Equal(v41, v12)
```

One node, no intermediate F32 tensors. The threshold init (0.25) is also dropped.

## Why it works

The Sub-Mul-Less idiom is computing `(v41 - v12)² < 0.25`, i.e. `|v41 - v12| < 0.5`. When both v41 and v12 are exact-integer F32 values (which they are for one-hot basis vectors and color codes in NeuroGolf):

- If v41 == v12: diff = 0, sq = 0, Less(0, 0.25) = True. Equal(v41, v12) = True. ✓
- If v41 ≠ v12: diff = ±1 (or larger), sq ≥ 1, Less(≥1, 0.25) = False. Equal(v41, v12) = False. ✓

The substitution is exact for integer-valued inputs. For non-integer inputs (e.g., values like 1.3 vs 1.5), the math diverges — Less would still be True, Equal would be False. The detector includes a safety check: only fires when v41 and v12 are provably integer-valued (one-hot basis vectors, color codes).

## Variants

### Pointwise (this task)

```
Cast(Less(Mul(Sub(v41, v12), self), 0.25), F32)   →   Cast(Equal(v41, v12), F32)
```

3 ops → 1. Saved -32,000 bytes on task233.

### With ReduceSum (task133)

```
Less(ReduceSum(Mul(Sub(scalar_v13, v1), self), axes=[k]), 0.25)
   →
Squeeze(Equal(scalar_v13, v1), axes=[k])
```

Over a size-1 axis. Drops 4 ops + the threshold init. Saved -55,636 bytes on task133.

## opset-10 caveat

Under opset 10, `Equal` accepts INT32/INT64 only — NOT FLOAT. The simple rewrite `Equal(v41_f32, v12_f32)` fails to load.

Workarounds:
1. Bump opset to 11+ (Equal accepts BOOL/FLOAT).
2. Cast inputs to INT32 first. When the same value is compared against N constants, **share one Cast** across all N — see `task368_shared_cast_equal_fold.md`.

## Generalizes to

`techniques.md` §5 (Equal-fold variants). Apply to any Sub→Mul(self)→Less<1 chain on integer-valued inputs.

Detector: `pass_equal_fold_distance.py` handles the pointwise form. ReduceSum variant requires manual rewrite or a future pattern.

## Reproducibility

Run `tools/spread_scan.py --tasks taskNNN`. Detector verifies:
- Less node with constant threshold in `(0, 1)`.
- Less's first input is Mul output where both Mul inputs are the same tensor (self-squaring).
- Mul's input is Sub output.
- Sub's two inputs are provably integer-valued (traced to one-hot or color sources).

Math validation: the detector also does sanity-verify against random inputs to confirm the rewrite preserves output.

## Cross-ref

- `Algorithm_Library/algorithm_patterns.md` §1 (color permutation) and §2 (bounding box) — Equal-fold often combines with these for color-equality predicates.
- `task368_shared_cast_equal_fold.md` — the N-way shared-Cast variant.
- `task133_equal_fold_reducesum.md` (in `04_algebraic_identities/`) — the ReduceSum variant.
