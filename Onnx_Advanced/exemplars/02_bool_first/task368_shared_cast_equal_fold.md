# task368 — `shared_cast_equal_fold` (N-way parallel Equal-fold with shared Cast)

| Field | Value |
|---|---|
| Task | task368 |
| Cost reduction | -7,345 (this rewrite) + -11,992 (slice_chain_to_multiaxis on the same task) = -19,337 combined |
| Pattern name | `shared_cast_equal_fold` (techniques.md §5, opset-10 variant) |
| Detector | `tools/spread_patterns/pass_shared_cast_equal_fold.py` |

## Before

10 parallel `Sub→Abs→Less<0.5` chains, each comparing the same float value `ocol` against a different color constant:

```
for k in 0..9:
    diff_k = Sub(ocol, color_k_const)        # F32
    abs_k  = Abs(diff_k)                     # F32
    eq_k   = Less(abs_k, 0.5)                # BOOL
```

30 F32 intermediates plus 10 threshold inits. Total per-channel cost ~720 bytes intermediate.

If you tried the naive Equal-fold (per-comparison):

```
for k in 0..9:
    eq_k = Equal(Cast(ocol, INT32), Cast(color_k_const, INT32))
```

Under opset 10, Equal requires INT32 — so each comparison needs two Casts. The 20 Cast nodes outweigh the savings from dropping Sub/Abs/Less.

## After (shared-Cast variant)

```
ocol_int = Cast(ocol, INT32)            # ONE Cast, shared across all 10 comparisons
color_k_int = INT32 initializer with the constant values

for k in 0..9:
    eq_k = Equal(ocol_int, color_k_int)
```

- 1 shared Cast vs N=10 per-comparison Casts.
- 10 Equal nodes (each is BOOL output, ~900 bytes for [1,1,30,30]).
- 10 INT32 inits at trivial bytes.

Net: -7,345 bytes on task368.

## Why share the Cast?

Under opset 10:
- `Cast(ocol_f32, INT32)` produces an INT32 tensor at the input shape.
- This Cast result can be consumed by N downstream Equal nodes.
- ORT supports this fanout natively.

If you don't share (N separate Casts), you produce N copies of the same INT32 tensor — wasted memory and N extra nodes.

If you bump opset to 11+ (Equal accepts float/numeric inputs; BOOL was already supported), you avoid the Cast entirely but pay the opset-bump validation cost. The shared-Cast variant works under opset 10 and is the more conservative rewrite.

## Generalizes to

`techniques.md` §5 (Equal-fold variants — specifically the multi-channel shared-Cast case). Apply when:

- N parallel comparisons of the same variable value against different constants.
- Model is opset 10 and the comparison input is float-valued in the graph (Equal needs an INT32/INT64 cast for numeric color codes).
- The variable and constants are provably integer-valued.

For N=1, the shared-Cast variant doesn't apply — use plain Equal-fold (`pass_equal_fold_distance`) or bump opset.

Detector: `pass_shared_cast_equal_fold.py`. It scans for parallel `Sub-Abs-Less<1` chains over a shared variable input, then emits:

```python
v_int = Cast(v, INT32, ...)
ck_int = (new INT32 initializer)
Equal(v_int, ck_int)
```

per chain.

## Reproducibility

Detector runs as part of `tools/spread_scan.py`. The shared-Cast emission requires careful handling:

1. Insert ONE `Cast(v, INT32)` node before the first downstream Less node that uses it (to maintain topological order — producer before consumers).
2. For each chain, create a new INT32 initializer with the constant value.
3. Replace each `Less(Abs(Sub(v, ck)), eps)` with `Equal(v_int, ck_int)`.
4. Drop the now-orphan Sub, Abs, Less, and the original float ck initializer (if no other consumer).

Manual application: when you see a parallel structure like task368's 10-channel color compare, use this pattern explicitly. ANY task that compares a tensor against 10 color codes can benefit.

## Cross-ref

- `ort_reality.md §4` — the opset-10 Equal dtype limit (BOOL/INT32/INT64; floats need opset 11+ or an INT cast).
- `task233_equal_fold.md` — the single-comparison Equal-fold (different op shape).
