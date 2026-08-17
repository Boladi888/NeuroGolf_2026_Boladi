# task266 — scalar broadcast (uniform-value init → scalar)

| Field | Value |
|---|---|
| Task | task266 |
| Pattern | `scalar_broadcast_ones` (techniques.md §17) |
| Cost reduction | small (~14 bytes on task266 staged; -99 params on task000 origin) |
| Detector | `tools/spread_patterns/pass_scalar_broadcast_ones.py` |

## Before

A non-scalar uniform-value initializer, e.g.:
```
ones10 = init [10] F32 all 1.0   # 10 params
```

## After

```
ones_scalar = init [1] F32 = [1.0]   # 1 param (or [] = scalar shape, also 1 param)
```

Downstream broadcasts replace the implicit-via-init shape. Mul against the scalar produces the same broadcast result as Mul against the `[10]` init.

## Why it works

When every element of a non-scalar init has the same value, the shape is irrelevant. ONNX broadcasting matches a `[1]` or `[]` scalar against any shape implicitly. Replacing the init with a scalar saves params (which is element-count, NOT byte-count — so 10 elements vs 1 element is 9 params saved).

## Generalizes to

`techniques.md` §17 (scalar_broadcast_ones). Apply to any uniform-value initializer.

Detector: `pass_scalar_broadcast_ones.py`. Scans every initializer, checks if all elements are equal, rewrites to scalar.

## Why params matter even when bytes don't change

For a `[10] F32` init: 10 params × 4 bytes/element = 40 bytes on disk.
For a `[1] F32` init: 1 param × 4 bytes = 4 bytes on disk.

But the `cost = memory + params` formula counts the ELEMENT COUNT of params, not the byte count. So:
- `[10] F32` = 10 params.
- `[1] F32` = 1 param.
- 9 params saved.

In the log-scale points formula, 9 params off a 10,000-cost task moves points by `log(10000/9991) ≈ +0.0009`. Trivial absolutely; meaningful when stacked across many tasks (the spread_scan finds these on dozens of tasks in one sweep).

## Reproducibility

Detector fully automated. Run `tools/spread_scan.py`.

## Cross-ref

- `08_gather_lookup/task167_opset9_attr_shrink.md` — separate init compression technique.
- `techniques.md` §17.
