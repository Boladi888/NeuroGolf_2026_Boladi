# task293 — `bool_andfold` (Mul of two bool casts)

| Field | Value |
|---|---|
| Task | task293 |
| Pattern name | `bool_andfold` (techniques.md §1 rule 1) |
| Cost reduction | -1,575 (per task293 sweep) |
| Detector | `tools/spread_patterns/pass_bool_andfold.py` |

## Before

10 parallel Mul nodes, each multiplying two F32 casts of bool predicates:

```
Mul(Cast(Equal(out_scalar_raw_0, ...), F32), Cast(valid_mask_bool, F32))   # F32 intermediate
Mul(Cast(Equal(out_scalar_raw_1, ...), F32), Cast(valid_mask_bool, F32))   # F32 intermediate
...
Mul(Cast(Equal(out_scalar_raw_9, ...), F32), Cast(valid_mask_bool, F32))   # F32 intermediate
```

Each Mul output: F32 [1, 1, 30, 30] = 3,600 bytes. Plus the Cast intermediates each = 3,600 bytes.

## After

```
And(Equal(out_scalar_raw_0, ...), valid_mask_bool) → bool_0 BOOL [1,1,30,30] = 900 bytes
And(Equal(out_scalar_raw_1, ...), valid_mask_bool) → bool_1 BOOL [1,1,30,30] = 900 bytes
...
Cast(And_output, F32)   (only once, at the consumer boundary)
```

Each And output: BOOL [1, 1, 30, 30] = 900 bytes (4× smaller than F32). The intermediate Casts are gone.

## Why it works

Mathematical identity: `0/1 × 0/1 = 0/1 AND` on integers. When both inputs are bool casts (values exactly {0.0, 1.0}), Mul produces 0.0 (one false) or 1.0 (both true) — which is exactly And's result, just in F32.

Replacing Mul with And and Casting once at the end:
- Same logical output.
- 4× memory reduction on every intermediate (F32 4 bytes → BOOL 1 byte).
- Net -1 node (one Cast eliminated).

## Special case on task293: shared valid_mask Cast

task293's `valid_mask` Cast feeds ALL 10 parallel Muls. The detector handles this by checking: "if every consumer of the Cast is a Mul we're rewriting, the Cast becomes dead and gets dropped." All 10 Muls → 10 Ands, the `valid_mask` Cast has 0 surviving consumers → dropped.

## Generalizes to

`techniques.md` §1 rule 1 (`bool_andfold`). This is the most mechanical of the bool-first rules — almost always applies when both Mul inputs are bool casts. Detected by `pass_bool_andfold.py`.

## Reproducibility

Pattern detector is fully automated. Run `tools/spread_scan.py --tasks taskNNN` and inspect for `bool_andfold` applied verdict. Manual application: find Mul nodes, check both inputs are Cast nodes from bool-producing ops (Equal, Greater, Less, Not, And, Or, Xor) to F32, and check the Cast outputs are single-use OR exclusively consumed by Muls in this rewrite set.
