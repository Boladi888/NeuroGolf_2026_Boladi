# task021 — `mul_mask_to_where` (Mul of bool cast × value → Where)

| Field | Value |
|---|---|
| Task | task021 |
| Pattern name | `mul_mask_to_where` (techniques.md §1 rule 2) |
| Cost reduction | -3,119 |
| Detector | `tools/spread_patterns/pass_mul_mask_to_where.py` |

## Before

```
Mul(Cast(bool_op, F32), V)
   M = Cast(bool_op, F32)   intermediate F32 [...,H,W] = 4·H·W bytes
   M × V                     result = V where bool_op is True, 0 where False
```

The F32 mask intermediate is wasted memory. The Cast is a wasted op.

## After

```
Where(bool_op, V, 0_const)
   selects V where bool_op is True, 0 where False
```

- No F32 mask intermediate (replaced by the bool input directly).
- One fewer Cast node.
- One additional 0-const initializer (~4 bytes for a scalar).

## Why it works

`0 × V = 0` and `1 × V = V`. Where with bool condition does the same selection: `True → V`, `False → 0`. The Cast(bool, F32)'s only purpose was to produce 0/1 floats for the Mul; with Where, we use the bool directly.

V can be any dtype — the Where output dtype matches V's dtype.

## Generalizes to

`techniques.md` §1 rule 2 (`mul_mask_to_where`). Universal rewrite for any "mask gates a value" structure where the mask is bool-derived.

This is the **value-gating** counterpart to rule 1 (`bool_andfold`, which handles mask × mask). Together they cover the two main bool-Mul shapes:
- Mul(mask, mask) → And then Cast
- Mul(mask, value) → Where(mask, value, 0)

## Reproducibility

Detector is fully automated. Run `tools/spread_scan.py --tasks taskNNN`. The detector requires:
- The Mul has exactly 2 inputs.
- Exactly one input is Cast(bool_op, F32) where bool_op is one of {Equal, Greater, Less, Not, And, Or, Xor, GreaterOrEqual, LessOrEqual}.
- The other input is any tensor (the "value" to be gated).
- The Cast output is consumed only by this Mul (else it'd orphan; skip).

If both Mul inputs are bool casts → that's `bool_andfold`'s territory, not this rule.
