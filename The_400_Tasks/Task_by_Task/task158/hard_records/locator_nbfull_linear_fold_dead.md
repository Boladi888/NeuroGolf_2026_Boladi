# task158 locator `nbfull` single-threshold fold — DEAD (2026-07-14)

**Live bar:** manifest/team-best `task158.onnx` cost/profile **18,560**, SHA-256
`b3be1ca261dfb2a6c499594461c0036280f0e05e0c233ba9e7126874bad8f32b`.

## Attempted route

The live locator computes `keyo`, `keyr`, `keyany = Max(keyo,keyr)`, `nbfull`, then
`kcand = keyany - nbfull` to reject all-non-background 3x3 windows. A tempting
shave is to fold `key_orientation AND any_bg` into each orientation kernel, then
delete `nbfull` and `Sub`, saving about 1,012 bytes if a single QLinearConv
threshold can represent the folded predicate.

## Exact failed gate

Pre-build feasibility check over all 512 binary 3x3 background masks:

```text
scipy.optimize.linprog constraints for keyo AND any_bg: infeasible
scipy.optimize.linprog constraints for keyr AND any_bg: infeasible
```

No ONNX candidate was built and no visible/truth/profile gate was run, because the
exact predicate is not linearly separable as one threshold.

## Why it is dead

`key_orientation` is a threshold that accepts the all-non-background window; the
separate `nbfull` term removes exactly that corner case. Adding `any_bg` creates a
non-linearly-separable predicate over the 3x3 background mask, so one replacement
QLinearConv cannot preserve byte-equality. Any exact version needs at least one
extra nonlinear tensor, which erases the intended one-kernel saving.

## Remaining hypothesis

A different locator formulation could still remove `nbfull`, but it must not be a
single threshold fold. It would need either a cheaper nonlinearity than the current
`nbfull` QLinearConv + `Sub`, or a different legend locator that ranks the same
windows without accepting all-non-background patches.
