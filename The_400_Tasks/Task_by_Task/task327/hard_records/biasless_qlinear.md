live bar: 268, sha 01ffc402949dce588c2e48584704325124f0355b57a7ffac798664e394b279eb
route: remove the final 10-element QLinearConv bias by using shifted input zero-points and a common output offset so the zero-point expansion supplies the quadratic constants.
failed search: scalar zero-point brute force over -20..30 found no all-color separator; per-channel shifted zero-point interval search found no integer INT8 two-weight separator for plausible outside sentinels; broader convex/concave pass timed out without a hit.
why dead: with only two features `(x, x^2)` and no per-channel bias/y_zero vector, the final one-vs-one color tests cannot all be represented by the constrained QLinearConv affine form while keeping padded cells non-positive.
remaining hypothesis: a different nonlinear feature could replace `x^2`, but it must cost less than the current 36-byte INT8 square plus 72-byte carrier; no such legal ORT op was identified in this pass.
