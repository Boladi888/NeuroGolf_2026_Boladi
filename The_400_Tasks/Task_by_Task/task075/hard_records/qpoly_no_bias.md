# task075 qpoly no-bias hard record

Live bar: 917 by `python tools/ng_current_bar.py --task 75 --json`, manifest source, SHA-256 `5b8a3382dbe5eab32020a32e2b96936d0b841df2a4a9359f1d908a6fab9fbc82`.

Attempted route: remove the final qpoly `qB[10]` bias from the `[idx, idx^2] -> QLinearConv -> output` carrier and use input zero-points as the implicit constant term.

Failed gate: pre-build algebraic feasibility check. Exhaustive bounded searches over INT8 weights for global feature zero-points found no classifier that makes exactly one code positive and all other reachable codes non-positive, both for all codes 1..10 and for the generator-aware subset 2..10.

Why dead: a no-bias quadratic with shared zero-points has a constant term tied linearly to its two weights. The per-class singleton tests need independent quadratic offsets; the current explicit bias vector supplies that missing degree of freedom. Removing it without changing the final-output family breaks at least one class.

One remaining hypothesis: replace the qpoly final-output family entirely, e.g. a different free-output op that can perform channel expansion and padding without the two-feature `[idx, idx^2]` carrier.
