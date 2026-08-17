# task058 single-Einsum prefix basis dead route

Live bar: cost/profile 128, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Attempted route: replace the live `ReduceL2 -> Pow -> Einsum` analytic spiral with a one-node `Einsum` that derives size features from two contracted copies of `input` and uses a 30x3 coordinate basis plus 3x3x3 coefficients. Static cost was 127.

Failed gate: `python tools/ng_gate.py visible --task 58 --cand The_400_Tasks/Task_by_Task/task058/prototypes/task058_prefix_einsum127.onnx` returned `[visible] 0/22 (0.00%)  FAIL`.

Why dead: the prefix/basis single-op formulation does not reproduce even the visible size-dependent spiral; it cannot be reviewer-ready and hidden would be a certain zero.

One remaining hypothesis: keep the live side-length scalar extraction and look for a true algebraic param deletion inside the current phase/Einsum formula, not a learned prefix surrogate.
