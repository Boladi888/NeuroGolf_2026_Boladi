# task357 hard record — sparse ConvInteger weight

- Live bar used: manifest/current_bar cost 279, sha256 `42ebe5e0ea08962858f8bd082f711a22b5996eea1510d58386b7f0e5e5f19d5c` (Vamshi, 2026-07-12T15:15:54Z).
- Attempted route: replace the live graph's dense `weights` `[10,1,1,1]` ConvInteger initializer with a sparse initializer containing only the two nonzero entries. Prototype: `prototypes/work_task357_probe_sparse_weight_live279_7bcd1648.onnx`, sha256 `7bcd1648fd013d345720766442164704edac48cb104541d3332f33a512c9a8a8`.
- Intended saving: params 32 -> 24 if sparse ConvInteger weights were accepted, a byte-equal cost 271-ish sibling under the 279 manifest bar.
- Failed gate: `_score_v2.score_network_v2(...) -> None`; `onnx.checker.check_model(full_check=True)` and strict shape inference both fail with `ConvInteger`: `inputs are expected to have tensor type and output type should not be null`.
- Why dead: sparse initializers are not accepted as `ConvInteger` weights by the scorer's checker/shape-inference path, even though ORT can load the raw model. No visible/truth/ab gates run because the model is unscoreable.
- One remaining hypothesis: a non-sparse way to share or derive the two nonzero channel weights would need to preserve the free `ConvInteger` output and save at least 1 charged byte; no such initializer-sharing surface is visible in the live graph.
