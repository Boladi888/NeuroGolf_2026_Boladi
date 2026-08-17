# DEAD ROUTE — task205 mixed-typed Slice axis reuse — 2026-07-15

- Live bar: manifest live `2084`, sha `a695b2481d6664c45e950ba07ba13c5fca443bc45e88c38a91c67ed92dc26fd8`.
- Attempted route: after the valid init dedup, replace `axCsl` INT32 `[2]` with existing `axCpad` INT64 `[2]` to save one more initializer element.
- Exact failed gate: `onnx.checker.check_model(full_check=True)` in `prototypes/task205_vamshi2084_axisreuse_probe.py`.
- Why dead: ONNX `Slice` requires `starts`/`ends`/`axes` to share the same integer type; checker raised `Slice: axes has inconsistent type tensor(int64)` because `cf`/`cfE` are INT32.
- One remaining hypothesis: only a rank-2 C-branch rewrite that removes the column singleton and keeps INT32 axes throughout could remove `axCsl`, but the current graph needs the `[1,1,10]` channel-mask shape for upstream broadcast/Einsum use, so any reshape/squeeze costs more than the saved param.
