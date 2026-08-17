# task194 qlinear optional-zp and shared-bias routes

- live bar: cost 234, sha256 `5b633082bb0cd78d1eded256094a1dddf9241aa56bae15676868013a5d500a4e`, source `python tools/ng_current_bar.py --task 194 --json` / team manifest.
- attempted route: preserve Sho's 7-node `Conv -> Cast -> Reshape -> Mul -> Concat -> Gather -> QLinearConv` topology and shave final `QLinearConv` params by omitting zero-point inputs or replacing the 10-element bias with a cheaper no-bias/shared-bias formulation.
- exact failed gate:
  - `python The_400_Tasks/Task_by_Task/task194/prototypes/build_task194_qlinear_optional_zp_probe.py omit_xw_zp` -> `onnx.checker.ValidationError: Node ()'s input 2 is marked single but has an empty string ... OpType: QLinearConv`.
  - `python The_400_Tasks/Task_by_Task/task194/prototypes/build_task194_qlinear_optional_zp_probe.py omit_y_zp` -> `onnx.checker.ValidationError: Node ()'s input 7 is marked single but has an empty string ... OpType: QLinearConv`.
  - coefficient search for `ConvInteger` no-bias form `a*(x-z)+b*(x*x-z)` over `z in [-128,127]`, `a,b in [-255,255]` found no solution isolating all codes `1..10` from padding/code `0`.
  - coefficient interval search for `QLinearConv` shared scalar bias `a*x+b*x*x+B` over int8 weights found disjoint feasible bias ranges across colors; no common `B` exists.
- why dead: installed ONNX checker requires the zero-point inputs for this `QLinearConv`; the current per-channel quadratic classifier also needs independent per-output constants, so the 10-element bias is not a removable redundancy under int8 weight constraints.
- one remaining hypothesis: beat 234 only by eliminating or fusing one of the forced-looking intermediates (`features6` 72 B, `code3_f` 36 B, or the 36-element flat index), likely via a genuinely different fused spatial/color op or a new one-node factorization below the older 283-Einsum floor.
