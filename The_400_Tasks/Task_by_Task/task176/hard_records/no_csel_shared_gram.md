# task176 no-csel shared Gram probe

- live bar: cost 84, sha256 `96c22521d7b20c19767e0c6b4347b68a34219a8502fa6a49d57584f038c1d5cc`, source `python tools/ng_current_bar.py --task 176 --json` / `Team_Best_Onnx/task176.onnx`
- attempted route: delete `csel[2,2]` from the 84-byte shared-basis free-output Einsum, forcing `basis[:,h] dot basis[:,w]`; prototype `prototypes/v_no_csel80_probe.onnx`, builder `prototypes/build_no_csel80_probe.py`
- expected cost: static `(macs=0, memory=0, params=80)`, named delta `-csel[2,2] = -4 params`
- failed gate: `python tools/ng_gate.py visible --task 176 --cand The_400_Tasks/Task_by_Task/task176/prototypes/v_no_csel80_probe.onnx` -> `[visible] 0/25 (0.00%)  FAIL`
- why dead: without a 2x2 coupling the shared-basis spatial field is symmetric or rank-1; task176 needs asymmetric same-color cells, e.g. `(row=0,col=1)` is background while `(row=1,col=0)` is yellow, so a no-csel shared basis cannot represent the visible target.
- remaining hypothesis: only revisit sub-84 if a different mechanism removes or replaces `color[10,2]` or `basis[2,30]` without a charged pre-output tensor; do not retry no-csel/shared-Gram variants.
