# task369 - QLinearConv scalar initializer shaves are dead

- Live bar: cost **982**, sha256 `06b03bcda0ed2309c93c2dee084bb9032a057483332ace744d1ff58b10695b01`, source `python tools/ng_current_bar.py --task 369 --json` / manifest.
- Attempted route: beat the 982 graph with a one-parameter legal shave from QLinearConv optional zero-points or bias inputs.
- Probe script: `The_400_Tasks/Task_by_Task/task369/prototypes/bo_ng_cdx_worker2_optional_zp_probe.py`.
- Failed gates:
  - optional zero-point omission failed ONNX checker before scoring: `Node (v_n2)'s input 2/5 is marked single but has an empty string in the graph`.
  - `task369_omit_first_bias.onnx` -> `ng_gate visible`: `[visible] 0/265 (0.00%)  FAIL`.
  - `task369_omit_second_bias.onnx` -> `ng_gate visible`: `[visible] 0/265 (0.00%)  FAIL`.
  - `task369_omit_both_biases.onnx` -> `ng_gate visible`: `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: QLinearConv zero-point inputs are not optional in this checker/ORT path, and the live bias thresholds are functionally load-bearing with the live downstream weights. Per visible-fail axiom, do not run truth/AB/battery or build minor variants that only delete these initializers without a full downstream refit.
- One remaining hypothesis: a different downstream refit may remove a bias only if it avoids padding-leaking explicit final bias and uses a legal scalar final zero-point.

