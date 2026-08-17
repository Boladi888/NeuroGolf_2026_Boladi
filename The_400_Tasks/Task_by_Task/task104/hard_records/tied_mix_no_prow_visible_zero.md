# tied_mix_no_prow_visible_zero

- Live bar used: cost 118, SHA-256 `34571a7d5ea37c04e7562fe7ab21d293de3197cf9746aa02dec81bacda25d84c`, source `python tools/ng_current_bar.py --task 104 --json` / team manifest.
- Attempted route: remove `prow[2,3]` and reuse `mix[2,3,2]` for the input projection and `k` weights via `sum_k mix`; prototype `../prototypes/task104_tied_mix_112.onnx`, static cost 112, SHA-256 `7f9e927137c269d23100cd8bf84c5265a27e0b900e232a58e08f328032932af7`.
- Named cost delta: `python tools/ng_graph_cost_diff.py --task 104 --live Team_Best_Onnx/task104.onnx --candidate The_400_Tasks/Task_by_Task/task104/prototypes/task104_tied_mix_112.onnx` -> params `118 -> 112`, net `-6`; removed `prow [2,3]`, changed `mix` values.
- Exact failed gate: `python tools/ng_gate.py visible --task 104 --cand The_400_Tasks/Task_by_Task/task104/prototypes/task104_tied_mix_112.onnx` -> `[visible] 0/7 (0.00%)  FAIL`.
- Why dead: the tied tensor cannot keep the required positive background margin at the shifted edge rows/cols; best bounded fits still had negative sign margin before the visible-zero gate.
- One remaining hypothesis: a two-initializer tie where only the `k` weights are borrowed for free may exist, but full `prow=sum_k(mix)` tying should not be re-run.
