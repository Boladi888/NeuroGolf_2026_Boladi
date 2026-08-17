# task012 TopK indices-only dead route

- Live bar: **858**, SHA-256 `e936055201ecc0182097c6faf68bae80c1dad3a95a9e7772c2b601aaddfbb87e` from `python tools/ng_current_bar.py --task 12 --json`; local staged bar to beat is **821**, SHA-256 `8c8c54ecfc685f4f35895c7a961889ae8b0dad34e3920a8bd3e006edc40e86ee`.
- Attempted route: drop the unused `TopK` values output `topv[1,3,1,1]` from the 821 Path-A graph to save 12 B while keeping `topi` for B/A color routing.
- Exact failed gate: build/checker probe, before scoring or visible. `python The_400_Tasks/Task_by_Task/task012/prototypes/build_topk_indices_only_probe.py` fails `onnx.checker.check_model(full_check=True)` with `Node ()'s output 0 is marked single but has an empty string in the graph` for `OpType: TopK`.
- Why dead: `TopK` does not have an optional first output in this scorer/checker stack, so the 12-byte values tensor is forced as long as the graph uses `TopK`.
- One remaining hypothesis: replace `TopK` entirely with a cheaper color-rank primitive; price first, because nonzero-min or bg-suppressed rank logic materializes at least one extra 10-lane predicate/count tensor and loses to the current 36 B `topv+topi` pair.
