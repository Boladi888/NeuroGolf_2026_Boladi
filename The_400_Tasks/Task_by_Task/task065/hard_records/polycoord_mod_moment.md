route: polynomial count-squared coordinate moment replacing `unique_color_f`
live_bar: 353, sha256 d80a6a7add9e2aeacf687bf4839439867f3959ceed8430604ed14a6c735ab1f7, source `python tools/ng_current_bar.py --task 65 --json`
attempt: `prototypes/build_task065_w1_polycoord.py` / `prototypes/task065_w1_polycoord.onnx`; cost 339 = memory 287 + params 52
failed_gate: `python tools/ng_gate.py visible --task 65 --cand The_400_Tasks/Task_by_Task/task065/prototypes/task065_w1_polycoord.onnx` -> `61/266 (22.93%) FAIL`
why_dead: the algebraic replacement for marker row/col moments is not byte-correct under the ONNX/ORT graph, and visible miss means certain hidden zero; do not run truth/ab or build small variants in this coordinate-family.
remaining_hypothesis: a fundamentally different way to delete `unique_color_f [10]` would need avoid both a materialized [10] polynomial vector and ORT repeated-count ambiguity; otherwise the current 40-byte cast is likely cheaper than the repair.
