# task370 hard record — drop k=4 diagonal copy cap

- Live bar: cost `3607`, profile `3607`, sha256 `c5a03117601f462483d58aa02b9006f792e53b9c77502fd501481d1ad84d89c0`, source `python tools/ng_current_bar.py --task 370 --json` (`authoritative_source=manifest`, owner Vamshi, updated `2026-07-12T15:15:54Z`).
- Attempted route: shrink Vamshi's combined `k={3,4}` 3x3 branch to only `k=3` by changing `kv34` from `[3,4]` to `[3]` and `shpf34` from `18` to `9`.
- Artifact: `The_400_Tasks/Task_by_Task/task370/prototypes/task370_drop_k4_probe.onnx`; builder `The_400_Tasks/Task_by_Task/task370/prototypes/build_task370_drop_k4_probe.py`.
- Exact failed gate: `python tools/ng_gate.py visible --task 370 --cand The_400_Tasks/Task_by_Task/task370/prototypes/task370_drop_k4_probe.onnx --no-watch-queue` -> `[visible] 262/266 (98.50%)  FAIL`.
- Why dead: one visible miss is certain hidden zero; four visible officials require the fourth diagonal translated copy in this representation. Do not run truth/profile variants that cap the render at three copies.
- One remaining hypothesis: savings must preserve the `k=4` copy, likely by reducing index/scalar plumbing or replacing the scatter/render family entirely.
