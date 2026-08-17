# task034 hard record: redonly_corner_detector

- live bar: 1334 by `python tools/ng_current_bar.py --task 34 --json`, sha256 `642a13a7dd051aff43afb6a4d6b9217d378d42f747d099d80ed8b15c30e3a268`; `_score_v2` on `Team_Best_Onnx/task034.onnx` = memory 1129 + params 205.
- attempted route: slice only red channel 2 over rows/cols `1:8`, drop the background channel from the four 2x2 QLinearConv seed detectors, and keep the rest of the current-best graph unchanged; probe `prototypes/task034_redonly_probe.onnx`.
- exact failed gate: `python tools/ng_gate.py visible --task 34 --cand The_400_Tasks/Task_by_Task/task034/prototypes/task034_redonly_probe.onnx` -> `[visible] 3/267 (1.12%)  FAIL`.
- why dead: the red pixel alone is ambiguous among the four possible 2x2 corner roles. The current background-channel veto is load-bearing for locating the actual non-background 2x2 square before painting the diagonal rays.
- one remaining hypothesis: a different lower-cost detector would still need an occupancy/background signal, but might compute it with a cheaper pooled veto rather than carrying the full 2-channel crop; do not retry red-only corner detection.
