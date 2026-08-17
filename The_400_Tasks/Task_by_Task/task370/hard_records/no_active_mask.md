# task370 hard record — no_active_mask

- live bar: 6204, SHA `d9643e949d751f6d3c5266908a35b400ce280ed185d96db71c1b3037ce9c5a94` (`python tools/ng_current_bar.py --task 370 --json`, manifest source, 2026-07-04T05:54:11Z).
- attempted route: pad `dzmp_proj_b` directly to 30x30 and delete `dzmp_active_rect_b20` / `dzmp_paint_b20` with DCE.
- artifact: `The_400_Tasks/Task_by_Task/task370/prototypes/task370_no_active_mask_probe.onnx`; profile cost 5296.
- failed gate: `python tools/ng_gate.py visible --task 370 --cand The_400_Tasks/Task_by_Task/task370/prototypes/task370_no_active_mask_probe.onnx` -> `[visible] 59/266 (22.18%)  FAIL`.
- why dead: the active rectangle is load-bearing; direct projection paints outside the active grid/padding on visible officials.
- one remaining hypothesis: only a different active-mask construction that still suppresses padding can help; deleting active gating outright is not viable.
