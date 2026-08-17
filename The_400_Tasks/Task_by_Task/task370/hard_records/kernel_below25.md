# task370 hard record — kernel_below25

- live bar: 6204, SHA `d9643e949d751f6d3c5266908a35b400ce280ed185d96db71c1b3037ce9c5a94` (`python tools/ng_current_bar.py --task 370 --json`, manifest source, 2026-07-04T05:54:11Z).
- attempted route: resize the dynamic projection kernel below 25x25 from the current 31x31 graph, dropping out-of-radius taps by duplicating the previous valid tap.
- artifact: `The_400_Tasks/Task_by_Task/task370/prototypes/task370_kernel23_dropfar.onnx`.
- failed gate: `python tools/ng_gate.py visible --task 370 --cand The_400_Tasks/Task_by_Task/task370/prototypes/task370_kernel23_dropfar.onnx` -> `[visible] 262/266 (98.50%)  FAIL`.
- why dead: one visible miss is a certain hidden zero; kernels at or below this radius cannot preserve the visible projection support in this representation.
- one remaining hypothesis: 25x25 remains the smallest visible-clean radius in this drop-far family, but it is non-byte-equal and worse than live on capped truth.
