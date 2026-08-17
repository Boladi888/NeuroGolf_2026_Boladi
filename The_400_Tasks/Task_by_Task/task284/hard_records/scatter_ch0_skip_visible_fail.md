# task284 hard record: sparse ScatterND without ch0 clears is dead

- Live bar: manifest/team-best cost `2501`, sha256 `f56d4cd53f0e4dc9ef579a1917b0855a1e50164583152c1718980e46b6f26a19`, source `python tools/ng_current_bar.py --task 284 --json` (`authoritative_source=manifest`, owner Vamshi, updated `2026-07-12T15:15:54Z`).
- Attempted route: halve the dominant sparse-renderer row count by deleting the 28 channel-0 zero writes and keeping only the 28 color-channel writes from the current `ScatterND` graph.
- Probe artifact: `The_400_Tasks/Task_by_Task/task284/prototypes/task284_skip_ch0_probe.onnx`, sha256 `54b315f69172ea2a8e2d6be6a0d915aabb3d4e4ad7b17b256d60cdbd46075a40`; static shape price `memory=1242`, `params=111`, `cost=1353`.
- Exact failed gate: `python tools/ng_gate.py visible --task 284 --cand The_400_Tasks/Task_by_Task/task284/prototypes/task284_skip_ch0_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: NeuroGolf compares the full one-hot output, including background channel `ch0=1` on blank active-grid cells and `ch0=0` on painted cells. If a color write does not also clear `ch0`, every newly painted background cell becomes multi-hot and visible fails completely.
- Remaining hypothesis: only a representation that clears `ch0` in the same charged structure as the color write can break the 56-row `ScatterND` floor; separate skip/decoder-shortcut variants are dead.
