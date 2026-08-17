# task324 — TopK2/no-marker-sum fallback dead route — 2026-07-15

## Live bar
- Authoritative bar: **7898** from `python tools/ng_current_bar.py --task 324 --json`.
- Owner/source: Sho, team manifest, updated `2026-07-09T01:09:17Z`.
- Parent artifact: `Team_Best_Onnx/task324.onnx`
- Parent SHA-256: `164004214e841c7cd4caaf061d82077801a0db71d88c54d621468a718d7edf9b`

## Attempted route
- Builder: `The_400_Tasks/Task_by_Task/task324/prototypes/build_task324_worker2_topk2_nofallback.py`
- Candidate: `The_400_Tasks/Task_by_Task/task324/prototypes/task324_worker2_topk2_nofallback.onnx`
- Candidate SHA-256: `80649089dfeb108f41e71d2b4ec747f826244823ea77034ef090ef8aa588afa7`
- Route: change Sho's TopK(4) color block to TopK(2), keeping only the two dominant background colors, and delete the marker-color-sum fallback by wiring direct row/column class reductions as `pa1`/`pb1`.

## Failed gate
- Command: `python tools/ng_gate.py visible --task 324 --cand The_400_Tasks/Task_by_Task/task324/prototypes/task324_worker2_topk2_nofallback.onnx`
- Result: `[visible] 10/266 (3.76%)  FAIL`

## Why this is dead
The deleted TopK(4) marker-color sum is load-bearing. When both horizontal and vertical stripe families are present, a base-class marker can be invisible to the direct row/column masked reductions because its row and column both contain the other background color. Sho's fallback uses the two marker colors from TopK(4) (`d1p + d2p`) to recover that hidden class. Without it, visible correctness collapses, so this representation family is a certain hidden 0.

Per gate rules, no score/profile/truth/A-B/safety was run after the visible miss.

## One remaining hypothesis
Only a different way to recover the distinct two-marker-color sum below the current TopK(4) block's tiny ~100 B footprint could help. Runtime recovery from `rare` needs a nonzero mask plus min/max or a small TopK over the 20x20 marker plane, which appears more expensive than Sho's existing GlobalAveragePool+TopK(4).
