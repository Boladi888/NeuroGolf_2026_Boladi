# task387 worker packet - bo_ng_cdx2_worker3

Date: 2026-06-30

## Bar used

Current team-best bar: cost 3516, points 16.8349.

Source: `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`
row `task387`, creator `Aditya`, SHA-256
`312be9bfde06a386ad53304ea06b76932a93f586485a267072015ce97ff7dfaa`,
updated `2026-06-14T17:43:29Z`.

Cross-checks:
- Prompt target: "beat team-best ~= 3516".
- `python tools/agent_helpers/show_task.py --task 387`: `current_cost: 3516`, `current_points: 16.8349`.
- BoBrain provider: `team_manifest_cost: 3516`, `source_authority: team_manifest_csv`.

Expected delta for any reviewer-ready candidate must be strictly negative vs 3516.
No tie is stageable.

## Reviewer-ready candidates

None. No artifact beat cost/profile cost 3516 while passing gates, so nothing was staged.

## Probe artifacts

| path | static cost | profile cost | SHA-256 | verdict |
|---|---:|---:|---|---|
| `The_400_Tasks/Task_by_Task/task387/prototypes/task387_team_best_3516.onnx` | 3516 `(mem 3466 + params 50)` | 3516 | `312be9bfde06a386ad53304ea06b76932a93f586485a267072015ce97ff7dfaa` | baseline copy; valid but ties bar |
| `The_400_Tasks/Task_by_Task/task387/prototypes/task387_team_best_3516_onnxsim.onnx` | 3516 `(mem 3466 + params 50)` | not run | `312be9bfde06a386ad53304ea06b76932a93f586485a267072015ce97ff7dfaa` | onnxsim no-op; tie |
| `The_400_Tasks/Task_by_Task/task387/prototypes/task387_team_best_3516_polished.onnx` | 3516 `(mem 3466 + params 50)` | not run | `312be9bfde06a386ad53304ea06b76932a93f586485a267072015ce97ff7dfaa` | polish no-op; tie |
| `The_400_Tasks/Task_by_Task/task387/prototypes/task387_probe_no_active_sentinel.onnx` | 2837 `(mem 2788 + params 49)` | not run | `8364467c22fcc7ee977b19655429bc439d5d907b8f1b9b2fbdf52c6902f7258d` | invalid floor probe; visible fails |
| `The_400_Tasks/Task_by_Task/task387/prototypes/probe_scatternd_i32_support.onnx` | n/a | n/a | `5a6496ef3e7b87c649f0775188694f5420012333a22f05b79955e65058305d11` | ORT support probe, not a task candidate |

Manifest path for this worker packet:
`The_400_Tasks/Task_by_Task/task387/worker_packet_bo_ng_cdx2_worker3_task387.md`.

## Gates run

Baseline visible:
`[visible] 266/266 (100.00%)  PASS`

Baseline profile:
`task387 task387_team_best_3516.onnx: static=3516 profile=3516 (memory 3466 + params 50; 266 real inputs)`

Baseline battery:
```
=== battery task387: PASS ===
[visible] 266/266 (100.00%)  PASS
[truth] vs ARC-GEN ground truth: 2000/2000 (100.000%)
[ab] vs submission.zip: match 2266/2266 | divergences=0 | byte_equal=True (first_div=None)
[score] cost=3516 (mem 3466+par 50)  points=16.835
```

Invalid no-sentinel visible:
`[visible] 9/266 (3.38%)  FAIL`

Scores:
```
task387_team_best_3516.onnx: (0, 3466, 50) cost=3516
task387_team_best_3516_onnxsim.onnx: (0, 3466, 50) cost=3516
task387_team_best_3516_polished.onnx: (0, 3466, 50) cost=3516
task387_probe_no_active_sentinel.onnx: (0, 2788, 49) cost=2837
```

INT32 ScatterND support probe:
```
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph:
Type Error: Type 'tensor(int32)' of input parameter (indices) of operator (ScatterND) in node (scatternd_i32) is invalid.
```

## Tried and failed

- `onnxsim` on the 3516 team-best graph: no graph or credited-cost change. SHA stayed identical.
- `tools/polish_candidate.py` on the 3516 team-best graph: reported `original cost: 3516`, `polished cost: 3516  (-0 bytes)`, no-op patterns.
- INT32 `ScatterND`: checker/shape-inference wrote the probe, but ORT 1.24 load rejected INT32 indices. The documented sub-3.3k ScatterND path is still blocked here.
- No-active-sentinel floor probe: replacing the active-grid sentinel base with all-zero `ConstantOfShape` drops static cost to 2837 but fails visible 9/266. This reconfirms that padding/background disambiguation is load-bearing.

## Remaining floor hypothesis

The valid 3516 graph is already the crop18 + `ScatterElements(reduction=max)` version of the color-grid rewrite. Its largest costs are still the U8 `color_grid_padded [30,30]` 900B, the four 18x18 U8 color-grid tensors, and the 52-cell INT32 scatter index/update path.

The only measured sub-bar local floor this pass was the invalid 2837 no-sentinel probe. A valid sub-3516 candidate needs either:
- a new way to preserve active-grid padding semantics for less than the current sentinel-base chain;
- ORT/scorer support for INT32 `ScatterND` indices, which is currently false;
- or a wholly different direct-output formula that avoids the color grid without materializing multiple 30x30 BOOL masks.

No lesson packet was created.
