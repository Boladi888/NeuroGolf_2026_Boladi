# DEAD ROUTE - task196 mechanical param shaves after binpaint_k2

Recorded by `bo_ng_cdx_worker3` for task196.

## Live bar

- Bar source: `python tools/ng_current_bar.py --task 196 --json`
- live_cost: **3167**
- live/team-best path: `Team_Best_Onnx/task196.onnx`
- live/team-best sha256: `e6cbb4d49482e5a4712ade0076a9059ea273598306b62d7a8857ef2ea956745c`
- owner/source: Boladi manifest row, updated 2026-07-14T00:05:59Z

## Attempted route

Mechanical byte-equivalent shaves on the current `binpaint_k2` graph:

1. Convert QLinearConv length-1 scale/zero-point initializers from shape `[1]` to scalar shape `[]`.
2. Run the standard `tools/polish_candidate.py` pipeline on the current best.
3. Remove the optional detector bias `bdet[2]` without refitting the detector.

Artifacts:

- `The_400_Tasks/Task_by_Task/task196/prototypes/build_task196_mechanical_probes.py`
- `The_400_Tasks/Task_by_Task/task196/prototypes/task196_scalar_quant_params.onnx`
- `The_400_Tasks/Task_by_Task/task196/prototypes/task196_team_best_polished.onnx`
- `The_400_Tasks/Task_by_Task/task196/prototypes/task196_omit_detector_bias.onnx`

## Results

- Scalar scale/zp probe: `score_network_v2` returned `(0, 3038, 129)` = **3167**, so scalar initializer shapes do **not** save under the actual scorer. Note: `ng_graph_cost_diff.py`/`cost_predictor` reports a false local delta of `-3` for this exact graph; do not use that predictor result as the bar.
- Polish probe: `tools/polish_candidate.py` wrote a byte-identical/cost-identical graph, final_cost **3167**, no applied saving.
- Bias omission probe: `score_network_v2` returned `(0, 3038, 127)` = **3165**, and `ng_graph_cost_diff.py` named the only charged delta as removed initializer `bdet [int32 [2]]` = `-2 params`.

Exact failed gate:

```text
python tools/ng_gate.py visible --task 196 --cand The_400_Tasks/Task_by_Task/task196/prototypes/task196_omit_detector_bias.onnx
[visible] 2/266 (0.75%)  FAIL
```

## Verdict

`task196_omit_detector_bias.onnx` is **REJECT_LOCAL**. One visible miss kills this representation family; the detector bias is load-bearing and no truth/AB/profile gates should be run on the bias-omitted graph.

The K=1 detector merge is also not a candidate route from the existing task-local probe:

```text
[K=1] single filter all sizes: fp=1140 fn=453  NOT separable
```

## Remaining hypothesis

The only plausible remaining floor move is not a mechanical param shave: find a new exact tail that deletes the 225-byte `Min`/`ring` plane without needing a 9-channel size-specific stamp, or a final-output construction that avoids the 900-byte `cond` mask while still copying `input` exactly. Current evidence says the existing K=2 detector plus `paint ∩ ch1` tail is near the local structural floor.
