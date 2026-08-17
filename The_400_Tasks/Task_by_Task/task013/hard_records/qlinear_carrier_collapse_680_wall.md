# task013 hard record - qlinear carrier collapse at 680

## Bar
- Authoritative live/team-best bar: 680 = memory 565 + params 115.
- Source: `python tools/ng_current_bar.py --task 13 --json`, authoritative source `manifest`.
- Owner/update: Boladi, 2026-07-10T22:11:58Z.
- Live SHA-256: `64ccf42b47a7d774932f89238c048b6bfc1f5ad2f93e81da13e9b936de7406d5`.
- Note: `Team_Best_Onnx/task013.onnx` has the same SHA but stale local/tool cost displays may show 679. The manifest/workbook bar is 680 and is the bar used here.

## Attempted Route
Audited the prompt's QLinear/U8 carrier-collapse angle against the current 680 scalar-moment graph.
The graph already carries the periodic stripe line as U8/I8 until the final output writer:

- `line_feat_u8` UINT8 `[2,30]` = 60 B.
- `line_feat` FLOAT `[2,30]` = 240 B, produced only because the final output `Einsum` must also consume the raw F32 `input` as the active-grid mask.
- The F32 `input` operand is load-bearing: it gates padding for free inside the graph-output op. Replacing it with an explicit active mask would cost at least the saving from deleting `line_feat`.

Probe builder: `The_400_Tasks/Task_by_Task/task013/prototypes/bo_ng_cdx2_worker2_probe_680.py`.

## Failed Gates
Foreground score/probe run:

```text
python The_400_Tasks/Task_by_Task/task013/prototypes/bo_ng_cdx2_worker2_probe_680.py
task013_cdx2w2_baseline_680.onnx: score=(0, 565, 115) total=680
task013_cdx2w2_probe_u8_final_680.onnx: score=None total=None
task013_cdx2w2_probe_f16_line_680.onnx: score=None total=None
task013_cdx2w2_probe_i8_linecode_680.onnx: score=None total=None
task013_cdx2w2_probe_lineids_from_coord_680.onnx: score=(0, 595, 85) total=680
```

Tie cost diff for the only valid mutation:

```text
python tools/ng_graph_cost_diff.py --task 13 --live Team_Best_Onnx/task013.onnx --candidate The_400_Tasks/Task_by_Task/task013/prototypes/task013_cdx2w2_probe_lineids_from_coord_680.onnx
delta memory +30, params -30, net cost delta 0 -> tie
```

Baseline safety checks rerun this pass:

```text
python tools/agent_helpers/profile_real_cost.py --task 13 --cand Team_Best_Onnx/task013.onnx --arcgen 266
task013 task013.onnx: static=680 profile=680 (memory 565 + params 115; 267 real inputs)

python tools/ng_gate.py visible --task 13 --cand Team_Best_Onnx/task013.onnx
[visible] 267/267 (100.00%)  PASS
```

No `truth`, `ab`, battery, or `ng_safety` was run for the probes because no probe strictly beat the 680 live/profile bar.

## Why This Route Is Dead
- `QLinearConv` and U8 final-carrier routes cannot consume the raw F32 graph input. Casting the full one-hot input to U8 costs far more than the 240 B `line_feat` target.
- Feeding `line_feat_u8` or F16 `line_feat` into the final `Einsum` is scorer-dead because this output writer is homogeneous and also consumes F32 `input`.
- The INT8 line-code constant-sharing variant is scorer-dead because `Where` with INT8 data is not supported on the pinned ORT CPU path.
- Replacing the stored INT8 arange initializer with `Cast(coord_f -> INT8)` only trades 30 params for a 30 B intermediate; it is a valid no-gain tie.

## One Remaining Hypothesis
A new representation could still beat 680 only if it folds active-grid gating and channel comparison into one graph-output op without materializing a full active mask and without requiring F32 `line_feat`. Treat that as a fresh output-writer family, not a continuation of the QLinear/U8 dtype-collapse route.
