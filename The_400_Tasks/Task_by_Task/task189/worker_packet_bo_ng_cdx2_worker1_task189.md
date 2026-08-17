# task189 worker packet - bo_ng_cdx2_worker1

## Bar
- authoritative bar used: 521 bytes, from `python tools/ng_current_bar.py --task 189 --json`
- live/team-best SHA-256: `fe2ea6bc155e42c00f364540ebb1100e650d45b2a2663dbc8dd97bc2a18937bd`
- owner/update: Sho, `2026-07-09T01:09:17Z`
- live profile line: `task189 task189.onnx: static=521 profile=521 (memory 458 + params 63; 266 real inputs)`

## Candidate
- candidate paths: none staged
- candidate_class: `NONE_NO_STAGE`
- parent_source: `Team_Best_Onnx/task189.onnx`
- parent_sha256: `fe2ea6bc155e42c00f364540ebb1100e650d45b2a2663dbc8dd97bc2a18937bd`
- candidate_sha256: null
- cost/profile_cost: null
- cost delta vs live: null
- status: `REJECT_LOCAL`
- x86_required: false
- kaggle_probe_required: false
- safety: null, not run because no candidate beat the 521-byte bar

## Proof Commands Run
- `python tools/ng_current_bar.py --task 189 --json` -> authoritative manifest bar 521, SHA `fe2ea6bc155e42c00f364540ebb1100e650d45b2a2663dbc8dd97bc2a18937bd`
- `python tools/agent_helpers/inspect_graph.py Team_Best_Onnx/task189.onnx` -> 25 nodes, output `UINT8 [1,10,30,30]`, memory-dominant `pat_f` 144 B and `features` 72 B
- `python tools/agent_helpers/largest_intermediate.py Team_Best_Onnx/task189.onnx --top 20` -> total static intermediates 458 B
- `python tools/agent_helpers/profile_real_cost.py --task 189 --cand Team_Best_Onnx/task189.onnx --arcgen 266` -> static/profile 521
- `python The_400_Tasks/Task_by_Task/task189/prototypes/bo_ng_cdx2_worker1_probe_variants.py` -> all zero-point omission variants checker-rejected

## What Failed
- QLinearConv zero-point omission: checker rejects empty `x_zero_point` and `w_zero_point` inputs as required single inputs in this schema. Hard record: `hard_records/qlinear_zero_point_omission.md`.
- Biasless 2-feature QLinearConv classifier: existing Sho code points are not separable under fixed shared thresholds; bounded alternate mask probes also found no separable 10-class codebook. Hard record: `hard_records/qlinear_biasless_codebook.md`.
- Direct raw-input Conv idea was priced out before build: dense kernel params swamp the scalar-memory saved by deleting dynamic Slice/gates.

## Remaining Hypothesis
A true sub-521 route likely needs a different final-output primitive that avoids QLinearConv's required zero points and per-channel bias while keeping the two-feature color-code carrier. Three-plus feature masks are only worth revisiting with a solver because the extra feature plane cost can exceed the 10 params saved.
