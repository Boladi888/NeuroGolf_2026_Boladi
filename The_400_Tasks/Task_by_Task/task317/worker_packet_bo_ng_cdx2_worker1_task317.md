# task317 worker packet - bo_ng_cdx2_worker1

## Verdict

No staged candidate. The only built sub-110 probe was a tied-channel-vector one-node `Einsum` at static cost 100, but it failed visible `0/265`, so status is `REJECT_LOCAL`.

## Current Bar

- authoritative command: `python tools/ng_current_bar.py --task 317 --json`
- source: team manifest
- live/team-best cost: 110
- live/team-best owner: Sho
- live/team-best updated UTC: 2026-07-09T01:09:17Z
- parent path: `Team_Best_Onnx/task317.onnx`
- parent sha256: `ebd09483ac75c93d2857c62e54dab9d3471c849153525d93b9a7fc725933038a`

## Candidate

- candidate path: `The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx`
- builder: `The_400_Tasks/Task_by_Task/task317/prototypes/build_task317_tied_channel_probe.py`
- candidate_class: `PATH_B_REPARAM_PROBE_VISIBLE_FAIL`
- parent_source: `Team_Best_Onnx/task317.onnx`
- parent_sha256: `ebd09483ac75c93d2857c62e54dab9d3471c849153525d93b9a7fc725933038a`
- candidate_sha256: `57dcc76ce0aed3c48f794d0010c7920d85f822c67fc69fa8857212f197d2fcf3`
- candidate_cost_static: 100
- candidate_cost_profile: not run, because visible failed
- cost delta vs live: -10
- x86_required: false
- kaggle_probe_required: false
- status: `REJECT_LOCAL`

## Proof Commands Run

```text
python tools/agent_helpers/context_pack.py --task 317
python tools/ng_current_bar.py --task 317 --json
python Onnx_Advanced/library_db/lib_query.py status
python Onnx_Advanced/library_db/lib_query.py context task317 --limit 8
python Onnx_Advanced/library_db/lib_query.py search "task317 PostMergeRuleRebuild" --limit 8
python -B -m neurogolf_provider.cli context task317
python -B -m neurogolf_provider.cli consistency task317
python -B -m neurogolf_provider.cli search "task317 PostMergeRuleRebuild"
python The_400_Tasks/Task_by_Task/task317/prototypes/build_task317_tied_channel_probe.py
python tools/ng_gate.py visible --task 317 --cand The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx
python tools/ng_graph_cost_diff.py --task 317 --live Team_Best_Onnx/task317.onnx --candidate The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx --json
python tools/ng_safety.py model --task 317 --cand The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx --json
```

## Proof Results

```text
RULE_PROBE task317 fresh_arcgen 3000/3000 PASS
score_v2(candidate): (0, 0, 100)
visible(candidate): [visible] 0/265 (0.00%)  FAIL
graph_cost_diff: live_cost=110, cand_cost=100, cost_delta=-10
graph_cost_diff named delta: removed pvec[10], removed tvec[10], added v[10]; reconciled params_delta=-10, memory_delta=0
```

No truth, AB, profile, battery, or staging was run after the visible miss.

## Safety

```json
{
  "kind": "model",
  "source": "The_400_Tasks\\Task_by_Task\\task317\\prototypes\\cand_task317_tied_channel_cost100.onnx",
  "verdict": "PASS",
  "summary": {
    "n_failures": 0,
    "n_warnings": 0,
    "runtime_integer_denominator_count": 0
  },
  "model": {
    "task": "task317",
    "name": "cand_task317_tied_channel_cost100.onnx",
    "source": "The_400_Tasks\\Task_by_Task\\task317\\prototypes\\cand_task317_tied_channel_cost100.onnx",
    "sha256": "57dcc76ce0aed3c48f794d0010c7920d85f822c67fc69fa8857212f197d2fcf3",
    "size_bytes": 618,
    "failures": [],
    "warnings": [],
    "op_inventory": {
      "Einsum": 1
    },
    "risk_op_flags": [],
    "runtime_integer_denominators": [],
    "runtime_denominators_unknown_dtype": [],
    "official_io_contract": {
      "data_inputs": [
        "input"
      ],
      "outputs": [
        "output"
      ],
      "ok": true
    },
    "load_after_sanitize": "ok",
    "verdict": "PASS"
  }
}
```

## What Failed

The shared-vector idea cannot distinguish input-channel role from output-channel role. To detect gray blocks, the reused vector needs a negative coefficient on channel 5. Reusing that same vector as the output-channel sign vector makes output channel 5 positive on gray/blue blocks too, causing multi-hot cells and visible `0/265`.

## Remaining Hypothesis

The live 110 graph is already a zero-memory, single-output `Einsum` with params `M[3,30]` + `tvec[10]` + `pvec[10]`. A sub-110 candidate needs a different legal way to eliminate part of `M[3,30]` or the output sign mask without introducing any charged intermediate; channel-vector tying is dead.

## Lesson / Hard Record

- hard record: `The_400_Tasks/Task_by_Task/task317/hard_records/tied_channel_vector_visible_zero.md`
