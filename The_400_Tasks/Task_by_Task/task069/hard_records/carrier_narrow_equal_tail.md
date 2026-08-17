# task069 carrier-narrow / Equal-tail — DROP

- Authoritative live bar: `1097` static/profile, SHA-256 `e266f341fef0846ef8e35fa88c14a4e0576eff3d60cd99e1086ecf9170cdf29b`, from `python tools/ng_current_bar.py --task 69 --json` (manifest/workbook agree; local `10000` and predictor `1067` are stale).
- Live remeasure: `score_v2=(0, 1016, 81) cost=1097`; full real profile: `static=1097 profile=1097 (memory 1016 + params 81; 264 real inputs)`; visible `264/264`.
- Route attempted: reused the task-local one-channel scalar carrier + `Equal` + BOOL `Pad` fast probe at `../prototypes/task069_scalar_no_norm_bool_pad.onnx` (SHA-256 `c6d7aa6973017b755a50742c5b5766e8c088cafbc6ecf7ec454b2773996145bf`; `score_v2=(0, 7702, 49) cost=7751`, delta vs live `+6654`).
- Exact failed gate: `python tools/ng_gate.py visible --task 69 --cand The_400_Tasks/Task_by_Task/task069/prototypes/task069_scalar_no_norm_bool_pad.onnx --json` -> `correct=21 total=264 pct=7.954545454545454 pass=false`.
- Why dead: stop-on-visible-miss applies. Independently, the refreshed live graph has already removed the prompt's 10-channel/repeated-F32 carrier: it uses one F32 `[1,1,10,10]` direct-input read (400 B), one INT8 `[1,1,10,10]` carrier (100 B), and QLinearConv output reconstruction. A contract-safe Equal-tail alone forces either a 900 B full U8 canvas or a 1000 B 10x10 BOOL one-hot intermediate, already above the `~720` target before rule logic. No truth/A-B/profile/full/safety gate was run on the visible-zero probe.
- Historical safety warning: the prior Aditya `1087` formulation (SHA `74b93ac52f260ca1093a82a8d1cc791457196990e78368274cb049450faba7af`) was public-negative (`-15.24`); the current `1097` repair was later stack-confirmed. Do not epsilon-shave this hidden-clean graph without byte-equality proof.
- One remaining hypothesis: only a non-Equal, packed/fused direct-output rewrite that removes or spatially packs the unavoidable 400 B F32 direct-input Conv carrier could approach 720; direct QLinearConv is blocked by the F32 input ABI, so do not replay this carrier/Equal-tail family.

```json
{
  "verdict": "DROP",
  "candidate_class": "PATH_B_STRUCTURAL_PROBE",
  "cost": 7751,
  "profile_cost": null,
  "live_cost": 1097,
  "live_profile_cost": 1097,
  "cost_delta_vs_live": 6654,
  "status": "REJECT_LOCAL",
  "parent_source": "Team_Best_Onnx/task069.onnx",
  "parent_sha256": "e266f341fef0846ef8e35fa88c14a4e0576eff3d60cd99e1086ecf9170cdf29b",
  "candidate_source": "The_400_Tasks/Task_by_Task/task069/prototypes/task069_scalar_no_norm_bool_pad.onnx",
  "candidate_sha256": "c6d7aa6973017b755a50742c5b5766e8c088cafbc6ecf7ec454b2773996145bf",
  "manifest_path": null,
  "proof_commands_run": [
    "python tools/ng_current_bar.py --task 69 --json",
    "python tools/ng_gate.py visible --task 69 --cand Team_Best_Onnx/task069.onnx --json",
    "python tools/agent_helpers/profile_real_cost.py --task 69 --cand Team_Best_Onnx/task069.onnx --arcgen 266",
    "python tools/ng_gate.py visible --task 69 --cand The_400_Tasks/Task_by_Task/task069/prototypes/task069_scalar_no_norm_bool_pad.onnx --json",
    "score_network_v2(The_400_Tasks/Task_by_Task/task069/prototypes/task069_scalar_no_norm_bool_pad.onnx)"
  ],
  "proof_results": {
    "live_visible": "264/264 pass=true",
    "live_score_v2": "(0, 1016, 81) cost=1097",
    "live_profile": "static=1097 profile=1097 (memory 1016 + params 81; 264 real inputs)",
    "candidate_visible": "21/264 pct=7.954545454545454 pass=false",
    "candidate_score_v2": "(0, 7702, 49) cost=7751",
    "candidate_truth_ab_full": "not run: visible-zero stop rule"
  },
  "safety": null,
  "x86_required": false,
  "kaggle_probe_required": false
}
```
