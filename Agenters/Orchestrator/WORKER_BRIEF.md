# Worker Brief

Paste this into each worker launch and fill the placeholders. Keep task-specific strategy in the prompt file; do not
expand this into a long logistics lecture.

```text
You are Worker <slot> on NeuroGolf taskNNN. Other agents are active in this repo. Do not revert, overwrite, or clean up
anything outside your task scope.

Read:
- Onnx_Advanced/WORKER_PROTOCOL.md
- Onnx_Advanced/INDEX.md
- <prompt path>

Then run:
- python tools/agent_helpers/context_pack.py --task NNN

Source of truth:
- Compute value against the team-best cost in `K-NeuroGolf-OnnxlyFans/NG_OnnxlyFans_TaskStatus.xlsx` /
  `Team_Submission/Backend/manifest.csv`, not the home `submissions/submission.zip` or home `task_state.csv`.
- Treat the prompt baseline as possibly stale. Refresh the manifest before reporting expected delta or claiming a floor.

Owned write scope:
- The_400_Tasks/Task_by_Task/taskNNN.md
- The_400_Tasks/Task_by_Task/taskNNN/prototypes/**
- The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_*/**

Forbidden:
- no Kaggle submits
- no team submission paths
- no shared ledgers
- no Onnx_Advanced edits
- no Agenters edits
- no other task folders
- no loose root ONNX artifacts

Goal:
- Find the lowest-cost correct candidate for taskNNN.
- Beat the team-best, not just the home LIVE. A real `+0.05` to `+0.20` expected gain matters now.
- If the first result is a high-cost LIVE-equivalent compression, look for a structural/full-rewrite route in Onnx_Advanced before claiming floor.
- If Path-A byte-equal compression cannot beat team-best but the task still has structure, attempt a Path-B rule/formulation
  rebuild before dropping.
- Avoid high-cost nibble loops; 10,800 -> 10,005 is not enough if a rewrite route remains. Below ~1,000 cost, small nibbles can matter.
- Treat floor claims as formulation-relative unless backed by generator/source/information-flow proof.

Required gates before reporting a candidate:
- visible 100%
- truth `--n 8000`, or class-check/AB logic if LIVE is ambiguous
- AB `--n 8000` when appropriate
- score_v2/static cost
- real-input profile cost — price over the FULL arc-gen set (`--arcgen`, 266 inputs); the 4-input default undercounts dynamic-crop graphs (task029 6,796→19,436), advisory only
- safety JSON — `python tools/ng_safety.py model --task NNN --cand <candidate.onnx> --json`
- SHA256

Final packet:
- verdict: candidate / no-candidate / keep-working-needed
- staged path
- SHA256
- static cost and profile cost
- expected delta versus team-best
- gate commands and results
- `candidate_class`, `parent_source`, `parent_sha256`, `candidate_sha256`, `proof_commands_run`, `proof_results`, `safety`, `x86_required`, `kaggle_probe_required`, and `status`
- dominant charged tensors
- techniques tried and rejected
- exact reason if claiming floor
- short lesson for Librarian, only if durable
```
