# Validation And Return Contract

## If You Have The Local Repo

Use these gates for a candidate `C.onnx`:

```powershell
python tools/ng_gate.py visible --task 374 --cand C.onnx
python tools/ng_gate.py truth --task 374 --cand C.onnx --full
python tools/ng_gate.py ab --task 374 --cand C.onnx --full
python tools/ng_gate.py battery --task 374 --cand C.onnx
python tools/agent_helpers/profile_real_cost.py --task 374 --cand C.onnx --arcgen 266
python -c "from tools._score_v2 import score_network_v2; p=r'C.onnx'; r=score_network_v2(p); print(r, None if r is None else r[1]+r[2])"
```

Visible pass is only smoke. If the candidate is not byte-equal to a known-good
graph, require strong generated truth or adversarial evidence.

## If You Are Web-Only

Return an implementation-ready packet:

```text
task: task374
decision: ready | probe | discard
candidate_name:
expected_cost:
expected_memory:
expected_params:
baseline_used: 1365 staged qscale-share
core_idea:
formulation_families_considered:
chosen_graph_ops:
charged_tensors_expected:
params_expected:
why_it_should_be_correct:
why_it_should_generalize:
known_risks:
first_local_command_to_run:
files_returned:
```

For `ready`, include an ONNX file or builder script if possible. For `probe`,
include enough graph-construction detail that a local worker can build it
without guessing. For `discard`, include the next creative route that is still
worth trying.
