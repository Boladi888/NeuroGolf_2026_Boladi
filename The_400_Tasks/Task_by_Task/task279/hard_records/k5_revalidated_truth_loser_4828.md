# task279 hard record: K5 regrow revalidation (4828 truth loser)

Live bar: manifest-authoritative Sho cost **4,875**, sha256
`e281359d428f0b084b6f609c90bdd2996e814526dd112bd15de8d911ead7230c`
from `python tools/ng_current_bar.py --task 279 --json`.

Attempted route: revalidate the existing fitted K5 regrow candidate
`prototypes/task279_k5.onnx`, which shrinks Sho's 2-channel recover kernel
from `recover_w [1,2,7,7]` to `[1,2,5,5]` and adds `recover_wzp`.

Cost result: `_score_v2` and `profile_real_cost --arcgen 266` both report
**4,828** (`memory=4740`, `params=88`), saving **47 bytes** versus the
authoritative 4,875 live bar. `ng_graph_cost_diff` names the charged-structure
change as the recover weight table remap; its local predictor total is stale
(`4871 -> 4823`), but the structure delta is still the same kernel shrink.

Exact gates run:
- `python tools/ng_gate.py visible --task 279 --cand The_400_Tasks/Task_by_Task/task279/prototypes/task279_k5.onnx`
  -> `[visible] 266/266 (100.00%)  PASS`
- `python tools/ng_gate.py truth --task 279 --cand Team_Best_Onnx/task279.onnx --n 2000`
  -> `[truth] vs ARC-GEN ground truth: 1994/2000 (99.700%)`
- `python tools/ng_gate.py truth --task 279 --cand The_400_Tasks/Task_by_Task/task279/prototypes/task279_k5.onnx --n 2000`
  -> `[truth] vs ARC-GEN ground truth: 1993/2000 (99.650%)`
- `python tools/ng_gate.py ab --task 279 --cand The_400_Tasks/Task_by_Task/task279/prototypes/task279_k5.onnx --n 2000`
  -> `[ab] vs submission.zip: match 2259/2266 | divergences=7 | byte_equal=False (first_div=1104)`

Why dead: LIVE is itself below 100% truth, so this is an ambiguous/fitted class.
A cheaper non-byte-equal candidate must be byte-equal to LIVE or more-correct
than LIVE on the same truth window. K5 is neither: it has 7 A/B divergences and
one more truth miss than LIVE. Stop before `--n 8000` and before `ng_safety`;
this is a local rejected probe, not a staged candidate.

One remaining hypothesis: only a different non-K5 formulation that keeps the
same 4,740 memory floor while improving or matching LIVE correctness can be a
probe. A cost tie with a better K7 fit is a correctness experiment, not a
task279 cost win under the current staging rules.
