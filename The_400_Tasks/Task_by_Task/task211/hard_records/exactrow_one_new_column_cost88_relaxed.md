# task211 — dead route: exact-row one-new-column cost88 with relaxed checked-width assumption

Live/team-best bar used: cost 92, sha256
`121c0bc0ef37cccef7edcab1b944fb042d3b77757eea76935d56772d66dc92de`
from `python tools/ng_current_bar.py --task 211 --json` (manifest authority,
owner Boladi, updated 2026-07-09T02:03:48Z).

Attempted route: keep six exact row-zero bilinear factors, reuse one row factor
as a column factor, and replace the two live new column factors with one new
column factor. Search/builder:

- `The_400_Tasks/Task_by_Task/task211/prototypes/search_exactrow_one_new_col_relaxed.py`
- `The_400_Tasks/Task_by_Task/task211/prototypes/build_task211_exactrow_one_new_col88.py`

Candidate:
`The_400_Tasks/Task_by_Task/task211/prototypes/task211_exactrow_one_new_col88.onnx`
sha256 `e0bde616a5ca1f54faca16c383aeb271308da89d009a2555850142ac633efb99`.

Charged cost proof:
`python -c "from tools._score_v2 import score_network_v2; p=r'The_400_Tasks/Task_by_Task/task211/prototypes/task211_exactrow_one_new_col88.onnx'; print(score_network_v2(p))"`
-> `(0, 0, 88)`.

Named static saving:
`python tools/ng_graph_cost_diff.py --task 211 --live Team_Best_Onnx/task211.onnx --candidate The_400_Tasks/Task_by_Task/task211/prototypes/task211_exactrow_one_new_col88.onnx`
-> params delta `-4`, memory delta `+0`, net cost delta `-4`; charged structure:
removed initializer `C1` (`float32 [2,2]`, -4 params).

Failed gate:

- First visible attempt with shell timeout 120s:
  `python tools/ng_gate.py visible --task 211 --cand The_400_Tasks/Task_by_Task/task211/prototypes/task211_exactrow_one_new_col88.onnx`
  -> shell exit 124 around 124s, inconclusive.
- Rerun in the foreground with longer timeout:
  same command -> `[visible] 0/266 (0.00%)  FAIL`.

Safety check:
`python tools/ng_safety.py model --task 211 --cand The_400_Tasks/Task_by_Task/task211/prototypes/task211_exactrow_one_new_col88.onnx --json`
-> `verdict: PASS`, `n_failures: 0`, `n_warnings: 0`; correctness rejection
still stands because visible is 0/266.

Why dead: the relaxed search enforced column correctness only for `s=0..3`.
The 9x4 decoded block is correct on a sample, but the graph emits positive
channels in padded output columns `s=4..8` (45 outside-window positives on the
first JSON example). `ng_gate visible` compares the full padded one-hot tensor,
so these columns are load-bearing and this relaxed representation is a certain
hidden 0. Do not run truth/AB/profile on this visible-zero family.

Additional note: `python search_exactrow_one_new_col_relaxed.py --full-width`
was started to require non-positive columns `s=4..8`, but hit the 604s shell cap
without a result. That timeout is not a proof of impossibility; it only shows
the relaxed repair search needs a more optimized formulation.

One remaining hypothesis: a full-width one-new-column completion with multiple
reused row factors, or a genuinely sign-only row set outside exact row-zero
factors, could still reach 88, but it must suppress `s=4..8` under full-tensor
visible comparison before any truth/AB work.
