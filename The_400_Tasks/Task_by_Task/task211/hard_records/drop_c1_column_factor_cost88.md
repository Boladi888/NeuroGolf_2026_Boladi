# task211 — dead route: delete one column sign factor from the 92-byte Einsum

Live/team-best bar used: cost 92, sha256
`121c0bc0ef37cccef7edcab1b944fb042d3b77757eea76935d56772d66dc92de`
from `python tools/ng_current_bar.py --task 211 --json` (manifest authority,
owner Boladi, updated 2026-07-09T02:03:48Z).

Attempted route: remove/fuse the `C1` column sign factor from the terminal
free-output separable Einsum while keeping the six row factors plus reused `R0`
and `C0`; builder:
`The_400_Tasks/Task_by_Task/task211/prototypes/build_task211_drop_c1_probe.py`.

Candidate:
`The_400_Tasks/Task_by_Task/task211/prototypes/task211_drop_c1_cost88_probe.onnx`.

Charged cost proof: `score_network_v2` returned `(0, 0, 88)`; graph has one
terminal `Einsum`, `F[30,2]` plus seven `J[2,2]` initializers.

Failed gate: `python tools/ng_gate.py visible --task 211 --cand The_400_Tasks/Task_by_Task/task211/prototypes/task211_drop_c1_cost88_probe.onnx`
-> `[visible] 0/266 (0.00%)  FAIL`.

Why dead: the deleted `C1` is load-bearing. With only `R0*C0`, true columns for
`w=0,s=1/2` are negative instead of positive, so the output fails decode on every
official example. By the visible-fail axiom this is not a candidate; no truth/ab
gate was run.

One remaining hypothesis: a genuinely sign-only six-row-factor set outside the
exact-row-zero search could still share a different column factor and reach 88/84,
but the exact-row-zero and one-reuse searches remain walled at 92.
