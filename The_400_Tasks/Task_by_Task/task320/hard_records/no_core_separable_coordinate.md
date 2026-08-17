# task320 — no-core separable-coordinate probe DEAD

Live bar used: 88 bytes, SHA256
`3b60b3cd73e4985df112d876a12b57ad7dbc9cd62e7ed87b71d9e2b22ee1c016`,
source `python tools/ng_current_bar.py --task 320 --json` / manifest
(owner Vamshi, updated 2026-07-02T02:35:15Z).

Attempted route: delete the current one-node output-Einsum's dense `core`
initializer `[2,2,2]` and replace the bilinear coordinate contraction

`birc,bjsc,ti,tj,to,pr,qs,tpq->borc`

with the cheapest separable t-specific coordinate product

`birc,bjsc,ti,tj,to,tr,ts->borc`.

Prototype:
`The_400_Tasks/Task_by_Task/task320/prototypes/task320_drop_core_separable_probe.onnx`

Static score:
`score_network_v2 (0, 0, 80)` → cost 80 = memory 0 + params 80.
SHA256:
`8db5c2be5fddcc6d5e59fbc0baebf91f42561c4b07f659441279fd956f318af6`.

Named charged delta:
`python tools/ng_graph_cost_diff.py --task 320 --live Team_Best_Onnx/task320.onnx --candidate The_400_Tasks/Task_by_Task/task320/prototypes/task320_drop_core_separable_probe.onnx`
reported `core` removed, params delta `-8`, net cost delta `-8`.

Failed gate:
`python tools/ng_gate.py visible --task 320 --cand The_400_Tasks/Task_by_Task/task320/prototypes/task320_drop_core_separable_probe.onnx`
→ `[visible] 0/266 (0.00%)  FAIL`.

Why dead: the removed `core` is not just a priced nuisance. It carries the
three required coordinate monomials: identity baseline, `r * count(column)`,
and `-sum(row_index)` for the bottom-half threshold. A two-row separable
coordinate product can keep the cost at 80, but it cannot express the live
antisymmetric `r - s` threshold while also preserving the background/source
identity channel split. A visible miss is a certain hidden 0, so no truth,
AB, battery, or profile gate is valid for this family.

One remaining hypothesis: the only plausible sub-88 path is a different dense
one-Einsum channel/coordinate factorization that keeps the antisymmetric
row-threshold information without a charged `core`; sparse initializers and
no-core separable coordinate products are both dead.
