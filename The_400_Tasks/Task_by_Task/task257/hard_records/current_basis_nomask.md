# task257 hard record - current-basis no-mask probe

Live bar: 114, sha256 `32e5452b9089ab217e13dc4aac064b7807f9397603a66e5a5a945ffa3b0f5ef6`,
from `python tools/ng_current_bar.py --task 257 --json` (`manifest`, owner Vamshi,
updated 2026-07-12T15:15:54Z).

Attempted route: start from `Team_Best_Onnx/task257.onnx` and delete the two
terminal `mask` operands plus the `mask[30]` initializer, leaving the same one-node
free-output `Einsum` with `feat[2,30]`, `color[2,10]`, and `proj_a[2,2]`.
Prototype: `The_400_Tasks/Task_by_Task/task257/prototypes/task257_nomask_probe.onnx`.

Static score: `(macs=0, memory=0, params=84)`, cost 84. Named delta from
`ng_graph_cost_diff`: removed `mask` float32 `[30]`, params 114 -> 84, memory
unchanged 0, net -30.

Exact failed gate: `python tools/ng_gate.py visible --task 257 --cand The_400_Tasks/Task_by_Task/task257/prototypes/task257_nomask_probe.onnx`
returned `[visible] 0/269 (0.00%)  FAIL`.

Why dead: the current shared `feat[2,30]` basis deliberately aliases rows/cols
`0..3` with `5..8` so both halves of the input cross map to the same 4x4 answer
coordinates. Without the explicit output `mask[30]`, that aliasing leaks decoded
positives into rows/cols outside the answer window. Per visible-fail axiom, do
not run truth/ab/battery/profile on this family.

One remaining hypothesis: a different role-aware projective basis might absorb
the output-window mask into the final `Einsum`, but it must still distinguish
input positions `5..8` from output positions `5..8`; expect extra 2x2 factors or
a separate role carrier, so price it against the 30-param mask before building.
