# task279 hard record: topology_cycle_probe

Live/team bar: manifest/workbook Vamshi team-best cost 5,057, sha256
`4f9a0651cbb5bf227feb6d794dab4e84df585bd83cdfb7fa1a7e2928c70574f9`.

Attempted route: replace Vamshi's 4-erode + 2-jump cycle seed with a direct
topology/cycle classifier. The exact Python oracle labels each 4-connected
color-1 component and recolors it iff its grid graph has a cycle
(`edges >= vertices`). That oracle is correct, but the compact local
representations do not assign a cycle to all pixels of the owning component
without either component labels or sequential local propagation.

Fast probe:
`python The_400_Tasks/Task_by_Task/task279/prototypes/probe_topology_cycle.py --arcgen 5000 --budget-planes 4`

Result lines:
- `[visible] component_cycle_oracle 266/266 PASS first_fail=None`
- `[arcgen5000] component_cycle_oracle 5000/5000 PASS first_fail=None`
- `[visible] local_axial_r3_reg2 47/266 FAIL first_fail=(2, (12, 13))`
- `[arcgen5000] local_axial_r3_reg2 898/5000 FAIL first_fail=(0, (16, 16))`
- `[visible] bounded_leaf_budget4 best=4/266 prune=4 regrow=0 planes=4 FAIL`
- `[arcgen5000] bounded_leaf_budget4 best=54/5000 prune=4 regrow=0 planes=4 FAIL`
- `[visible] pure_leaf_min_pass prune=8 regrow=2 planes=10 PASS`
- `[arcgen5000] pure_leaf_min_pass prune=8 regrow=1 planes=9 PASS`

Concrete under-bar ONNX probe:
`The_400_Tasks/Task_by_Task/task279/prototypes/topology_leaf_d8_f1_underbar.onnx`
from `build_qlinear_v2.py "d8 f3x1"`.

Static score: `_score_v2` `(macs=0, memory=4996, params=36)`, cost **5,032**
(`-25` vs authoritative live), sha256
`bf8ab62d0584ce5370fcedfbeb24705d3de58a0a16dec73b19cc78ebd7a5a787`.
`ng_graph_cost_diff` named the apparent saving as dropping the 25-param
`jump_kernel_u8`; memory was neutral because four added leaf-prune U8 planes
replace the two jump plus two second-flood planes.

Exact failed gate:
`python tools/ng_gate.py visible --task 279 --cand The_400_Tasks/Task_by_Task/task279/prototypes/topology_leaf_d8_f1_underbar.onnx`
returned `[visible] 264/266 (99.25%)  FAIL`.

Why dead: the generator-only minimum local topology route (`prune=8, regrow=1`)
passes 5,000 fresh ARC-GEN boxes, but two official visible pairs have
two-cell tails/barnacles on closed components. They require a second blue-gated
regrow. Adding that second regrow would add two more 16x16 U8 planes
(+512 memory), taking the same family to about 5,544 before profile, above the
5,057 live bar. The prompt's <=4-plane local topology budget is far worse
(4/266 visible), so this is not a staged win.

Stopped gates: no `truth`, `ab`, `battery`, or `profile_real_cost` were run
after the visible miss.

One remaining hypothesis: only revisit if a legal ONNX component-level aggregate
can compute `edges >= vertices` per 4-connected component and broadcast the
component decision without materialized labels or flood/regrow planes. Otherwise
the current QLinear erosion/jump/flood graph remains the practical floor.
