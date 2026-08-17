# task122 hard record: no-bias 4x4 group=2 Conv

- Live/team-best bar: 810, SHA256 `9d256b458e31da4423eee2f1c90ac42204a8e573ea622c1b679c25865828eabf`, source `python tools/ng_current_bar.py --task 122 --json` / team manifest.
- Attempted route: keep the v10 one-node `group=2` free-output Conv with kernel area 16, but delete the full Conv bias so cost would be `10*5*4*4 = 800`.
- Failed gate: construction-level separability probe before ONNX emission. Command:
  `python The_400_Tasks/Task_by_Task/task122/prototypes/probe_task122_area15_group2_conv.py --max-dilation 3 --max-side 4 --max-area 16 --shapes 4x4 --no-bias --early-conflict`
- Exact result: `checked=441 conflict_dead=440 no_conflict=1 lp_dead=1 feasible=0`; the only no-conflict survivor was `kh=4 kw=4 dh=1 dw=1 pads=[2,2,1,1] cost=800 unique_patches=595`, and LP feasibility was false.
- Why dead: almost every 4x4 no-bias placement has identical-patch/different-label conflicts, and the current v10 receptive field still requires an intercept for all live output channels. The full-length bias is not removable in this formulation.
- One remaining hypothesis: leave the one-node Conv family and find a direct free-output formula whose row/col operands stay below the documented 960 scalar-rebuild floor; short/omitted/subdeclared Conv bias should not be retried.
