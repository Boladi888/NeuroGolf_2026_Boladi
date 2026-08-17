# task055 ScatterElements update broadcast route - DROP

- Live bar: **1,113** static/profile, SHA-256
  `b7bab0c46528aece35efdce0ae4db78dfad986e2eb685b3103fba98a07cb7853`.
  Source: `python tools/ng_current_bar.py --task 55 --json` on 2026-07-13;
  authoritative source is the team manifest. `Team_Best_Onnx` cost metadata
  1,106 and local task-state cost 1,115 are stale for this hash.
- Attempted route: replace the v14 parent `C3` UINT8 `[2,1] = [3,3]`
  ScatterElements update initializer with broadcastable smaller tensors
  (`[]`, `[1]`, `[1,1]`, `[2]`) to shave duplicated line-code params without
  changing any node topology.
- Exact failed gate: `score_v2` / ORT runtime gate = FAIL for every variant.
  Probe artifact:
  `The_400_Tasks/Task_by_Task/task055/prototypes/scatter_update_broadcast_probe/scatter_update_broadcast_probe.json`.
  ORT errors: "Indices and updates must have the same rank" or
  "Indices vs updates dimensions differs at position=0 2 vs 1".
- Why dead: ORT accepts only update tensors matching ScatterElements index
  rank/dimensions here. Replacing the `[2,1]` duplicate update tensor with a
  scalar or singleton shape yields scorer `None`, so it is not a valid 1-byte
  shave and must not be staged.
- Remaining hypothesis: only a different source-read/free-output formulation
  that deletes a charged 30-byte axis step or the 300-byte `xk` expansion can
  beat the current parent. Do not revisit ScatterElements update broadcasting.
