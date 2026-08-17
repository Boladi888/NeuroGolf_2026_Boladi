# task316 hard record - rank9 scatter table

- Live bar: 307 profile cost, SHA `c6d2f15d567f1b7243d8cd23ada653d32dedc2dcc4ab548c9946e5c502b567e2` (manifest, Sho, 2026-07-09).
- Attempted route: shrink `rank_w` / `codes_by_rank10` from 10 slots to 9 slots, saving 1 initializer param and 1 memory byte while keeping the same rank-scatter structure.
- Failed gate: visible. `python tools/ng_gate.py visible --task 316 --cand The_400_Tasks/Task_by_Task/task316/prototypes/task316_rank9_probe.onnx` returned `ERROR: InvalidArgument ... ScatterElements ... indices element out of data bounds, idx=9 must be within the inclusive range [-9,8]`.
- Why dead: rank 9 is a real no-op/sentinel index when all nine occupied columns occur before an absent column. The task-local data audit found arc-gen cases with nine occupied columns and a blank final column, so the tenth scatter slot is load-bearing even though updates at that index are zero.
- Remaining hypothesis: rank-9 clamping would need another INT32 tensor or mapping table and costs far more than the two bytes saved by deleting the sentinel slot.
