# rank2_channel_cp

Live bar: cost 225, sha256 `d3ed7d76188be734565d1620ae49ffc6fdc819b145d228e5ca602497113dcb80`, source `python tools/ng_current_bar.py --task 60 --json` (manifest, owner Aditya, updated 2026-06-12T18:26:48Z).

Attempted route: keep the live full-shape one-node `Einsum` scaffold but reduce the channel CP block from q=3 to q=2, dropping params from 225 to 202. Builder/probe: `The_400_Tasks/Task_by_Task/task060/prototypes/build_task060_candidates.py`; candidate `task060_rank2_fit.onnx`, sha256 `ae1887b3d93632f4406bc475a19b362e1787506c5303ca08cf5b72a4e395cdc5`.

Exact failed gate: `python tools/ng_gate.py visible --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_rank2_fit.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.

Why dead: q=2 cannot separate the required shared channel signs for 10-way identity copy plus divider map `0->0, nonzero->5`. Numerical hinge search over the two channel maps reached only `192/200` sign constraints with worst margin `-0.4114823664035592`; visible failure confirms the representation miss. No truth/AB/profile run.

One remaining hypothesis: a different final-op family that preserves channel identity for the halves without paying the q=3 sign-rank block, while separately producing the active-row gray divider, could beat 225; grouped/depthwise Conv does not appear to supply the needed cross-channel gray routing.
