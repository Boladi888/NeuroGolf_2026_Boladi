# task248 mod_reducemin_pair_no_gain hard record

- Live bar: cost 151, SHA-256 `6a001ce01217f4e505777fb369eb36ec242ffdc413030285fdb28dc76479d892`, source `python tools/ng_current_bar.py --task 248 --json` / team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Attempted route: replace `Mod -> Sub -> Min` reflection with a positive/negative INT8 offset pair, `Mod([k,-k], period) -> ReduceMin(axis=1)`, hoping node deletion would shave cost while preserving the same ScatterElements tail.
- Prototype: `The_400_Tasks/Task_by_Task/task248/prototypes/probe_mod_reducemin_pair.onnx`, SHA-256 `ca45a1e2781a9af75207df6295d1ab99ccb7116c2097ff98780af930d63ec1b2`.
- Exact failed gate: visible passed `[visible] 13/13 (100.00%)  PASS`, but `_score_v2`/real profile was `static=160 profile=160 (memory 122 + params 38; 13 real inputs)`, which is +9 vs the 151 bar.
- Why dead: NeuroGolf charges tensors and params, not node count. The rewrite keeps memory flat but doubles `row_offsets` from 9 to 18 INT8 elements; `ng_graph_cost_diff` attributes the net regression to `+9 params`, with `+0 memory`.
- One remaining hypothesis: a strict sub-151 win must avoid or shrink the final two-channel INT32 ScatterElements index/update tail; algebraic rewrites of the triangular-wave path that preserve the same `[1,2,9,1]` scatter shape are exhausted unless they delete params.
