# task266 hard record: route_b_einsum_terminal

- Live bar used: 144 bytes, manifest source, sha256 `ba939e8f4d4f6ae913e8bdee81490293374ea48e61814e5b8bbf80b6c6b13e9a` (`python tools/ng_current_bar.py --task 266 --json`).
- Reviewer-locked route: rebuild from the decoded ARC-GEN 3x5 diagonal-marker rule and make one terminal output writer consume the forced `[1,1,3,5]` F32 marker slice directly, eliminating live `Cast`/2-channel state.
- Numpy rule proof: `python The_400_Tasks/Task_by_Task/task266/prototypes/prove_task266_rule_arcgen.py` -> `PASS`, 3000 fresh ARC-GEN rows, all 15 marker positions covered. Proof JSON: `The_400_Tasks/Task_by_Task/task266/prototypes/task266_rule_proof_3000.json`.
- Builder attempt 1/2: `python The_400_Tasks/Task_by_Task/task266/prototypes/build_task266_route_b_einsum_terminal.py`.
- Prototype: `The_400_Tasks/Task_by_Task/task266/prototypes/task266_route_b_einsum_terminal.onnx`, sha256 `afc2b4a244984a21175a3757aca2718db2488ab25edfd29c3282dc5f05299a34`.
- Skeleton: `Slice(input[:,2:3,0:3,0:5]) -> Einsum(marker,row,col,chan) -> output`.
- Score/profile result: `score_v2=(0,60,818)`, `profile_real_cost --arcgen 266 -> static=878 profile=878 (memory 60 + params 818; 20 real inputs)`.
- Exact stop condition: profiles `878 >= 144`, so no visible/truth/safety gates were run.
- Why dead: the terminal #89 writer keeps only the 60B marker slice as charged memory, but boundary-correct 30x30 output requires dense row/column selector params: `row[3,3,30]=270`, `col[3,5,30]=450`, `chan[3,3,10]=90`, plus slice starts/ends `8`. The selector params alone exceed the team-best bar.
- Remaining hypothesis: only a non-dense zero-param selector generator or an ORT-supported terminal op with convolution-like local shifts plus an active-grid boundary mask could rescue Route-B; the direct dense #89 selector skeleton is not competitive.
