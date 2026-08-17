# task215 hard record: rank2_phase_sub100

- live bar used: cost 100, sha256 `8c39d265256613c2eae73e883f5ed72cc95708849bc3d791f80a1d4cc6f63bbf`, source `python tools/ng_current_bar.py --task 215 --json` / team manifest.
- attempted route: replace the shared `[30,3]` one-hot phase map with a shared `[30,2]` real trig phase factor in the current terminal-Einsum family, keeping the shared `chan[10]`; static cost 70, sha256 `3af20b29ea0ed10064c067cc61f99818894f59e2889884d7958b9aa586271127`.
- fast probe: `python The_400_Tasks/Task_by_Task/task215/prototypes/probe_low_rank_phase.py --fresh 50 --random 10` showed rank3 one-hot PASS 315/315, rank2 hand/random factors fail on pair 0.
- exact failed gate: `python tools/ng_gate.py visible --task 215 --cand The_400_Tasks/Task_by_Task/task215/prototypes/task215_worker3_rank2_phase70.onnx --json --out The_400_Tasks/Task_by_Task/task215/prototypes/gate_visible_worker3_rank2_phase70.json` -> `correct=0 total=265 pct=0.0 pass=false`.
- why dead: the modulo-3 all-same tensor needs three phase classes in this one-node sign-family; rank-2 phase codes leak cross-phase contributions and fail the first visible row.
- one remaining hypothesis: a different final-op family that avoids explicit row-phase equality entirely, not another lower-rank shared phase factor inside this contraction.
