# all_shared_probe_idx

- live bar: 7,852 from `python tools/ng_current_bar.py --task 110 --json`, sha `f4d73ac8e7763eee447268e055519f195b40a19e9edb11d9bebd08c9edf3222b`; task-local reviewer bar before this pass was 7,849, sha `ba91c89f437f080d01979e01b8a61261ecaff1e7ccf0bccf863c3089b5c03749`.
- attempted route: choose probe points `[(0,3),(2,1),(3,2),(3,3)]` so all seven `GatherND` probe-index tensors are identical and reuse one `INT64[1,1,4,2]` initializer.
- exact failed gate: `python tools/ng_gate.py truth --task 110 --cand The_400_Tasks/Task_by_Task/task110/prototypes/bo_ng_cdx2_worker2/task110_shared_probe_idx.onnx --n 20000` -> `[truth] vs ARC-GEN ground truth: 19526/20000 (97.630%)`.
- why dead: despite `visible 266/266` and cost/profile `7801`, the all-shared probe set is a visible-fitting selector and falls below the §14e 99% truth floor. Do not run full truth/AB or stage this representation.
- one remaining hypothesis: less aggressive sharing can still work; this pass staged the p6/p8-only share at cost 7,841. Future search should constrain by full-corpus truth, not visible-only sharing count.
