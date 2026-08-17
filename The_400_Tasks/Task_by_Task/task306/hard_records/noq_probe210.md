# task306 hard record: no-Q 210-param Einsum probe

- Live bar used: cost 220, SHA256 `c935fce45d093602ea4b8e4a1cbf9d611539d14bc88000cc76384795c57689a1`, source `python tools/ng_current_bar.py --task 306 --json` / team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Attempted route: keep the live graph's shared `A[30,2]` + `B[30,5]` coordinate factorization but remove the `Q[10]` channel-sign vector, producing a single free-output `Einsum` with 210 params and no intermediates.
- Probe artifact: `The_400_Tasks/Task_by_Task/task306/prototypes/build_task306_noq_probe210.onnx`, SHA256 `75fd6d42bc6f1eec62af7c73ca069ad355789d7eea4905bdb4723aa1a3157b9b`, static cost 210 (`memory=0`, `params=210`).
- Failed gate: `python tools/ng_gate.py visible --task 306 --cand The_400_Tasks/Task_by_Task/task306/prototypes/build_task306_noq_probe210.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: without `Q[H]` and `Q[h]`, the replicated source color and background channel have the same sign at non-background offsets, so channel 0 stays positive where it must be suppressed. This confirms the 10-param channel sign/select vector is load-bearing for this single-Einsum family. Per visible-fail axiom, no truth/AB/profile gates were run.
- Remaining hypothesis: only a fundamentally different free-output contraction that replaces both the channel sign vector and the 2x5 coordinate factors with a lower-rank exact relation could beat 220; no such relation was identified, and the coordinate equality still needs 10 modulo classes with the known best categorical factor sum `2+5=7`.
