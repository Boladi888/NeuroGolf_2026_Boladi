# task350 R0-via-R1 remap gate-timeout record

- Live bar: 428, SHA-256 `36347083a3e94eeb3aee50bfc307b5171afa56c691e6d0bbf642399025b546e6` (`python tools/ng_current_bar.py --task 350 --json`, authoritative manifest, owner Vamshi, updated 2026-07-12T15:15:54Z).
- Attempted route: one-node free-output `Einsum` algebraic remap. Delete `R0` `[3,3,3]` and reconstruct it as `P0[f,z] * R1[z,a,b]`, where `P0` selects R1 slices `[1,2,2]`. Static/profile cost dropped 428 -> 410.
- Candidate: `The_400_Tasks/Task_by_Task/task350/prototypes/task350_r0_via_r1_remap.onnx`, SHA-256 `5db98e0e4d36db2dd21dbb9225db76305d32b2d7457602d876a16ed05788138b`.
- Passed gates before timeout: visible `267/267`, profile `static=410 profile=410`, truth default `2000/2000`, A/B default vs explicit Team_Best `2267/2267 byte_equal=True`, truth `--n 8000` `8000/8000`, safety `PASS` with 0 warnings/failures.
- Exact failed gate: `python tools/ng_gate.py ab --task 350 --cand The_400_Tasks/Task_by_Task/task350/prototypes/task350_r0_via_r1_remap.onnx --ref Team_Best_Onnx/task350.onnx --n 8000` returned `[ng_gate] wall budget 900s exceeded for ab task350; exiting to avoid orphaned gate work`.
- Why dead for this worker pass: protocol forbids relaunching the same per-candidate gate with a larger `--max-seconds` after a shared wall-budget timeout. This candidate is not staged and must not be called LOCAL_GOLD/reviewer-ready from this pass.
- One remaining hypothesis: a reviewer/captain with an allowed heavy-gate policy can finish the explicit-parent `ab --n 8000`, or a further factorization can save params without adding enough operands to make A/B exceed the default worker wall budget.
