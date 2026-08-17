# task086 qscale-role-scale-dedup hard record

- Live bar used: manifest live cost 1726, sha256 `427eefa31e5d958f6f92de003be51278d465084b0e596305f0ae9f080cdbe14c` from `python tools/ng_current_bar.py --task 86 --json`.
- Attempted route: replace the two role-detector `sat_scale=1/255` QLinearConv scale uses with existing `q_scale=1.0`, delete `sat_scale`, and preserve the same three-channel role-code renderer.
- Failed gate: `python tools/ng_gate.py visible --task 86 --cand The_400_Tasks/Task_by_Task/task086/prototypes/task086_worker2_qscale_role_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the detector outputs are not merely binary signs under `q_scale`; the saturation scale is load-bearing for the downstream outer detector and final positive/zero decode, so the 1-byte initializer shave is not semantics-preserving.
- Remaining hypothesis: a separately trained/proved two-feature nonlinear direct detector may still delete `occ_u8`/`outer_score_u8`/`Concat`, but it must beat the 1726 bar and visible-gate its support role before any truth or A/B run.
