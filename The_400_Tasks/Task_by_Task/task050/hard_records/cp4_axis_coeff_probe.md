# task050 approximate CP4 one-axis coefficient probe — DEAD

- Live bar used: cost **164**, sha256 `c88d4f740152c3458c3389cf1634f385b2ae8ffee589c78982a1d0b6d6fbf14a`, from `python tools/ng_current_bar.py --task 50 --json` (authoritative_source=manifest, owner Boladi, updated 2026-07-10T22:11:58Z).
- Attempted route: replace the direct-164 `S''[2,3,4] + A[4,4,3]` one-axis coefficient storage (72 params) with a CP-rank-4 approximation `U[2,3,4] + V[2,3,4] + Q[3,4]` (60 params), keeping `P[10,2]`, `G[30,2]`, and shared `D[2,2,3]`.
- Prototype: `prototypes/build_direct152_cp4_approx_probe.py` -> `prototypes/task050_direct152_cp4_approx_probe.onnx`, sha256 `0fab8a109aa880be95d5b80eb065649e07a9d05f1cb8051db7079fdf578aaf29`.
- Static score: `(macs=0, memory=0, params=152)`.
- Failed gate: `python tools/ng_gate.py visible --task 50 --cand The_400_Tasks/Task_by_Task/task050/prototypes/task050_direct152_cp4_approx_probe.onnx` -> `[visible] 62/271 (22.88%)  FAIL`.
- Why dead: the rank-4 ALS residual sits on the active-baseline term (`-1/256`); visible confirms that approximating/dropping this baseline breaks the argmax preservation logic. One visible miss kills the family, so no truth/profile/AB gates were run.
- Rank check note: a follow-up full `W[6,6,3]` CP4 ALS check (allowing cancellation through otherwise-unused raw basis slots) still bottomed out at residual norm `0.003958591891117823`, max abs `0.003911361074253319` (near `1/256`). This is numeric evidence, not a symbolic rank proof. CP5 is the known five-term expansion and costs `P20 + G60 + D12 + U30 + V30 + Q15 = 167`, so it loses to 164.
- One remaining hypothesis: a symbolic exact CP4 factorization would be worth building only if proven algebraically first; absent that, look outside this coefficient-compression route for a different one-node signed-output encoding.
