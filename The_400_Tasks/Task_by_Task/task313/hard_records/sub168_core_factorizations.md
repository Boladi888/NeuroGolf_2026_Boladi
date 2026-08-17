# task313 sub-168 core factorizations

- Live bar used: Sho manifest/team-best cost 168, sha256 `67cb4b77d04d3305d2f1a23e0979e40ad753a5cd019e2c193c420d3e657338fa`, profile `static=168 profile=168 (memory 0 + params 168)`.
- Attempted route: keep Sho's fixed period-6 coordinate basis and active-mask free-output `Einsum`, but reduce the 40-param grouped core or replace it with a lower CP rank.
- Failed gates:
  - `task313_probe_ae_bf_k1.onnx`, cost 148, sha256 `2fc8f5f711fd2996809afdb98fe2da391e2d680248e10f5335a2a78b2c30708d`: `python tools/ng_gate.py visible --task 313 --cand The_400_Tasks/Task_by_Task/task313/prototypes/task313_probe_ae_bf_k1.onnx` -> `[visible] 181/266 (68.05%)  FAIL`.
  - `task313_probe_ab_ef_k2.onnx`, cost 160, sha256 `09aac83145c94c523e6c97456e86bcf551a713ed315f0b084cacc95fb889682b`: `python tools/ng_gate.py visible --task 313 --cand The_400_Tasks/Task_by_Task/task313/prototypes/task313_probe_ab_ef_k2.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
  - `task313_probe_af_eb_k2.onnx`, cost 160, sha256 `1006b6f123a5439db6d38cf22699092fd8cc4daba3d5f2398220f609acee8459`: `python tools/ng_gate.py visible --task 313 --cand The_400_Tasks/Task_by_Task/task313/prototypes/task313_probe_af_eb_k2.onnx` -> `[visible] 143/266 (53.76%)  FAIL`.
  - `task313_probe_cp_rank3.onnx`, cost 164, sha256 `e1c4b2590ed791c11b7c2c9b09bef4e3287877f8432258376116f8a23a9d465f`: `python tools/ng_gate.py visible --task 313 --cand The_400_Tasks/Task_by_Task/task313/prototypes/task313_probe_cp_rank3.onnx` -> `[visible] 122/266 (45.86%)  FAIL`.
- Why dead: all sub-168 fixed-basis core reductions missed visible after sign-training, so the 40-param Sho grouped core is not replaceable by rank-1 `(AE)x(bf)`, rank-2 `(Ab)x(Ef)`, rank-2 `(Af)x(Eb)`, or CP rank-3 under the current period basis.
- One remaining hypothesis: train a genuinely new period-6 coordinate basis with dimension 3 and a dense/factored core; quick fixed-projection checks of all 3-column subsets failed the thresholded one-hot objective, but a full coordinate optimizer was not attempted in this pass.
