# task313 learned 3D period-6 dense-core infeasibility

- Live bar used: Sho manifest/team-best cost 168, sha256 `67cb4b77d04d3305d2f1a23e0979e40ad753a5cd019e2c193c420d3e657338fa`.
- Locked hypothesis: replace Sho's shared `v_w0[30,4]` period-6 coordinate basis with learned `coord3[30,3]`, keep one free-output `Einsum`, and use dense `core[2,3,2,3]`; expected cost `90 + 6 + 36 = 132`.
- Builder: `The_400_Tasks/Task_by_Task/task313/prototypes/build_task313_learned3d_dense.py`.
- Candidate: `The_400_Tasks/Task_by_Task/task313/prototypes/task313_learned3d_dense.onnx`, sha256 `b0b10a08d527208f93ef705c322cdcd65f988dd20a0c06d32479e379537c9be9`, static score `(macs=0, memory=0, params=132)`.
- Optimizer cap used: 12 restarts, 300 L-BFGS iterations per restart. Best training result was restart 12 (`noise_11_from_subset_013`): argmax `36782/43463`, thresholded one-hot sign `35789/43463`, loss `212108.9194933998`.
- Failed gate: `python tools/ng_gate.py visible --task 313 --cand The_400_Tasks/Task_by_Task/task313/prototypes/task313_learned3d_dense.onnx` -> `[visible] 11/266 (4.14%)  FAIL`.
- Why 3D basis failed: the learned dimension-3 period basis can produce many correct argmax colors but cannot make the NeuroGolf decoder's required one-positive-channel/all-other-nonpositive sign pattern across the visible cells. The fourth coordinate degree in Sho's `v_w0[30,4]` appears load-bearing for the thresholded one-hot margins, not just for exact dense-core reconstruction.
