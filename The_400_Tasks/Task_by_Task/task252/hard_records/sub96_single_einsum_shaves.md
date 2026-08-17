# task252 sub-96 single-Einsum shaves

- Live/team-best bar: cost 96, owner Sho, sha256 `fd9ccf660dc58398f6234fe62deca4e1f02cae0f19f61b77b0e153268f4cbf6f`, source `python tools/ng_current_bar.py --task 252 --json` on 2026-07-14.
- Parent graph: one final `Einsum` with `code_out[3,10]`, `diag_state[2,3]`, `parity[2,30]`; static/profile cost 96.

## Sparse parity probe

- Attempted route: replace dense `parity[2,30]` with a sparse initializer containing the 30 one entries, preserving the parent graph exactly otherwise.
- Prototype: `The_400_Tasks/Task_by_Task/task252/prototypes/task252_sparse_parity_probe.onnx`
- Builder: `The_400_Tasks/Task_by_Task/task252/prototypes/build_task252_sparse_parity_probe.py`
- Exact failed gate: `python -c "from tools._score_v2 import score_network_v2; print(score_network_v2('The_400_Tasks/Task_by_Task/task252/prototypes/task252_sparse_parity_probe.onnx'))"` returned `None`.
- Profile failure: `python tools/agent_helpers/profile_real_cost.py --task 252 --cand The_400_Tasks/Task_by_Task/task252/prototypes/task252_sparse_parity_probe.onnx --arcgen 266` failed before running with ORT `Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`
- Why dead: scorer sanitization does not preserve the sparse initializer reference for this `Einsum` input; this is an invalid model path, not a candidate.

## Rank-2 shared color code

- Attempted route: reduce the parent formulation from shared rank 3 to shared rank 2, which would cost `parity[2,30] + code[2,10] + diag_state[2,2] = 84`.
- Why dead: the even-parity map alone requires the 10-color identity sign pattern under a shared symmetric bilinear form `v_o^T A v_c` with positive diagonal and negative off-diagonal. In 2D, a positive-definite form allows at most three pairwise obtuse vectors; an indefinite form has two positive cones and any two vectors in the same cone have positive product, so it allows at most two mutually negative cross-pairs. Degenerate forms do not improve this. Therefore 10 colors force shared dimension at least 3 before the odd recolor constraints are considered.

## Remaining hypothesis

A sub-96 win likely needs a different final-op family that avoids storing `parity[2,30]` or a formally new way to generate period-2 column state inside the output op without a counted intermediate. Dense/sparse initializer shaving and rank-2 shared color compression should not be re-run.
