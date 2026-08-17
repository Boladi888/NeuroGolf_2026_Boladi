# task237 hard record - QLinear tail microshaves

Live bar refreshed with `python tools/ng_current_bar.py --task 237 --json`: manifest live cost 705, sha256 `a59a2fa610fa58f67f8026812f7a160bf82f9fb1b65955c50fb91dde71bf538e`, updated 2026-07-02T20:17:39Z. Lower task-local staged bar to beat before reviewer-ready staging: `staged_candidates/task237_rankone_cost696`, cost/profile_cost 696, sha256 `64549e67db188f7dceb7ccfd133fd2d058158845f3c1fd138ec668060fad33e6`.

Attempted route: delete tiny charged tail pieces from the 696 rank-one graph without changing the cropped carrier. Two probes were checked:
- Absorb final `QLinearConv`'s 10-element `q_bias` into quantization zero-points. Failed pre-build linear feasibility: no scalar input zero-point `z in [0,100]` admits two-channel quadratic separators `A*(x-z)+B*(x*x-z)` for all code classes 1..10.
- Replace `code_vec2` false branch `col_active_u8` with `active_col_code`, then remove `col_active_u8` for a 9-byte shave. Failed pre-build linear separability: with background points `(j,j)` for active columns and color points `(t,t*t)`, output color1 at `(2,4)` is not separable from the remaining background/color points by any 2-channel linear `QLinearConv` classifier in coefficient range `[-128,127]`.

Exact failed gate: pre-build algebraic feasibility check, not a runner gate. No ONNX was staged and no visible/truth/ab gate was run on these impossible formulations.

Why dead: both shaves require the final two-channel `QLinearConv` to classify code pairs that are not linearly separable under the decoder's strict one-hot requirement. Adding a third feature/channel would increase `code_vec2` by 81 bytes and loses the intended microshave.

Remaining hypothesis: a different final output family that performs the 9x9 spatial expansion inside the free output node could still delete `region_select` (81 B) and `code_vec2` (162 B), but a dense spatial `QLinearConv` kernel must stay under the saved bytes; naive 3-4 input-channel kernels price above this floor.
