# Joint Terminal Factor Floor

- Bars used: authoritative live cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`, source `python tools/ng_current_bar.py --task 51 --json`; task-local staged reviewer bar cost 251, sha256 `a8052987d7b160721429ddd03f552ffff29d848603e1dab6626ff029f493e23b`.
- Locked scope: all probes start from `staged_candidates/task051_linemasknarroweinsum_cost251/task051.onnx`; nodes 0-25 are byte-identical to the staged 251 graph. Only node 26 (`output` Einsum) and final-only initializers were changed.
- Fixed param accounting: frozen upstream initializers cost 101 params (`one_m`, `nonzero_f`, `color_basis`, `coord_basis`, `linear_pick`, `state_core`). The 251 graph's removable terminal-only tail is 22 params: `term_t_sel_swapped` 6 + `term_t_sel` 6 + `term_p_sel` 6 + `component_scale` 4. Static memory remains 128 because frozen nodes 0-25 still produce the same scalar/vector intermediates.

## Attempt 1 - absorb `term_t_sel_swapped` into `term_t_sel`

- Candidate: `prototypes/task051_worker2_v15_absorb_swapped_into_term_fit.onnx`
- SHA-256: `0339a5d1c584e3f902165aba61f1352f8a70ae626e8acec2efaf02da0dd72619`
- Equation change: every final `term_t_sel_swapped` input was replaced by `term_t_sel`; `term_t_sel_swapped` initializer removed.
- Solve: refit the four `component_scale[H,q]` coefficients against the staged 251 graph raw output on the visible/arc-gen bundle.
- Fitted `component_scale`: `[[1.0004658, 0.0], [-0.0008072639, -0.0000139975]]`
- Param accounting: 101 fixed + `term_t_sel` 6 + `term_p_sel` 6 + `component_scale` 4 = 117 params; score `(0, 128, 117)`, cost 245.
- First failed gate: `python tools/ng_gate.py visible --task 51 --cand ...v15...onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Verdict: not a candidate; the swapped selector's orientation semantics are not recoverable by the 4-coefficient scale refit.

## Attempt 2 - absorb `term_t_sel` into `term_t_sel_swapped`

- Candidate: `prototypes/task051_worker2_v16_absorb_term_into_swapped_fit.onnx`
- SHA-256: `ba35e48e47021b21ffd4fb88f2a36ee8c36106445fd9a322a6b05d6572b92688`
- Equation change: every final `term_t_sel` input was replaced by `term_t_sel_swapped`; `term_t_sel` initializer removed.
- Solve: refit the four `component_scale[H,q]` coefficients against the staged 251 graph raw output on the visible/arc-gen bundle.
- Fitted `component_scale`: `[[1.0004674, 0.0], [-0.0008569644, 0.0000040945]]`
- Param accounting: 101 fixed + `term_t_sel_swapped` 6 + `term_p_sel` 6 + `component_scale` 4 = 117 params; score `(0, 128, 117)`, cost 245.
- Failed gate: `python tools/ng_gate.py visible --task 51 --cand ...v16...onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Verdict: not a candidate; the unswapped selector's row/column term semantics are not recoverable by the 4-coefficient scale refit.

## Attempt 3 - low-rank `term_p_sel` through `term_t_sel`

- Candidate: `prototypes/task051_worker2_v17_term_p_via_term_t_tied.onnx`
- SHA-256: `0b362d2eb62df3e53250139117a1dc1663c66c8994f3d608f8f58e027afee649`
- Intended independent equation: replace each `term_p_sel(a,t)` with `p_from_t(a,x) * term_t_sel(t,x)` and delete `term_p_sel`.
- Label constraint: the exact independent replacement needs four fresh branch labels for the distinct 3-state labels `{o,t,U,u}`. The 251 terminal equation already uses 49 of ONNX Einsum's 52 label letters; only `p`, `O`, and `S` are unused. The buildable probe tied the `U`/`u` branch to `S`.
- Actual replacement: `term_p_sel(j,o) -> p_from_t(j,p), term_t_sel(o,p)`; all `(*,t)` uses -> `p_from_t(*,O), term_t_sel(t,O)`; `term_p_sel(l,U) -> p_from_t(l,S), term_t_sel(U,S)`; `term_p_sel(B,u) -> p_from_t(B,S), term_t_sel(u,S)`.
- New initializer: `p_from_t = [[0.5, 1.0], [-0.5, 0.0]]`, the least-squares map for `term_p_sel ~= p_from_t @ term_t_sel.T`.
- Param accounting: 101 fixed + `term_t_sel_swapped` 6 + `term_t_sel` 6 + `component_scale` 4 + `p_from_t` 4 = 121 params; score `(0, 128, 121)`, cost 249.
- Failed gate: `python tools/ng_gate.py visible --task 51 --cand ...v17...onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Verdict: not a candidate; the exact four-label low-rank form is not representable within the current one-node equation label budget, and the tied-label representative visibly breaks all officials.

## Floor for This Locked Hypothesis

Under the reviewer lock (nodes 0-25 frozen; preserve `color_basis[0]` as all-channel constant, Sign-based `sh/sv`, both selector semantics, and dynamic `count_basis`), all sub-251 terminal-only factor attempts visible-fail before truth/profile gates. A visible-passing path would need either a fourth free branch label for the `term_p_sel` replacement or a different terminal tensor network that changes more than one selector family at once without increasing final-only params above 21; the three allowed probes did not find one.
