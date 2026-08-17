# task302 worker2 scalar micro-shaves

- **Live bar**: 1574, Vamshi, sha256 `366237800121a80b62e22e725646ec0d3c7a6a56a5ec3dd71a57d350e6498528`, source `python tools/ng_current_bar.py --task 302 --json` / team manifest.
- **Effective local reviewer bar**: 1573, `staged_candidates/task302_regionfillrewrite_cost1573/task302.onnx`, sha256 `f39b26ef41e9d40cb7cfd1ecb511d289e57af66377cbb0e188ef87669b21937e`.
- **Attempted route**: one-param scalar eliminations inside the staged six-node representation, without changing the detector stack shape.

## Attempts

| Probe | Target | Exact failed gate | Why dead |
|---|---:|---|---|
| `build_task302_final_xzp0.py` / `task302_final_xzp0.onnx` | 1572 | `ng_gate visible`: `[visible] 0/266 (0.00%)  FAIL` | Replacing final `x_zero_point=1` with `0` and bias-shifting by the full kernel sum ignores QLinearConv pad semantics. The original `x_zero_point=1` is the neutral pad value for the output-expanding final Conv; a single per-channel bias cannot compensate the position-dependent border overlap. |
| `build_task302_final_wzp1_u8.py` | 1572 | Builder MILP: `channel 0 UINT8 wzp=1 MILP failed: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)` | Reusing `one_u8` as a UINT8 final weight zero point constrains effective final weights to `[-1, 254]`. Channel 0 is not linearly separable under that bound on the 1,377 staged feature patches. |
| `build_task302_frame_count_final.py` | 1572 | Builder MILP: `channel 0 frame-count MILP failed: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)` | Reusing `q_scale=1` for the frame detector deletes `y_scale5` but changes the second feature from binary fill support to count-valued fill support. Even with 273,617 unique count-feature patches, channel 0 is not linearly separable by the fixed 3x5 final classifier. |

## Verdict

The remaining scalar initializers in the 1573 representation are load-bearing for either QLinearConv border semantics, dtype typing, or channel-0 separability. These probes do not justify truth/AB/full gates.

## Remaining hypothesis

A further win likely has to remove a full feature plane or replace the final classifier family, not scalar-golf the current QLinearConv stack. The only plausible unexplored direction is a different low-memory nonlinear signal for active background that preserves the final pad-neutral behavior without materializing another full 12x12 plane.
