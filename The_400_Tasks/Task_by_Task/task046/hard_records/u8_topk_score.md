live bar: 2278, sha a5e2b2cd6adcfd3bdb0826b4ce68d557205f6428c410188d4569a57a81300509.

Attempted route: narrow v30 TopK score lane from F16 to UINT8 (`prototypes/build_task046_snake_v31_u8_topk_score.py`, `prototypes/task046_v31_u8_topk_score.onnx`) so `score[20]` and unused `tk_vals[16]` would bill one byte per element.

Failed gate: score/profile. `_score_v2` returned `None`; `profile_real_cost.py --task 46 --cand prototypes/task046_v31_u8_topk_score.onnx --arcgen 266` failed with `NOT_IMPLEMENTED : Could not find an implementation for TopK(11)` on the UINT8 score input.

Why dead: ONNX checker accepts the type, but ORT has no UINT8 TopK kernel in this environment. F16 is the practical score-lane floor unless the TopK compaction itself is replaced.

One remaining hypothesis: replace TopK with a different ordered compaction primitive that does not materialize the INT64 `tk_idx[16]`; prior cumsum/OneHot and prefix/band routes were already priced worse, so this needs a genuinely new compaction formulation.
