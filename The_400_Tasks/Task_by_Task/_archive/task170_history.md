# task170 — archived history

Cold storage for superseded task170 prose. Current state lives in `Task_by_Task/task170.md`. See its Attempts table for the one-row-per-rung ladder.

## Cost analysis (pre-2535 rungs, archived 2026-06-12)

Baseline cost is memory 78,315 + params 3,815 = 82,130. The old scalar-v17 cleanup reduced params only to 3,140, yielding +0.00825 points and no memory change.

The selected v2 candidate reduces params to 216 and memory to 77,675. The parameter drop comes from replacing row-constant top/bottom masks with broadcast row masks and preserving the redundant scalar `v17`; the memory drop comes from removing the final Pad and stale `v1152` value_info so the 1x10x4x4 tile is returned directly instead of materializing a counted pre-pad intermediate. Total cost is 77,891, delta -4,239, approximately +0.05299 points under `25 - log(cost)`.

Evidence packet: `.hermes/task170_t_073ea722/delta_packet_task170_v2_row_masks_no_final_pad.json`.
Candidate ONNX: `.hermes/task170_t_073ea722/task170_v2_row_masks_no_final_pad.onnx`.
Candidate local submission zip: `.hermes/task170_t_073ea722/submission_task170_v2_row_masks_no_final_pad.zip`.

2026-06-11 staged best `task170_microshrink_i32` reduces the live team-best from 2,967 to 2,568 by deleting the separate `row_fg`/`col_fg` profiles, replacing 4x4 cursor `ArgMax` with a weighted F32 `Einsum`, and narrowing post-ArgMax Slice-control tensors to INT32. (Superseded 2026-06-12 by `task170_motifsel_fold` at 2,535.)

## Public submit result (archived 2026-06-12)

2026-05-17 submit-captain staged the package as `.hermes/board/staged_submits/task170_t_073ea722_v2_on_5736_60_20260517T025900Z/submission.zip` and submitted it on top of the 5736.60 canonical best. Kaggle completed at **5722.92**, i.e. **-13.68 vs 5736.60** despite the local +0.05299 expectation. The unchanged no-final-Pad/direct-4x4-output package is now public-negative and must not be requeued unchanged. (This is why the Don't-do log forbids the no-final-Pad shape; padded 30x30 output is preserved by all current candidates.)
