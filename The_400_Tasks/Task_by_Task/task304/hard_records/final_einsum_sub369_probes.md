# task304 final-Einsum sub-369 probes dead record

## Live bar
- Team-best bar used: cost 369, profile_cost 369 per task note; authoritative manifest/workbook agree on cost 369.
- Team manifest sha: `39596836ef02364ba27c051e5da3431299dfd0f9730136de65bf132db3a06cd8`.
- Local extracted team-best: `The_400_Tasks/Task_by_Task/task304/prototypes/extracted_submission_task304.onnx`, score `(0, 88, 281)`, cost 369, same sha.

## Attempted routes
- `probe_out9_selector_shrink.onnx`: v5 math with graph output shrunk to `[1,10,9,9]`, split `Dmod9/Dmod30`, cost 333, sha `a01315fc1095bc9459046f9846170118944ebefd0336b96009533b754a8aea4d`.
  - Failed gate: `[visible] 0/266 (0.00%)  FAIL`.
  - Dead because NeuroGolf expected outputs are encoded as `[1,10,30,30]`; padding the 9x9 tensor after the final Einsum would make the `[1,10,9,9]` tensor a charged intermediate and lose the selector saving.
- `probe_no_h2_direct_channel.onnx`: v5 final Einsum with H2 deleted and tile input tied directly to output channel `c`, cost 269, sha `87e633a8017c0354cfe5f1896f59ba299ded6e10b1d83317b25c6ab5cad4d595`.
  - Failed gate: `[visible] 0/266 (0.00%)  FAIL`.
  - Dead because direct channel tie produces foreground channels but never emits positive channel 0 for unstamped/background blocks. H2 implements the required `T_c - e0[c] * sum_d T_d` channel mix; deleting it removes the background term.
- `probe_sparse_selector_inits.onnx`: v5 with H2/Ddiv/Dmod as sparse initializers; theoretical sparse cost would beat the bar.
  - Failed at build/score before gates: `InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum): [ShapeInferenceError] Rank of input 2 (0) does not match the equation indices (2).`
  - Dead because ONNX shape inference does not expose sparse initializer rank to Einsum. Adding dense value_info would charge H2/Ddiv/Dmod as intermediates and lose the bar.
- `probe_reuse_cnt_name.onnx`: intentional SSA/name-reuse attempt to overwrite `cnt` with the signed margin and save the separate 40-byte `sgn` value_info.
  - Failed at build/score before gates: `ValidationError: Graph must be in single static assignment (SSA) form, however 'cnt' has been used as output names multiple times.`
  - Dead because the official checker enforces SSA before scoring.

## Remaining hypothesis
The only plausible sub-369 path left is a true single-output-node way to express the sum of the free identity channel term and the channel-0 background rank-1 term without materializing a full output tensor and without the 100-param H2 matrix. Current ONNX Einsum is single-term multiplication/summation, so this would require a different final op family or a scorer-accepted sparse/structured initializer mechanism.
