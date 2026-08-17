# task317 hard record: tied-channel vector visible-zero

- live bar used: team-best manifest / `Team_Best_Onnx/task317.onnx`, cost 110, sha256 `ebd09483ac75c93d2857c62e54dab9d3471c849153525d93b9a7fc725933038a`.
- attempted route: collapse Sho's two channel operands `tvec[10]` and `pvec[10]` into one shared `v[10]` inside the one-node final `Einsum`, reducing static cost 110 -> 100.
- candidate: `The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx`, sha256 `57dcc76ce0aed3c48f794d0010c7920d85f822c67fc69fa8857212f197d2fcf3`.
- exact failed gate: `python tools/ng_gate.py visible --task 317 --cand The_400_Tasks/Task_by_Task/task317/prototypes/cand_task317_tied_channel_cost100.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- why dead: the negative coefficient needed for input channel 5 also becomes an output-channel-5 positive coefficient on gray/blue blocks when the vector is shared. That produces multi-hot output cells, so the apparent -10 param saving is not a legal representation.
- one remaining hypothesis: only a different legal way to provide either the size-30 rank-3 block basis or the output-channel sign mask without dense operands and without charged intermediates could beat 110; this tied channel-sharing family should not be retried.
