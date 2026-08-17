live bar: 2360, sha f12c26da73d8ad2a657625cc50a07fa1faac4f95694b6195ffb763355439e9cf

attempted route: remove Equal(tv,132) / Squeeze(valid2) / Where(valid,sums,0) so ArgMax sees raw sums.

failed gate: `python tools/ng_gate.py visible --task 365 --cand The_400_Tasks/Task_by_Task/task365/prototypes/task365_probe_no_valid_mask.onnx` -> `[visible] 265/266 (99.62%)  FAIL`.

why dead: the invalid third TopK slot can accumulate a positive rectangle sum and win at least one official visible case; one visible miss is a certain hidden zero.

one remaining hypothesis: only shape-preserving rewrites of the valid mask are viable; deleting the guard is not.
