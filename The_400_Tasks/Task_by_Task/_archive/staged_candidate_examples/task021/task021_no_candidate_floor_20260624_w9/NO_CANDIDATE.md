# task021 no-candidate packet - worker w9, 2026-06-24

Verdict: **no-candidate**. Team-best for `beta_p7s_codex_cli` remains Aditya cost 171:
`Team_Best_Onnx/task021.onnx`, sha256 `973e1edb62ba96f50e4db8d2224032e019ff98c9d85ed565f39666ae67e3cac9`.

## EV source

`Agenters/Orchestrator/runs/beta_p7s_codex_cli/team_best.csv`:
`task021,Aditya,973e1edb62ba96f50e4db8d2224032e019ff98c9d85ed565f39666ae67e3cac9,171,19.8583,2026-06-08T23:23:49Z`

Expected delta versus team_best=171: **0**, because no strict-improvement candidate was found.

## Gates on anchor

- `python tools/ng_gate.py visible --task 21 --cand Team_Best_Onnx/task021.onnx --json` -> 77/77, 100%, pass true.
- `python tools/ng_gate.py truth --task 21 --cand Team_Best_Onnx/task021.onnx --full --json` -> 200000/200000, 100%.
- `python tools/ng_gate.py ab --task 21 --cand Team_Best_Onnx/task021.onnx --full --json` -> byte_equal true, 200077/200077, divergences 0.
- `python -c "from tools._score_v2 import score_network_v2; print(score_network_v2('Team_Best_Onnx/task021.onnx'))"` -> `(0, 156, 15)`, static cost 171.
- `python tools/agent_helpers/profile_real_cost.py --task 021 --cand Team_Best_Onnx/task021.onnx --arcgen 266` -> failed in the generic helper encoder with `IndexError: index 30 is out of bounds for axis 0 with size 30`.
- `python The_400_Tasks/Task_by_Task/task021/prototypes/profile_task021_official_arcgen.py --cand Team_Best_Onnx/task021.onnx` -> `inputs_total=266`, `inputs_profiled=77`, `oversize_skipped=189`, `static_cost=171`, `profile_cost=171`, `profile_memory=156`, `params=15`.

The generic profile helper failure is task021-specific: bundled arc-gen includes oversized raw grids that official `neurogolf_utils.convert_to_numpy` returns as `None`. The task-owned wrapper uses official semantics and profiles all encodable members of the 266-pair full set.

## Floor audit

Fresh ARC-GEN source read: `external/ARC-GEN/tasks/task_1190e5a7.py` samples `rows` and `cols` independently and returns output `len(rows) x len(cols)` in the fill color.

Fresh encodable-generator audit (`H<=30,W<=30`, 20,000 valid samples, 66,840 raw draws):
- row0-only signatures: 6,137 total, 4,148 ambiguous output-height signatures.
- col0-only signatures: 6,119 total, 4,150 ambiguous output-width signatures.
- row0+H signatures: 18,606 total, 889 ambiguous output-height signatures.
- col0+W signatures: 18,565 total, 918 ambiguous output-width signatures.
- Example row0 collision: `((5,24),(6,6),W=30)` produced `2x7` and `6x7`.
- Example row0+H collision: `((7,22),(9,4),W=26,H=22)` produced `4x5` and `5x5`.

Dominant charged tensors in the 171 anchor:
- `first_row_avg` `[1,10,1,1]` F32 = 40 B.
- `first_col_avg` `[1,10,1,1]` F32 = 40 B.
- `resize_roi` `[6,1,1]` F32 = 24 B.
- `bg_idx_i64_raw` `[1,1,1]` INT64 = 8 B.
- Remaining scalar F32 intermediates complete memory 156; params 15 = `source_const[1,2,2,2]` 8 + `sizes[3]` 3 + four scalar constants.

Rejected techniques:
- Single-projection collapse: wrong on generator; row and column segment counts vary independently.
- One-op/two-projection carrier: `[1,10,2]` or `[1,20,1,1]` is 20 F32 elements = 80 B, tying the two current 40 B projections and only saving uncharged node count.
- Row0 + active-size scalar: wrong on encodable generator collisions.
- Row0 + global per-channel counts: ties the 80 B vector floor and adds scalar/Gather plumbing.
- Free-output rectangle/Einsum tail: output is rank-1, but it still needs the same upstream projections/extents plus row/col/channel selector vectors, so it loses to the current Resize ROI tail.

No lesson drop created; this is a task021-specific re-confirmation of the existing floor, not a new transferable library result.
