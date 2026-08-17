# task049 no-round height probe

- Bar used: manifest live/team-best cost 344, sha256 `d03189eb0da3a10254ce6ab2572e74e766a60b7c17e70dc0919b399f645ae71a`.
- Attempted route: remove `Round(height_sq)` and feed F16 `height_sq` directly to `Sqrt`, saving the 2-byte `rounded_height_sq` intermediate.
- Candidate: `prototypes/task049_no_round.onnx`, static/profile cost 342.
- Failed gate: `python tools/ng_gate.py visible --task 49 --cand The_400_Tasks/Task_by_Task/task049/prototypes/task049_no_round.onnx` -> `[visible] 180/268 (67.16%)  FAIL`.
- Why dead: F16 moment arithmetic can land below the exact square; without `Round`, `Sqrt` then `Cast(INT8)` underestimates height on visible cases.
- Remaining hypothesis: preserve rounding but shrink one of the 30-param row/col index initializers or the selected-color F32 cast path.
