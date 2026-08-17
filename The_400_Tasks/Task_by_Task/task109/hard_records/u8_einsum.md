# task109 hard record - U8 Einsum route

- Live/team-best bar used: cost 1255, profile_cost 1255, sha256 `56a1dd1b9b4658b02cd2a642012f3ec7857b6fe1029a464c6ff1246808776a4f` from workbook/manifest refresh and task-local staged manifest.
- Attempted route: keep the 1255 free-output topology, convert `line_vec_h`, `chan_terms`, `spatial_terms`, and `projC_30` from F16 to UINT8, and declare UINT8 output.
- Failed gate: `python tools/ng_gate.py visible --task 109 --cand The_400_Tasks/Task_by_Task/task109/prototypes/task109_u8_einsum.onnx`.
- Exact result: `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Einsum(12) node with name 'output'`.
- Why dead: ONNX schema admits integer Einsum, but ORT 1.24.4 lacks the kernel; all integer-Einsum dtype variants share this load-time failure surface.
- One remaining hypothesis: a non-Einsum final renderer that keeps output free while avoiding the `[6,30]` F16 projection charge; direct Pad/Concat active-canvas render is expected to lose.
