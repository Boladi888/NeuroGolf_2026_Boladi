# task165 — bitthreshold active-column shave is ORT-dead

- Live/team-best bar: **1,641** (`memory=1,538`, `params=103`), SHA-256 `6faf9093ffe11c0fbe8583054205b9ee4ddc36b6f7c4d6c31222055a8d5b43a9`.
- Bar source: `python tools/ng_current_bar.py --task 165 --json` → authoritative manifest cost 1,641; workbook agrees; `Team_Best_Onnx/task165.onnx` has the same SHA.
- Attempted route: replace the 16-column F16 `Log/Mul/Cast` lowest-nonzero row path with `active_col = occupancy_u16 // (2 << lowest_arch_u8) > 0`, then log only the selected marker column.
- Prototype: `The_400_Tasks/Task_by_Task/task165/prototypes/task165_bitthreshold_castshift.onnx`, SHA-256 `e66c849583c994b775d6699c0a850d2b4ed10f9de99a06ffc43ff0c8c6d076cd`.
- Static price before ORT load: `memory=1,530`, `params=104`, apparent cost **1,634**.
- Exact failed gate: `python tools/ng_gate.py visible --task 165 --cand ...task165_bitthreshold_castshift.onnx` → `[visible] ERROR: NotImplemented ... Could not find an implementation for BitShift(11) node`.
- Why dead: ONNX shape inference requires the shift count to be UINT16, and ORT 1.24.4 has no `BitShift(11)` kernel. The no-cast U8-shift form fails shape inference: `Y has inconsistent type tensor(uint8)`. Pow/Div fallbacks reintroduce full 16-column casts/arithmetic and lose the 1,641 bar; U16 occupancy is also not a safe marker-row carrier for rows whose bit powers exceed UINT16.
- One remaining hypothesis: find an ORT-supported, narrow dynamic power-of-two threshold op; absent that, the live F16 log path is the legal floor for this subproblem.
