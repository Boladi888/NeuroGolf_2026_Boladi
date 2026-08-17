# task165 — narrow ArgMax selector shaves are ORT-dead

- Live/team-best bar: **1,641** (`memory=1,538`, `params=103`), SHA-256 `6faf9093ffe11c0fbe8583054205b9ee4ddc36b6f7c4d6c31222055a8d5b43a9`.
- Bar source: `python tools/ng_current_bar.py --task 165 --json` → authoritative manifest cost 1,641.
- Attempted route: avoid the live graph's 20-byte `kite_match_unmasked UINT16 -> F16` vector cast by running `ArgMax` on the UINT16 or BOOL match vector and casting only the selected scalar for `Log`.
- Prototype: `The_400_Tasks/Task_by_Task/task165/prototypes/task165_argmax_u16.onnx`, SHA-256 `6b941cf7bbbc6c6ed4e8ae8b50d29dcfa62a474d2440060f4a464ed6909140ce`.
- Static price before ORT load: `memory=1,520`, `params=103`, apparent cost **1,623**.
- Exact failed gate: `python tools/ng_gate.py visible --task 165 --cand ...task165_argmax_u16.onnx` → `[visible] ERROR: NotImplemented ... Could not find an implementation for ArgMax(13) node`.
- Additional probe: BOOL selector variant fails ONNX shape inference: `(op_type:ArgMax, node name: select_left): data typestr: T, has unsupported type: tensor(bool)`.
- Why dead: ORT/scorer require the live F16 vector cast for this ArgMax. Direct `UINT16 -> UINT8` is unsafe because ORT wraps powers (`256 -> 0`, `512 -> 0`), so a U8 selector can silently erase valid high-row matches.
- One remaining hypothesis: if a future ORT version supports ArgMax over unsigned integers, this is a real ~18-byte shave; under the pinned ORT it is not reviewer-ready.
