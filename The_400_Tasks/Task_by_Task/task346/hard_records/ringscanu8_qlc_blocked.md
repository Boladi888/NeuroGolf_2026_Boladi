# task346 RingScanU8/QLinearConv blocked route

Live bar used: 475 bytes, SHA-256
`2dc0342caab3b34b468b7864f04920337b0e45d29af042eda9c5b53eff1b9a77`, from
`python tools/ng_current_bar.py --task 346 --json` / manifest owner Aditya.

Attempted route: replace the charged `ringval` tensor (`[1,1,8,8]` F32 = 256 bytes) in the live
high-color ring detector with a U8 `QLinearConv` scan.

Exact failed gate: pre-build feasibility/price gate. `QLinearConv` requires INT8/UINT8 input
(`ort_reality.md` and techniques #137). The NeuroGolf graph input is fixed F32. A direct F32
`QLinearConv(input, ...)` is schema/ORT invalid. The legal bridge variants are already losing:

- `Cast(input -> U8)` materializes `[1,10,30,30]` U8 = 9000 bytes before the scan.
- `Slice(input 10x10) -> Cast(U8) -> QLinearConv` materializes at least `[1,10,10,10]` F32
  = 4000 bytes plus `[1,10,10,10]` U8 = 1000 bytes before the 64-byte U8 scan.

Why dead: the intended 256-byte deletion is dominated by the required F32-to-U8 bridge. There is no
existing narrow feed in the 475 graph; the only cheap producer of the scan is the F32 Conv reading
the free graph input directly.

Remaining hypothesis: only scalar/metadata polish remains unless a new formulation avoids the
64-window spatial scan entirely. Worker3 found one such scalar polish, Pad-18 axes, at prototype
`The_400_Tasks/Task_by_Task/task346/prototypes/task346_v5_pad18.onnx` with cost/profile 473, but
left it unstaged pending full gates because the heavy gate slot was occupied by other workers.

