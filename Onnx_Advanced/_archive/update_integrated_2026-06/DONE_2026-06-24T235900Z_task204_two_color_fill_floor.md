# Floor proof: task204 two-color hollow-square fill at 7682 (2026-06-24, Worker-10)

## What I discovered

Task204 (hollow blue squares, interior fill orange=odd size / red=even size) is at the provable cost floor of **7682** (memory=6652, params=1030). The current v2 architecture (`build_task204_v2.py`, manifest SHA `d2f77ac3e6bca0c1e5afdb88378da368ae17805bb70bb62651e5d9d10b150c75`) cannot be improved.

## Why it's at floor

The task requires **two-color fill by size parity** — orange for odd side lengths (3,5,7,9), red for even (4,6,8,10). This forces keeping 8 distinct length channels in the detector:
- No 1-channel value-coded scheme works: QLinearConv SUMS over the receptive field, so value-coding multiple lengths in one channel produces ambiguous sums when multiple squares are present.
- Crossing-parity (technique #50) gives binary interior/exterior but NOT per-square size parity. Separating H_odd and H_even crossing layers requires more intermediates than the 8-channel approach.
- Two-channel parity-only detector cannot paint without knowing actual size L (border placement depends on L, and interior color depends on L mod 2).

## Memory floor breakdown (6652B)

Every intermediate is load-bearing:
- [1,1,18,20] F32 = 1440B: mandatory F32 intermediate before Cast (input is F32)
- [1,1,18,20] U8 = 360B: reused by QLinearConv AND row Slice; cannot skip
- [1,8,18,18] U8 = 2592B: 8-channel detector output; minimum for 8 sizes
- 3× [1,1,20,20] U8 = 1200B: paint output, bg_code, label_code — all load-bearing
- [1,1,30,30] U8 = 900B: must pad before Equal (Equal-then-Pad order costs 2000B)
- Small row tensors = 160B

## Params floor (1030 elements)

- paint_w_u8 [1,8,10,10] = 800 elements: max square is 10×10, no smaller kernel possible
- topleft_w_i8 [8,1,2,12] = 192 elements: 2 rows (above-guard + current) × 12 cols (10 span + 2 guards)
- Slice/Pad index initializers = 24 elements (already shared where possible)
- Quantization scalars + color codes = 14 elements

## Traps confirmed

1. **Constant nodes for index tensors are WORSE than initializers**: Constant node outputs appear in `value_info` (memory = INT64 byte cost), while initializers are counted as params (element count only). Probe: +192B memory with Constant nodes (6844 vs 6652). Do NOT use Constant nodes for Slice starts/ends or Pad pads.
2. **Equal-then-Pad order is worse**: [1,10,20,20] BOOL = 2000B vs [1,1,30,30] U8 = 900B. Current Pad-then-Equal order is optimal.
3. **Opset-10 Pad attributes cannot save the 8-element label_pads13 initializer**: downgrading to opset 10 breaks UINT8 Max nodes (ORT does not support UINT8 Max at opset 10).

## Where I encountered it

- Task: task204
- Build script: `The_400_Tasks/Task_by_Task/task204/prototypes/build_task204_v2.py`
- Team-best ONNX: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task204.onnx` (cost 7682, SHA d2f77ac3...)
- Floor proof in task204.md under `## Floor proof (2026-06-24, Worker-10)`

## What's missing from the library

The cost_floor_reference.md does not have a task204 entry. Worth adding:
```
| task204 | topleft_len `[1,8,18,18]` U8 = 2592 | two-color fill by size parity forces 8 distinct channels; no simpler two-color parity technique exists | 7682 |
```

Also: the by_symptom.md symptom "two-color fill by enclosed object SIZE PARITY" could route to this finding. Currently technique #50 (crossing-parity) routes to single-color fill only.
