# task055 retry: under-1000 wall check

Date: 2026-06-20

## Verdict

No valid sub-1000 candidate was found in this retry.

Follow-up retry produced a valid small improvement at cost 1113:
`../task055_valid_cost1113_axis_shape_share_20260620/task055.onnx`.

The best valid candidate at the time of this wall probe was the previously staged v13 bitset/scatter-active graph:

- path: `../task055_bandselector_sectionid_cost1115/task055.onnx`
- cost: 1115 = memory 1024 + params 91
- gates from prior staged run: visible 263/263, truth 200000/200000, A/B 200263/200263 byte-equal

## Measured under-1000 probe

`do_not_submit_task055_noactive_pruned_cost994.onnx`

- score: `(macs=0, memory=904, params=90)`, cost 994
- visible gate: 0/263
- sha256: `E2A0A8B6E503993C52D60304214394132309F1CA11F76AAC3B17E77DC9E8EF61`
- status: failing probe only, not a submission candidate

This probe removes both per-axis active repair gates from v13 and prunes the now-unused zero initializer. That saves 120 bytes of memory and 1 param, which is enough to reach 994, but it lets bottom/right section codes leak into padded rows and columns. Since NeuroGolf compares the full one-hot 30x30 tensor, padding must stay all-zero. Background/color in padding is still wrong.

## OnnxAdv-inspired routes checked

- #141 axis-bitset intersection: already the current v13 architecture. Final expansion is at the 330-byte bitset floor: 300-byte channel-expanded axis operand plus 30-byte opposite axis operand.
- #119 free-output separable band selector: a direct final Einsum would need triangular relation data for `h < divider` / `w < divider`. The dense 30x30 triangular table alone is 900 params before input sample selectors and color tables, so it does not give a credible sub-1000 route. Sparse initializers are blocked by the documented scorer failure.
- #122 MatMulInteger / quantized route: blocked by the F32 graph input. A full input cast/quantization costs about 9000 bytes before it can feed integer matmul/conv.
- #142 single-color one-hot dissolve: not applicable; task055 has five fill colors plus divider/background behavior.
- CumSum/section-id route: U8 CumSum is not supported; F16/INT32 CumSum variants price above the ArgMax/iota locator used by v13.

## Cost wall note

The only simple way under 1000 is dropping the active repair gates:

- v13 correct: 1024 memory + 91 params = 1115
- no-active pruned: 904 memory + 90 params = 994, but visible 0/263

Those two active repairs cost exactly 120 memory total and are correctness-critical. A replacement still needs to distinguish active rows/cols from padded rows/cols and zero the axis code before final channel expansion. The current `Cast(s8 -> BOOL)` plus `Where(active, code, 0)` is already the 30-byte mask plus 30-byte repaired-vector floor per axis.
