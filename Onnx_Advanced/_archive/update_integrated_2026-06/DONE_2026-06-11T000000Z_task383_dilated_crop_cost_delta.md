# task383 #98 dilated crop-conv transfer — confirmed cost delta

## What I discovered

Technique #98 (DILATED-KERNEL SPATIAL-CROP + CHANNEL-REDUCE) applied to task383 v5:
- Replaced the 3-op 30→24 color-index bridge (`Conv 1×1 → Cast → Slice`) with ONE dilated `Conv K[1,10,2,2] dilation=6` that crops 30→24 on both axes and reduces 10→1 channel simultaneously.
- Kernel: tap (0,0) per channel = color ramp [0..9]; taps (0,1), (1,0), (1,1) = 0. Dilation=6: effective receptive field = (2-1)·6+1 = 7; output = 30-7+1 = 24. Both axes symmetric.
- Byte-equal: far taps have weight 0 → dilated conv computes identical `Σ_ch ch·input[ch,r,c]` as the old 1×1 conv at positions (r,c) ∈ 0..23.

## Where encountered

- Task: 383, Worker 1, session 2026-06-11
- Build script: `The_400_Tasks/Task_by_Task/task383/prototypes/build_task383_v6.py`
- Staged: `staged_submits/task383_dilated_crop_conv_v6_cost10942/task383.onnx`

## Why it matters — the actual measured cost delta

The prior #98 validation (task046) showed the technique works. This confirms the cost delta on a symmetric 30→24×30→24 crop (both axes same dilation=6):

| | old v5 | new v6 | delta |
|---|---|---|---|
| `cidx_full` F32 [1,1,30,30] | 3,600 B | — | −3,600 |
| `cidx_u30` U8 [1,1,30,30] | 900 B | — | −900 |
| `cidx_f24` F32 [1,1,24,24] | — | 2,304 B | +2,304 |
| `cidx_u` U8 [1,1,24,24] | 576 B | 576 B | 0 |
| `color_w` params [1,10,1,1] | 10 | — | −10 |
| `dil_w` params [1,10,2,2] | — | 40 | +40 |
| `crop_e` params [4] | 4 | — | −4 |
| **total memory** | 13,052 | 10,856 | **−2,196** |
| **total params** | 60 | 86 | **+26** |
| **total cost** | 13,112 | 10,942 | **−2,170** |

Gates: visible 266/266; ab --full 200266/200266 byte_equal=True; truth --full 200000/200000 (100%); profile==static==10942.

## Key calibration for #98 transfers

For a symmetric k→k' crop on a `[1,10,k,k]` input:
- Old bridge: `cidx_full` F32 [1,1,k,k] = 4k² + `cidx_u30` U8 [1,1,k,k] = k² bytes (both counted as intermediates)
- New bridge: `cidx_f24` F32 [1,1,k',k'] = 4k'² bytes only (cidx_u stays)
- Net memory saving: 4k² + k² - 4k'² = 5k² - 4k'²
- For 30→24: 5×900 - 4×576 = 4500 - 2304 = 2,196 B ✓
- Params overhead: K[1,10,2,2]=40 vs K[1,10,1,1]=10 = +30, minus the crop_e[4] = +26 net

## What's missing from the library

§98 already documents the technique with the task046 example (30→3, 30→20 asymmetric crops). The symmetric 30→24 case confirms the formula scales correctly. No update needed to the technique definition — this is purely a confirmation/calibration data point.

The floor proof in `cost_floor_reference.md` may have a stale 13k estimate for task383. The actual post-#98 floor is ~10.9k.

## Sorter note: cost_floor_reference.md — extend floor-verdict ledger entry for task383 (13,112 → 10,942 via #98; formulation-relative lower bound now ~10,884 B)
