# task174: Conv color-index map dissolves 3 F32 dynamic Slice intermediates (-769 cost)

## What was discovered

A dilated 1-output-channel Conv with integer color-index weights dissolves 3 F32 dynamic Slice
intermediates. The graph had: `Slice(input, [0,k_idx,0,0],[1,k_idx+1,10,10])` -> F32 [1,1,10,10]
(400B) × 3 = 1,200B for TopK-selected color plane masks. These were the dominant charged tensors
after the 900B output canvas.

**Replacement pattern:**
1. `Conv(input, W[1,10,2,2], dilations=[20,20])` -> `labels_f32` F32 [1,1,10,10] (reads FREE input)
2. `Cast(labels_f32, U8)` -> `labels_u8` U8 [1,1,10,10] (100B)
3. Per-k: `Cast(k_color_idx, U8)` -> `k_color_u8` (scalar), then `Equal(labels_u8, k_color_u8)` -> `k_mask_b` BOOL

Kernel design: `W[0,c,0,0] = c` for c=0..9 (color index weight at tap (0,0)). All other taps zero.
Dilations [20,20]: effective span = (2-1)*20+1 = 21, output size = 30-21+1 = 10. This simultaneously
crops the 30->10 spatial axis AND reduces the 10 color channels to a scalar color index. Since input
is one-hot, the Conv output = color index at each cell (0=background, 1-9=colors).

## Where encountered

- Task 174, cost 4786 -> 4017 (-769B, -16.1%)
- Builder: `The_400_Tasks/Task_by_Task/task174/prototypes/build_task174_conv_coloridx.py`
- Staged: `staged_submits/task174_conv_coloridx_cost4017/`
- All gates green: visible 266/266, truth 200k/200k (100%), AB byte-equal 200266/200266, profile=static=4017

## Why it matters

The existing don't-do note (2026-06-12) said "the F32 input-read tax is structural for dynamically-chosen
channels from an F32 input." This was WRONG. The cost_floor_reference.md headline correction (task271
CARVE-OUT, 2026-06-19) establishes: "Conv reads the free input and pays only OUTPUT bytes." The 3×400B
F32 Slices were not structural at all — they were just a materialization choice.

Net savings:
- Memory: -1,200B (3 F32 masks) - 108B (starts/ends plumbing) + 400B (labels_f32) + 100B (labels_u8) + 3B = -805B
- Params: +40 (Conv kernel [1,10,2,2]) - 4 (slice_axes4_i32 removed) = +36
- Net cost: -769B

## Transferable pattern

Whenever a graph selects N channels from a one-hot F32 input via DYNAMIC Slice/Gather and the downstream
consumers only need boolean/U8 membership, consider:
1. Use a dilated Conv with integer channel-weights to compute a single U8 color-index map (#98).
2. Replace each dynamic Slice with Equal(color_index_map, dynamic_scalar). 
3. The Conv pays only its output bytes (output is small due to dilation crop); the 3 large F32 planes vanish.

Cost tradeoff: +40 params (Conv kernel) for -1,200B memory on 3×F32[1,1,10,10] planes = very profitable.

For non-one-hot inputs or >10 channels: adjust weights accordingly. The dilation formula is dilation = L-k
where L is the input axis length and k is the desired output length (technique #98).

## Library cross-references

- cost_floor_reference.md: "F32 input-read tax" section - the task271 CARVE-OUT says Conv output is the only charge
- techniques.md #98: dilated spatial-crop + channel-reduce in ONE conv
- task271 repriced from 1942->1136 using the same insight

## What was missing from the library

The don't-do trap that was written: "the F32 input-read tax is structural for dynamically-chosen channels
from an F32 input" is the opposite of the truth when using a Conv. The library correctly documents the
carve-out in cost_floor_reference.md, but the task's own don't-do log was stale. Workers who see a
don't-do about "F32 input-read tax" on dynamic channel selection should check if a Conv approach would
bypass it entirely.
