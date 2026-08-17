# DONE: dense 10-out Conv is the floor when the rule is cross-channel + 10-channel output (task095)

**Topic:** cost-floor reasoning / when a dense `Conv[10,10,3,3]=900` is genuinely optimal and the crop+small-one-hot route loses.

**Context:** task095 = "keep gray markers (ch5), paint a blue (ch1) 8-neighbor ring, bg=ch0". Live/team-best is a single dense `Conv[10,10,3,3]` = 900 params, memory 0. The dense Conv "looks wasteful" (only 26 nonzero weights). It is NOT — it is the floor.

## Two-family floor argument (reusable for any single-Conv "looks sparse" task)

**Family A — single-op floor.** A memory-0 graph is a single op `input→OP→output` (any 2nd op = a charged non-I/O intermediate). If the rule is **cross-channel** (here ch5→ch0 and ch5→ch1, read off the live weights: nonzero (out,in) taps = 10 diagonal identities + (0,5) + (1,5)) AND the output needs all 10 channels, then:
- out_dim = 10 forced (10-channel one-hot output).
- the cross taps force the source + destination channels into one conv *group*; if those channels are **non-contiguous** (0,1,5 here), no grouped Conv/ConvTranspose can group them without a Transpose/Gather (= an intermediate). So groups=1, weight `[10,10,3,3]` = 900. ConvTranspose `[in,out/g,k,k]` = same.
- params count *element count*, not nonzero count — sparsity buys nothing (sanitizer rejects sparse Conv weights too).
- **=> no single op < 900.**

**Family B — crop + small-one-hot (#81 small-grid inversion / #67 free-output) loses.** The natural sub-900 attempt: crop to the active grid (ARC-GEN `task_4258a5f9.py` hardcodes size=9, markers inboard ⇒ always 9×9), build a value-index plane, `Equal(value, ramp)` → small one-hot, `Pad`-as-output. Built it, it is **fully correct** (visible 265/265, truth 200000/200000) but costs **1162** (static==profile). Two costs are structurally forced and already exceed 900:
- **f32 channel-slice = 324 B**: extracting one input channel as a small plane is `Slice(input,ch,0:9,0:9)` → `[1,1,9,9]` **F32** (Slice preserves dtype; Gather can't crop spatially; casting the whole input first = 9000 B).
- **channels-0..5 one-hot span = 486 B**: output nonzero channels are 0,1,5; `Pad`-as-output extends *contiguously*, so the pre-pad tensor must span ch0..5 = `[1,6,9,9]` (486 B BOOL/U8). Putting ch5 content at 30×30 to Concat instead = a 900-B `[1,1,30,30]` plane.
- 324 + 486 = 810 forced, + ≥2 ring-compute planes (~162) ⇒ **~970 floor > 900**.

## Transferable rules

1. **Before attacking a "looks-sparse" single dense Conv, read the nonzero (out,in) taps.** If they are cross-channel AND the destination/source channels are non-contiguous AND the output is full-10-channel, the dense `[10,10,3,3]=900` is the single-op floor — sparsity, depthwise, and grouped routes all fail. Don't spend a session; write the floor proof.
2. **#81 small-grid inversion / crop+one-hot only wins when the *whole* pre-pad assembly (f32 extraction + contiguous channel span + compute) is < 900.** For a rule whose nonzero output channels straddle a wide non-contiguous span (e.g. 0..5), the contiguous-`Pad` constraint inflates the pre-pad tensor to that full span (here 6×81=486), and the forced F32 channel-slice (324) on top means you cross 900 before computing anything. Price `324 + (max_nonzero_channel_index+1)*active_cells + compute` first.
3. **The forced-F32-slice tax (324 B at 9×9):** any single-channel extraction from the F32 one-hot input is F32 (Slice/Gather preserve dtype); the U8 cast is a *second* op. Budget it.

## Verdict
task095 stays at 900 (live dense Conv). No-candidate. Correct-but-overcost artifact: `Task_by_Task/task095/prototypes/task095_crop_lean.onnx` (SHA `7ce4b5525b144ff2e8be473b9d26beca55197668e0cf52879e2897aaf9b1c865`).
