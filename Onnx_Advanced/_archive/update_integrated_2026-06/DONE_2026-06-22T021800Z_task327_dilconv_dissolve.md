# task327 — #98 dilated-Conv dissolves the F32 source crop in a ConvTranspose-stamp graph

## What I discovered
The team-best 881 graph for task327 (Aditya's ConvTranspose diagonal-staircase) had a
charged **`src_f32[1,9,3,3]` F32 = 324B** crop produced by an explicit `Slice` of the input,
fed to a 1×1 channel-collapse `Conv`. That crop is exactly the #98 case: a fixed top-left
sub-rectangle read as a channel-collapsed (color-index) map. I replaced `Slice`+`Conv` with
ONE dilated valid Conv reading the FREE `input`:

```
Conv(input[1,10,30,30], K[1,10,2,2], dilations=[27,27], pads=[0,0,0,0]) -> color_grid_f32[1,1,3,3]
K[0,ch,0,0] = ch   # tap (0,0) = the cell, all far taps zero
# dilation = L - k = 30 - 3 = 27 shrinks each 30-axis to 3 via the valid conv
```

This is the SAME #98 formula already documented (task046/task174/task383), but applied here
to a 3×3 crop on BOTH axes equally and feeding a downstream ConvTranspose. The 324B crop
vanished; cost **881 → 580** (memory 810→486, params 71→94). Byte-equal vs LIVE over 200,265
inputs, truth 200000/200000, profile_cost==static==580.

## Where
- Builder: `The_400_Tasks/Task_by_Task/task327/prototypes/build_task327_dilconv.py`
- Staged: `staged_submits/task327_dilconv_cost580/`
- Source-of-truth team-best: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task327.onnx` (re-measured 881, not the 934 staged Einsum).

## Why it matters
1. **#98 transfers to ConvTranspose-stamp graphs**, not just to pure crop+reduce tails. Any
   Aditya/Boladi graph that opens with `Slice -> Conv(channel-collapse)` on a fixed top-left
   sub-rectangle can drop the F32 crop with a single dilated Conv. The library's #98 examples
   are all post-conv color-index maps; this is the first case where the dilated read feeds a
   ConvTranspose diagonal stamp downstream.
2. **The #132 free-output Einsum (934) LOSES to the ConvTranspose stamp (580→) for this rule.**
   The (1,1)-shift Toeplitz over a 6-wide output axis is NOT low-rank enough — the rank-7
   `A[7,3,30]`=630 placement basis is the dominant cost. #132's own floor note ("diagonal/ray
   spreads are Conv/render floors, not separable floors", referencing task324) is correct here:
   ConvTranspose with a 6×6-identity kernel does the diagonal staircase as a translation-invariant
   stamp at near-zero param cost, beating the dense separable basis.
3. **Angle-1 emit dissolve (#116/#81) does NOT help.** Padding the 1-channel color-index plane
   to `[1,1,30,30]` F16=1800B before a final Equal-as-output is worse than the 360B 6×6 `small`
   one-hot + Pad. For a 6×6 output the small one-hot IS the #81 emit floor (confirms #81/#116
   size crossover: 6×6 = 36 cells < 90, small one-hot wins).

## What's missing from the library
#98 (techniques.md ~line 3891) could note: the dilated-read transfers to graphs whose crop
feeds a ConvTranspose/stamp, not only to terminal color-index maps. And #132's task327 entry
should record that the named-home ConvTranspose route (581/580) beats the Einsum (934) for the
diagonal-ray rule — the separable basis is the loser, the Conv stamp is the floor for this task.

## Rule note (was stale in task327.md)
ARC-GEN `task_d13f3404.py`: input colors one cell per diagonal `c-r`; each emits a down-right
ray `output[r+i][c+i]=color`. Rays never collide (unique diagonals), so ConvTranspose sum-overlap
is safe on real inputs but would corrupt color indices if the invariant were violated.
