# Selector replacement: 90→3 param Slice + 9-ch C_begin=1 Pad trick (task100)

## What I discovered

**Replacing a `[10,9]` F32 Einsum selector with a 3-param Slice**

For moment/marginal Einsums that use a `[C_in, C_out]` channel-selector matrix to map input channels 1..N to output channels 0..N-1 (a diagonal shift selector), you can:
1. Rewrite Einsum equations to contract over all C_in channels directly (drop `ck->k` subscript)
2. Compute everything in full C_in-dimensional space
3. Apply a single `Slice(score, starts=[1], ends=[C_in], axes=[1])` ONCE on the FINAL score vector

This replaces a `[10,9] F32` selector (90 params) with 3 INT64 scalar constants (3 params) for Slice. Net saving: 87 params. Penalty: the intermediates grow from [1,9] to [1,10] (each +4 bytes for F32; +8 intermediates × 4 = +32 bytes memory). One extra intermediate for score_full [1,10] before Slice.

task100 result: 460 → 441 with this change alone (remove 90-param selector, add 3-param Slice).

**The C_begin=1 Pad trick for 9-channel one-hot output in 10-channel space**

When the one-hot output covers channels 0..9 but the WINNING CHANNEL IS ALWAYS 1..9 (never 0), you can:
1. Build a 9-channel one-hot (channels 0..8 mapping to winning channel indices 1..9)
2. Pad with `C_begin=1` to shift all channels right by 1, inserting a False channel 0 at the front
3. Result: correct [1,10,30,30] BOOL one-hot with channel 0 = False (background = not selected)

This drops the 10-param `[1,10,1,1] INT64` channel_ids tensor (with values `[-1,0,1,...,8]`) and replaces it with a 9-param `[1,9,1,1] INT64` clean iota `[0,1,...,8]` + changes to `mp_pads` (same 8-param count, just different values).

Net saving from this tail change: 1 param + 5 bytes memory (smaller [1,9,1,1] and [1,9,2,2] tail tensors).

task100 combined: 460 → 435 (memory 409 + params 26).

## Gate results (task100 cost 435)

- visible: 266/266 ✓
- truth-full: 200000/200000 ✓
- ab-full: 200266/200266 byte_equal=true, divergences=0 ✓
- score_v2: macs=0, memory=409, params=26, cost=435
- profile_real_cost: static=435 profile=435

## Why it matters

- Replaces diagonal-shift selector matrices (common for "filter channels 1..N from all-channel input") with Slice
- Eliminates the -1 offset in channel_ids tensors ([-1,0,...,8]) via C_begin=1 Pad
- Generalizes: any time there's an "equal-and-compare" output tail where the result channels are offset by +1, the C_begin pad trick saves 1 param + some bytes

## Where encountered

- task: task100 (hollow-rect AREA via moment identity)
- builder: The_400_Tasks/Task_by_Task/task100/prototypes/build_task100_slice_score_9ch.py
- staged: staged_submits/task100_slice_score_9ch_cost435/

## What's missing from the library

Checked techniques.md §128 (Boladi task100) — doesn't mention the selector-→-Slice replacement.
Checked by_symptom.md — no entry for "diagonal-shift selector" pattern.

The C_begin Pad trick is new and not in the library.

## Sorter note: techniques.md §128 — extend with the selector-→-Slice approach and C_begin=1 Pad trick
## Sorter note: by_symptom.md — add symptom "diagonal-shift selector matrix (N+1 to N channels)" → Slice-last + C_begin Pad
