# DONE — 4-axis Slice omits `axes` input, saves 1 param (task350, 2026-06-22)

## What was discovered

When `Slice` is given a full-rank `starts`/`ends` covering all 4 axes of the input tensor
(batch, channel, H, W), the `axes` input can be OMITTED because the default is `[0, 1, ..., r-1]`.
This saves 1 initializer (3 INT64 elements = 3 params in NeuroGolf accounting).

**Before (3-axis form):**
```
Slice(input, ch1_st=[1,0,0], ch1_en=[2,26,24], axes_chw=[1,2,3])
# 9 params: 3+3+3
```

**After (4-axis form):**
```
Slice(input, ch1_st=[0,1,0,0], ch1_en=[1,2,26,24])
# 8 params: 4+4, default axes=[0,1,2,3]
```

Both produce the same F32[1,1,26,24] output (channel 1, rows 0..25, cols 0..23).

## Where encountered

- Task: task350 (`staged_submits/task350_slice2_cost9036/task350.onnx`)
- Builder: `The_400_Tasks/Task_by_Task/task350/prototypes/build_task350_slice2.py`
- Prior best: `padaxes` at 9,037 (memory=9,012, params=25)
- New: `slice2` at 9,036 (memory=9,012, params=24)

## Why it matters

This is a 1-param saving that beats a prior team best. The key insight:
- When slicing axes=[1,2,3] (not 0), the 3-axis form needs 3+3+3=9 params.
- When you include batch axis (start=0, end=1 = keep all batch), the 4-axis form needs only 4+4=8 params.
- The "include batch axis as no-op" trick avoids the `axes` initializer entirely.

**Rule of thumb:** If slicing channels/H/W but NOT axis 0 (batch), check whether extending to
include axis 0 (start=0, end=input_batch_size) is cheaper: 4+4=8 params < 3+3+3=9 params.
If slicing only 1 or 2 axes and specifying axes explicitly is cheaper than a 4-axis no-op spec,
keep the explicit axes form.

## When to apply

- Slice with axes=[1,2,3] and all 4 axes of the tensor need covering in starts/ends anyway
- The task needs to crop both channel and spatial dims in one Slice
- Opset 10+ (Slice input form)
- Confirmed working on ORT 1.24.4, opset 18

## What's missing from the library

Techniques #105 and §scoring_internals discuss param minimization, but there's no explicit note
that "4-axis Slice (with batch axis as no-op) can save 1 param vs 3-axis explicit-axes form."
This could be a bullet in `cost_floor_reference.md` under param minimization examples, or noted
in `techniques.md #105` residual register.

Gates verified: visible 267/267, truth 200000/200000, AB 200267/200267 byte-equal, profile=static=9,036.
