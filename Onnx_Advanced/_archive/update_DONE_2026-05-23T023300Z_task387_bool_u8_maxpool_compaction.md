# DONE 2026-05-23T023300Z - task387 BOOL/UINT8 MaxPool compaction

## Result

- Prototype: `The_400_Tasks/Task_by_Task/task387/prototypes/task387_boolinput_maxpool_compactbbox_andreorder_fgmax_xor_activemax_nocast43_u8out_polished.onnx`
- Cost: 100,793 -> 44,917 (memory 44,800 + params 117)
- Validation: `verify_against_live --tid 387` = 266/266 proto-vs-LIVE
- Advisory synthetic check: `adversarial_against_live --n 2000` = 2000/2000
- No Kaggle submission; `submissions/submission.zip` was not edited.

## Technique

This is a structural rewrite, not an algorithm replacement. It preserves the live graph's predicates but changes the plumbing:

- Use BOOL masks for one-hot channel slices, but keep numeric paths as UINT8 where opset-12 needs numeric tensors.
- `Pad` does not accept BOOL here, so final output remains UINT8: cast final one-channel masks to UINT8, `Concat`, then `Pad`.
- Replace binary F16 `Conv(3x3 ones) -> Greater` dilation with `BOOL -> UINT8 -> MaxPool(3x3 pads=1) -> Greater`.
- Reduce bbox min/max on skinny row/column presence first; avoid 18x18 F16 `Where`/`Mul` tensors when 18x1 or 1x18 is sufficient.
- Replace `active_foreground AND NOT(color_mask)` with `Xor(active_foreground, color_mask)` when the color mask is a proven subset of the active foreground mask.
- Reuse per-color UINT8 presence casts for active-foreground folding: variadic `Max(c1_u8, ..., c9_u8) -> Greater` replaces a long BOOL `Or` chain.
- Reuse already-cast final UINT8 output channels for final foreground occupancy: variadic `Max` replaces another BOOL `Or` chain before background construction.

## Generalization Notes

Good targets:
- bool-heavy one-hot ARC graphs with repeated F16 dilation convs over binary masks.
- graphs where `Conv` weights are all ones and the only consumer is `Greater(>0)`.
- graphs with row/col bbox reductions that broadcast skinny coordinates to HxW before reducing back to scalars.
- output tails where BOOL is semantically enough but `Pad` forces a numeric dtype.

Checklist:
- Verify `Pad(bool)` before trying a BOOL output; in this runtime it fails.
- Do not use UINT8 `Sum`; ORT rejects it. Use BOOL `Or`, UINT8 `Max`, or `ReduceMax` instead.
- Do not replace safe background logic with arithmetic subtraction unless you have proved foreground is a subset of the active grid on adversarial raw-live inputs.
