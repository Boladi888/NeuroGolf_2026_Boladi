# DONE 2026-05-22T151500Z - task069 structural crop10 FP16

## Result

Built a Task069 candidate below the prompt target:

- Builder: `The_400_Tasks/Task_by_Task/task069/prototypes/build_task069_structural_crop10_fp16.py`
- ONNX: `The_400_Tasks/Task_by_Task/task069/prototypes/task069_structural_crop10_fp16_notcnz_pad30.onnx`
- Local package: `The_400_Tasks/Task_by_Task/task069/prototypes/task069_structural_crop10_fp16_notcnz_pad30/submission.zip`
- Manifest: `The_400_Tasks/Task_by_Task/task069/prototypes/task069_structural_crop10_fp16_notcnz_pad30/manifest.json`

Cost:

- Current canonical: 127,412
- Equal-fold opset-11 probe: 126,565
- Structural crop10 FP16: 22,906 = memory 22,762 + params 144

Validation:

- `check_network=True`
- `verify_against_live.py --tid 69`: 264/264 expected, 264/264 live, 264/264 proto==live
- `adversarial_test.py --max-perturbations 2`: 4,033/4,224 perturbation parity vs live = 95.5%, worst bucket rotate = 87.3%
- Distribution-faithful structural synthetic-LIVE: 500/500 separated-template generated inputs matched current live
- Local package has 401 entries and changes only `task069.onnx`

## Technique

The preserved compact structural external graph expressed the rule directly but ran the whole template-stamping pipeline over 30x30 tensors, scoring 313,491 despite a tiny file size. The safe candidate:

- crops the channel-8 mask and non-background channels to Task069's 10x10 active area;
- shrinks the structural template normalization, match convolution, and stamp transpose-convolution to 10x10;
- preserves the public-positive attempt14 raw output contract with a final 30x30 Pad;
- replaces foreground/background channel Pad+Add with a direct Concat;
- drops redundant foreground Clip and nonzero-template gating for valid task-shaped inputs;
- runs internal structural math in FP16, with a final Cast back to F32 before Pad.

## Safety Note

This is not the failed attempt9/attempt13 no-final-Pad family. It emits `[1,10,30,30] -> [1,10,30,30]` and keeps all non-active output padding zero. The unchanged compact external graph failed broad random synthetic-LIVE, so use this crop10 descendant and its Task069-shaped synthetic evidence for review.
