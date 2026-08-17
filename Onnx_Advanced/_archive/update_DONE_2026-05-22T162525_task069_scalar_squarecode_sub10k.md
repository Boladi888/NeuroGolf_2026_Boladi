# Task069 scalar squarecode sub10k

Task069 now has a local sub-10k scalar-code prototype:

- Builder: `The_400_Tasks/Task_by_Task/task069/prototypes/build_task069_scalar_squarecode_sub10k.py`
- ONNX: `The_400_Tasks/Task_by_Task/task069/prototypes/task069_scalar_squarecode_sub10k_bool_pad.onnx`
- Cost: 9,999 = memory 9,951 + params 48
- `check_network`: True
- Visible/live full-tensor parity: 264/264
- Color-relabel synthetic-LIVE: 500/500
- Standard adversarial advisory: 6711/7057 = 95.1%; worst perturbation rotate = 83.1%

Technique: collapse one-hot colors to a scalar color-index plane, crop to the
10x10 active area, use squared FP16 color codes to avoid some accidental overlap
sums, match channel-8 marker placements with a 19x19 Conv lattice, stamp with
ConvTranspose, decode with Equal against squared channel constants, and Pad the
BOOL one-hot output back to `[1,10,30,30]`.

Risk note: output dtype is BOOL instead of the incumbent F32 ABI. The graph
keeps the final 30x30 Pad contract and is not the public-negative no-final-Pad
10x10-output family, but it should remain review-only until the dtype change is
explicitly approved. No `submission.zip` was edited and no Kaggle submit was
made.
