task043 hard record - 10x10 output crop for live Einsum

Live bar:
- cost 84
- sha256 1e0d69485acbed45aafb3f94ab63b182f635896031144d8fb7fa9865c6c8ad01
- source: `python tools/ng_current_bar.py --task 43 --json` (manifest authoritative)

Attempted route:
- Crop the live free-output Einsum to declared `[1,10,10,10]` output and shorten `top_row`/`right_col` selectors from 30 to 10.
- Prototype: `The_400_Tasks/Task_by_Task/task043/prototypes/task043_live_einsum_10x10_probe.onnx`
- Prototype sha256: 2fbaea4711f7450ecaa531022bf5e34159c6b6a9205ecb9a474062d08b1e6372

Exact failed gate:
- `python -c "from tools._score_v2 import score_network_v2; ..."` returned `score None None`.
- Shape inference failed because the input-carried output axes still infer as 30:
  `Inferred shape and existing shape differ in dimension 2: (30) vs (10)`.
- ORT also rejected the shortened selector operand:
  `operand 4 ... shape is: {10}; another operand has dim value of 30 in the same dimension`.

Why this route is dead:
- The live Einsum carries output row/col labels directly from the 30x30 input operands. Cropping output to 10x10 would require charged row/col crop selector matrices or a pre-slice, which costs far more than the 40 params saved by shortening two selectors.
- No visible/truth/profile gates were run because the model is scorer-invalid.

One remaining hypothesis:
- A sub-30 output could only help if the whole contraction is rewritten around a different low-param output-axis basis; the live-axis shrink alone is invalid.
