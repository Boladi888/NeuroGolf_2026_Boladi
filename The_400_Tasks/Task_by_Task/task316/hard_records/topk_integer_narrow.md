# task316 hard record - integer TopK narrow

- Live bar: 426, SHA `243902b766c5356dd5974b29d3e508fba11d9999b00b0d5965ee20b23e9e3d4b`.
- Attempted route: replace the current F16 TopK key with direct I8/U8 TopK to delete the 20 B key cast and shrink the 18 B values output.
- Failed gate: score/profile. `task316_cur_topk_i8.onnx`, `task316_cur_topk_u8.onnx`, `task316_conv_topk_i8.onnx`, and `task316_conv_topk_u8.onnx` all returned `score_network_v2 = None`.
- Why dead: ONNX schema accepts integer TopK, but the local ORT/scorer stack does not provide a usable kernel for these integer TopK variants.
- Remaining hypothesis: INT16/UINT16 TopK may load but cannot beat F16 because the key/value tensors are the same byte width as F16.
