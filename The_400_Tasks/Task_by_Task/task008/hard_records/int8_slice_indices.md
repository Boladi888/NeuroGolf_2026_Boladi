# int8_slice_indices

- live bar: manifest live 2040, SHA `8c7098348625ec740dfbc82025dbb68e6219a2a4e26b341304b0294c2fc13c30`; effective local staged bar 1972, SHA `3994a9ea1fde1f80e76d69cce902da629d9f3d8c4fa8f215b687e13c12637c31`.
- attempted route: remove the two INT8->INT32 Slice-boundary casts by feeding INT8 starts/ends to the dynamic Slice.
- failed gate: `visible` load failed. Mixed constants failed at `Concat` with INT32/INT8 type mismatch; all-INT8 starts/ends failed with `InvalidGraph: Type 'tensor(int8)' of input parameter ... of operator (Slice) is invalid`.
- why dead: ORT Slice requires INT32/INT64 starts/ends even though the ONNX checker did not catch the narrower index type before ORT load.
- remaining hypothesis: further savings must avoid dynamic Slice index tensors entirely or change the output construction; do not retry sub-INT32 Slice indices.
