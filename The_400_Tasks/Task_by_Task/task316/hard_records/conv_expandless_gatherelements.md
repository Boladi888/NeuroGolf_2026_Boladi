# task316 hard record - expandless conv GatherElements

- Live bar: 426, SHA `243902b766c5356dd5974b29d3e508fba11d9999b00b0d5965ee20b23e9e3d4b`.
- Attempted route: use negative-pad Conv to emit `[1,1,1,10]` column scalars, then GatherElements directly into `[1,1,3,3]` snake order without a squeeze/reshape.
- Failed gate: score/profile. `task316_conv_topk_f16.onnx` returned `score_network_v2 = None`; ORT reported `GatherElements op: 'indices' shape should have values within bounds of 'data' shape. Invalid value in indices shape is: 3`.
- Why dead: GatherElements cannot expand a non-axis dimension; indices `[1,1,3,3]` are invalid against data `[1,1,1,9]`.
- Remaining hypothesis: the valid squeeze route is staged at cost 415; any further route must delete the Squeeze/TopK values/index cost, not repeat expandless GatherElements.
