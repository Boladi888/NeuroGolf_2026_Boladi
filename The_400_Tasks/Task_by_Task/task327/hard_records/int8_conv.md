live bar: 268, sha 01ffc402949dce588c2e48584704325124f0355b57a7ffac798664e394b279eb
route: replace both QLinearConv nodes with plain INT8 Conv+bias to delete QLinearConv scale/zero-point scalar initializers.
failed gate: `python tools/ng_gate.py visible --task 327 --cand The_400_Tasks/Task_by_Task/task327/prototypes/task327_int8_conv.onnx` -> `[visible] ERROR: InvalidGraph ... Type 'tensor(int8)' ... of operator (Conv) ... is invalid.`
why dead: ORT 1.24.4 CPU rejects standard Conv on INT8 tensors, so the apparently legal checker/shape-inferred graph is unshippable.
remaining hypothesis: QLinearConv is the legal INT8 render path; any win must trim scalar plumbing or avoid the square/Concat carrier without using standard INT8 Conv.
