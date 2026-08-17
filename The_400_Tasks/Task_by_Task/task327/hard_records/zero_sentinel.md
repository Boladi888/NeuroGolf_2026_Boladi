live bar: 268, sha 01ffc402949dce588c2e48584704325124f0355b57a7ffac798664e394b279eb
route: rewrite Sho's +1 painted-color sentinel to raw color-id space and retarget the final quadratic one-hot.
failed gate: `python tools/ng_gate.py visible --task 327 --cand The_400_Tasks/Task_by_Task/task327/prototypes/task327_zero_sentinel.onnx` -> `[visible] 0/265 (0.00%)  FAIL`
why dead: raw color 0 is both in-grid background and QLinearConv padded outside; the final ch0 polynomial leaks into the 24-padded rows/cols.
remaining hypothesis: keep the +1 sentinel, but shave optional zero-point plumbing or another scalar initializer around the same QLinearConv graph.
