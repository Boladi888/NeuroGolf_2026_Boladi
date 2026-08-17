# cropped_output_shape

Live bar: cost 225, sha256 `d3ed7d76188be734565d1620ae49ffc6fdc819b145d228e5ca602497113dcb80`, source `python tools/ng_current_bar.py --task 60 --json` (manifest, owner Aditya, updated 2026-06-12T18:26:48Z).

Attempted route: use the prompt's cropped-output idea by shrinking the final output width factor to 11 in a one-node endpoint-broadcast `Einsum`. Static cost is 168 because `outcol_w` becomes `[3,11]`; candidate `The_400_Tasks/Task_by_Task/task060/prototypes/task060_cropped_width_probe.onnx`, sha256 `b8f09eafdb15e663879328e2d954c19af8afecdf2011099ae63101dc47897b7a`.

Exact failed gate: `python tools/ng_gate.py visible --task 60 --cand The_400_Tasks/Task_by_Task/task060/prototypes/task060_cropped_width_probe.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.

Why dead: the shared `ng_gate` harness compares raw decoded tensors against official `[1,10,30,30]` encodings. A narrower graph output is shape-incompatible even when the active 5x11 content is the intended rule. Padding a cropped tensor back to 30x30 would charge at least the cropped one-hot intermediate and is far above the 225 params-only live graph.

One remaining hypothesis: only revisit if the official verifier changes to accept cropped output tensors; under the current gate contract, stay full-shape and make the final full-canvas op the free `output`.
