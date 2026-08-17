live bar: 10942, sha256 16801a1ad992c8a92d7f459d9093398c90227b41e7785e487f3daddc7dcfae60 (`python tools/ng_current_bar.py --task 191 --json`, manifest, 2026-07-05T04:54:56Z).

Attempted route: direct 30x30 stamp output rewiring the stamp QLinearConv to emit `fill_mask_30` and removing `render_pad_fillmask_false`.

Failed gates:
- `python tools/ng_gate.py visible --task 191 --cand The_400_Tasks/Task_by_Task/task191/prototypes/task191_direct_stamp30.onnx` -> `[visible] 0/267 (0.00%)  FAIL` (`pads=[1,1,10,10]`, shifted top/left alignment).
- `python tools/ng_gate.py visible --task 191 --cand The_400_Tasks/Task_by_Task/task191/prototypes/task191_direct_stamp30_aligned.onnx` -> `[visible] 194/267 (72.66%)  FAIL` (`pads=[3,3,8,8]`, preserves live top/left but exposes nonzero trailing fringe).

Why dead: the existing 23x23 stamp domain plus 23->30 pad is semantically load-bearing. Keeping the live top/left alignment still emits wrong values in the trailing 30x30 fringe, so removing the crop/pad boundary is not a byte-equivalent shave. One visible miss is a certain hidden zero; do not run truth/AB/profile or build more direct-stamp padding variants.

One remaining hypothesis: a different output-assembly shave must preserve the existing 23x23 stamp domain and 23->30 pad semantics, or prove byte-equivalence against the 10,942 team-best before any profiling.
