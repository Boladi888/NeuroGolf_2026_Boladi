# 06_cse_dead_op — worked case studies

Common-subexpression elimination and dead-op pruning. Both classes of "the graph has redundancy that can be cleaned mechanically".

## Files

- `task306_slice_cse.md` — Slice CSE (duplicate Slice nodes with identical params). onnxsim misses this; `pass_slice_cse.py` catches it. -6,504 bytes.
- `task093_dead_relu_post_maxpool.md` — Relu(Sub(MaxPool, X)) is provably identity when MaxPool has stride=1 and symmetric pads. -3,136 bytes on 4 dead Relus.
- `task302_drop_relu_shift.md` — Variant on dead-Relu.
- `task240_dead_end.md` — Documented dead-end (Pad-mode merge fails, fp16 regresses, no algebraic identities). Future workers: skip this set of attempts.

## When to suspect dead ops

- Many Slice nodes with similar inputs → CSE candidate.
- Relu after Sub/Subtraction that's provably non-negative → dead-Relu candidate.
- ConstantOfShape / Identity / Reshape with identity dims → often droppable.
