# 07_conv_reformulation — worked case studies

Replace Conv chains with smaller-cost equivalents.

## Files

- `task374_conv_run_combined.md` — Fold downstream Mul/Add into Conv weights (linearity). -11,728 bytes.
- `task323_conv_bool.md` — Conv with 0/1 weights → BOOL counting via Slice/Concat/ReduceSum. -62,838 bytes when downstream is bool-heavy.
- `task119_diag_conv.md` — Custom Conv kernel for shape detection (diagonal pattern). Lives in `01_whole_graph_rewrites/`. Cross-ref.

## When applicable

- Conv with integer weights → bool counting equivalent.
- Conv with downstream Mul/Add by per-channel constants → fold into weights.
- Conv detecting a SPECIFIC spatial pattern → use a Conv kernel of that exact shape.

## When NOT to reformulate

- Conv with non-trivial float weights (real filtering): keep.
- Conv with downstream nonlinear ops (Relu, Sigmoid): folding breaks.
- Conv with very small kernel (1×1): often already optimal.
