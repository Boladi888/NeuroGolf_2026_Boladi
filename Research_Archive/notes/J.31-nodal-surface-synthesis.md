# J.31 — Nodal-surface synthesis (Chladni / standing waves)

## Theory
A Chladni standing-wave nodal pattern on a square plate is the zero-set of φ_{n,m}(x,y) = cos(nπx/L)·cos(mπy/L) − cos(mπx/L)·cos(nπy/L). The pattern decomposes a square into sign-domains separated by 1-D nodal lines. More generally, the interference pattern A·sin(αx + βy) + B·cos(γx + δy) has nodal lines that *split a 2-D grid along a chosen direction*; the simplest case sin(πx/L) crosses zero exactly once, dividing the grid in half.

## ARC fit
A few percent of ARC training tasks ask the solver to draw a separating line, fold along an axis, or split the grid in two then perform some operation per-half (*dc1df850* family, *1c0d0a4b* family, gridline-detect tasks like *1cf80156*, *27a28665*). For these, one needs a single zero-crossing oriented along the symmetry axis. A nodal-surface synthesis with sin(πx/30) requires (a) a 30×30 constant tensor of sin values (1 KB initializer), (b) one Add against the input image, (c) one Greater+Cast to binarise. Total runtime: 900 (Add) + 900 (Greater) + 900 (Cast) = 2.7 K MACs.

**However** — the same effect is achievable with `Slice` (free!) of the input by half, then `Concat` (free). For pure grid-bisection ARC tasks the Slice/Concat approach beats nodal synthesis by a factor of ∞ in MAC count. **The only case where nodal synthesis wins is when the fold/split must depend on a learned phase that does not align with grid axes — and ARC tasks do not have such cases at 30×30 resolution.**

## ONNX implementation rules
1. Sin/Cos are opset-7 element-wise float ops, never subgraph-attributed, fully legal. Each costs N MACs in onnx-tool (one per element, since the tool doesn't have a special-function MAC formula).
2. **Crucially: build sin/cos arrays as `Constant` initializers, never as Sin/Cos nodes applied to a coordinate tensor.** Constant folding will eliminate the Sin/Cos nodes if their input is a constant, but the *initializer bytes* still count: a 30×30 float32 constant = 3,600 bytes per pattern.
3. Use `Add` of two pre-computed sine-grids, then `Greater(0)`, then `Cast(to=FLOAT)`, then `Mul` against the gather-encoded input.

## Penrose / quasicrystal interference
Penrose tilings and aperiodic Chladni patterns require the sum of ≥ 5 incommensurate cosines. ARC has no aperiodic tasks at 30×30 — quasicrystal mathematics is not exploitable here.

## Budget

| Config | MACs | Bytes | Score |
|---|---|---|---|
| Single sin grid + Add + Greater + Cast | ≈ 3,600 | 3,600 | ~16.8 pts |
| Two-mode Chladni Add+Greater | ≈ 4,500 | 7,200 | ~16.6 pts |
| 4-mode interference | ≈ 9,000 | 14,400 | ~15.9 pts |
| Slice+Concat equivalent (free) | 0 | 0 | 25 pts |

## Competitiveness
Lose-lose: for any task that nodal synthesis can solve, Slice+Gather+Reshape can solve it at zero cost. **Skip J.31** unless a future ARC task family forces an angled split that does not align with grid axes — unlikely at 30×30. Keep the technique as a fallback for compositional tasks where a continuous nodal mask must be combined multiplicatively with another channel.

## References
Chladni, *Entdeckungen über die Theorie des Klanges* 1787; Bourke 2003 (closed-form Chladni equation); Tuan et al., recent inhomogeneous Helmholtz analyses summarised in PMC10969725.
