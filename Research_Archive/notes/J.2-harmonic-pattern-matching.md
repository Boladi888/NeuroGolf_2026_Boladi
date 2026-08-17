# J.2 — Harmonic pattern matching (2D-DCT / Haar-wavelet fixed kernels)

## Theory
The 2-D type-II DCT on an N×N block is
  X(u,v) = α(u)α(v) Σ_{x,y} f(x,y) cos[π(2x+1)u/(2N)] cos[π(2y+1)v/(2N)],
with α(0)=√(1/N), α(k≥1)=√(2/N). The kernel is rank-1 separable: K_uv(x,y) = c_u(x)·c_v(y). For a fixed (u,v) the coefficient X(u,v) is a *single* dot-product against a constant kernel of size N×N, i.e. it can be encoded as one Conv with weight equal to the kernel image. The Haar wavelet at level 1 has analysis filters h_0 = [1,1]/√2, h_1 = [1,−1]/√2; the 2-D analysis bank has the four kernels LL = (1,1;1,1)/2, LH = (1,1;−1,−1)/2, HL = (1,−1;1,−1)/2, HH = (1,−1;−1,1)/2, all 2×2. Daubechies and Mallat proved these constitute an orthogonal basis — vital because it means a single fixed Conv is a *projector* with bounded gain.

## Where this fits ARC
ARC families with detectable periodicity include: tasks where the input is a small motif repeated K times across the grid (e.g. evaluation tasks descended from "fill-the-tile" — typified by tasks like *5582e5ca*-style colour-field detection, *bd4472b8*-style row-replication, *8eb1be9a*-style two-row patterns, and the entire "periodic extension" family discussed in the ARC AIX writeup). The *DC coefficient* X(0,0) of a 2×2 or 3×3 sliding window already tells you whether each block is uniform; the *first AC coefficient* X(0,1) detects horizontal stripes; X(1,0) detects vertical stripes; X(1,1) detects diagonal alternation — i.e. the 2×2 Haar HH kernel is the cheapest possible "is this a checkerboard?" detector at 30·30·10·1·4 = 36,000 MACs, scoring ≈ 25 − ln(36k) ≈ 14.5 pts at single-channel-output, *better than the 1×1 identity baseline of 13.591 pts*.

## ONNX implementation rules
1. Express each fixed kernel as a Conv weight initializer (float32, [out_C, in_C, kH, kW]). Do not use DFT/DCT ops — a Conv is cheaper because onnx-tool counts a 3-channel-out 2×2 Conv at 30·30·10·3·4 = 108k MACs versus a DFT at 30·30·30·log₂(30) ≈ 1.3M MACs (`onnx-tool` profiles DFT as a complex multiplication).
2. To threshold a single coefficient, follow `Conv → Greater(threshold) → Cast(to=FLOAT)`. Greater and Cast are O(N²) elementwise; for a 1-channel 30×30 output this is 900 + 900 = 1.8 K op-MACs, negligible.
3. Static shapes: use `pads`, `strides` attributes only — no `auto_pad="SAME"` runtime computation.
4. Constant-folding: keep weights as `Constant` initializers, not as graph inputs; the graph optimiser will fold them but their bytes still count toward the 1.44 MiB cap (a 30×30×10×10 float32 weight = 360 KB; a 3×3×10×10 weight = 3.6 KB).

## Budget for typical configs (30×30, 10 in-channels)

| Kernel | out-C | MACs | Params | Bytes | Approx. score |
|---|---|---|---|---|---|
| Haar 2×2 single sub-band | 1 | 9,000 | 40 | 160 | ~15.9 pts |
| Haar 2×2 full bank (4 sub-bands) | 4 | 36,000 | 160 | 640 | ~14.5 pts |
| DCT 3×3 four AC modes | 4 | 324,000 | 360 | 1,440 | ~12.3 pts |
| DCT 5×5 single mode | 1 | 225,000 | 250 | 1,000 | ~12.7 pts |
| Full 30×30 DCT (Conv) | 1 | 9,000,000 | 9,000 | 36,000 | ~9.0 pts (worse) |

## Competitiveness verdict
The 2×2 Haar HH kernel at ~16 pts beats every Conv-based identity baseline for any task whose answer is "checkerboard? yes/no/locate." It does NOT beat Identity/Transpose for tasks that are pure layout. Use J.2 only on (a) checkerboard-detect, (b) stripe-detect, (c) period-detection where the period divides 30 evenly. Candidate ARC tasks: *5582e5ca*, *868de0fa*, *9d9215db*, *e98196ab*, *3f7978a0* and similar period-7-or-period-3-style tasks; some lattice tasks like *6d75e8bb* (where output is the input with a uniform region inferred from periodicity).

## References
Ahmed, Natarajan, Rao 1974 (DCT-I/II); Wang 1984 (fast 2-D DCT); Mallat 1989 (multiresolution / Haar); Daubechies 1988 (orthonormal wavelets); ONNX Conv spec (opset 11+ stable).

## Cross-cutting findings (from brief #02 cluster)
- **Cost arithmetic is dominated by MACs.** With `points = max(1, 25 − ln(macs+memory+params))`, every doubling of MACs costs ~0.69 points. A single Conv k×k between 10 input and 10 output channels on a 30×30 grid burns 30·30·10·10·k² = 90,000·k² MACs.
- **The DFT op (opset 17+) is *not* banned**, but a 30×30 DFT costs ~810 K MACs (≈ Conv 3×3) — rarely cheaper than a 3×3 Conv with a hand-designed kernel.
- The "shape-only free" ops list (Reshape/Slice/Shape/Gather/ScatterND/Tile/Transpose/Cast/Squeeze/Unsqueeze/Split) is the secret weapon — biases competitive designs toward index-arithmetic.
