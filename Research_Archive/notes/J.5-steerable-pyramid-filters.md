# J.5 — Steerable pyramid filters

## Theory
Freeman & Adelson, "The Design and Use of Steerable Filters," IEEE TPAMI 13(9):891-906, 1991, showed that any rotated copy of a band-limited function f^θ(x,y) can be expressed exactly as f^θ = Σ_{j=1..M} k_j(θ)·f^{θ_j} where M depends only on the angular bandwidth. For a 2-D Gaussian first-derivative the basis size is M=2 (cos θ, sin θ); for second-derivative M=3. Simoncelli, Freeman, Adelson, Heeger 1992 ("Shiftable Multiscale Transforms") generalized this to a self-similar pyramid with K oriented bands and L scales. Cohen & Welling's G-CNN (2016) and Weiler et al.'s Steerable CNNs (2017) reformulate the same idea in modern equivariant-network language; the *p4* group (90° rotations) and *p4m* (rotations + flip) are the relevant discrete groups for ARC.

## Steerability vs D4 group at 30×30
ARC contains *only* discrete 90° rotations and reflections (the dihedral group D4 of order 8). For a discrete group, exact equivariance is bought via parameter-tying: one conv weight w yields four rotated copies {w, R₉₀w, R₁₈₀w, R₂₇₀w} for free at *parameter* level but each rotated kernel still costs the full MAC bill of an independent Conv. The Simoncelli-style steerable basis buys nothing here because (a) the group is finite so no interpolation is needed, (b) on a 30×30 grid with ≤ 5×5 kernels the angular bandwidth is too low to need a continuous orientation parameterization. **Verdict: at 30×30, plain D4 kernels with weight-sharing dominate; full steerable pyramids are a parameter-efficiency win that does not translate to a MAC-efficiency win.**

## ONNX implementation rules
1. Pre-compute the D4 orbit of a base kernel (e.g. an oriented 3×3 derivative) offline, store all 4 (or 8) rotated versions as a single Conv weight tensor of shape [4·C_out, C_in, 3, 3]. This is one Conv node, MAC = 30·30·C_in·4·C_out·9.
2. Use `Transpose` (free) to produce the *image-side* rotations needed for cyclic-slice tricks (Dieleman et al. 2016): cycling-slice a 4-stack of D4-rotated copies of the input through one Conv with a *symmetric* kernel produces an equivariant feature map at the same MAC cost as a single Conv. Because Transpose is a no-MAC op, this is essentially free.
3. Keep the orientation count to the minimum needed: 2 for axis-aligned-vs-anti-axis-aligned, 4 for full p4, 8 for p4m. ARC reflection-symmetry tasks resolve at 4.

## Math
A 3×3 first-derivative steerable basis is
  G_x = [[-1,0,1],[-2,0,2],[-1,0,1]]/8 and G_y = G_xᵀ.
For arbitrary θ, ∂_θ G = cos θ·G_x + sin θ·G_y. For ARC's discrete θ ∈ {0,90,180,270} you only need {G_x, G_y, −G_x, −G_y}, but −G_x is just sign-flipped G_x → handled by a free `Mul` with constant −1 (still costs 9000 MACs as elementwise). Cheaper: emit four convs with the four sign patterns as separate output channels in one Conv node.

## Budget

| Config | out-C | MACs | Approx. score |
|---|---|---|---|
| Single oriented Sobel 3×3 | 1 | 81,000 | ~13.7 pts |
| 4-rotation D4 bank, 3×3 | 4 | 324,000 | ~12.3 pts |
| 8-element p4m bank, 3×3 | 8 | 648,000 | ~11.6 pts |
| Steerable 5×5 pair (G_x,G_y) | 2 | 450,000 | ~12.0 pts |
| Two-scale steerable pyramid 3×3+downsample-by-Slice | 4 | ~405,000 | ~12.2 pts |

## Competitiveness
Useful for the rotation-symmetry ARC family (*c3e719e8*, *7468f01a*, *ba97ae07*, *3631a71a*, *ed36ccf7*) where the answer is "the rotated copy of the unique non-rotational-symmetric part" — but cheaper to do the rotation via Transpose+Slice (free) and compare with the input via a 1×1 Conv. **Use J.5 only when an oriented-edge response is needed at sub-30 resolution; otherwise prefer Transpose+arithmetic.**

## References
Freeman & Adelson 1991 IEEE TPAMI; Simoncelli, Freeman, Adelson, Heeger 1992 "Shiftable Multiscale Transforms" Vision Modeling Tech Rep 161; Cohen & Welling, "Group Equivariant Convolutional Networks," ICML 2016 (arXiv:1602.07576); Weiler, Hamprecht & Storath 2017 "Learning Steerable Filters" arXiv:1711.07289; Dieleman et al. 2016 "Exploiting Cyclic Symmetry" arXiv:1602.02660.
