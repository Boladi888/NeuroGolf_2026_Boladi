# J.40 — Diffractive phase-mask emulation (D2NN)

## Theory
Lin et al., "All-optical machine learning using diffractive deep neural networks," *Science* 361:1004–1008, 2018, propose a multi-layer optical inference: input wavefront u_0(x,y) propagates through L phase-modulating planes M_ℓ separated by distance d, each layer applies u_{ℓ+1} = H_d * (M_ℓ · u_ℓ), where H_d is the Fresnel propagator with kernel h(x,y) = (e^{ikz}/(iλz)) · exp(ik(x²+y²)/(2z)). M_ℓ is a phase-only mask, M_ℓ(x,y) = e^{iφ_ℓ(x,y)}. Free-space propagation is therefore a *complex-valued convolution* with a fixed Fresnel kernel; the trainable parameters are the phase-mask pixels.

## D2NN → ONNX compilation
Each layer compiles to: (1) split u into real/imag channels (Slice, free); (2) multiply by mask = (cos φ, sin φ) initializers (Mul, 2·N² MACs); (3) Fresnel-propagate via Conv with a complex kernel split into real and imag channels (4·N²·k² MACs for a k×k truncated Fresnel kernel — a small number of zones, k ≤ 7 typically suffices for low-Fresnel-number propagation). Stacking L layers therefore costs L·(2N² + 4N²·k²) MACs. For N=30, k=5, L=3 this is 3·(1800 + 90,000) = 275,400 MACs ≈ 12.5 pts.

## Free Fourier compute claim — does it beat regular Conv?
No. The "free" in optics refers to *energy*, not MACs. In the ONNX graph each propagation is *literally a Conv* with a Fresnel kernel. The phase-only mask layer is a depthwise Mul. Compared to a single trainable 5×5 Conv (10.37 pts), a 3-layer D2NN with 5×5 Fresnel propagation between phase masks at 12.5 pts is *strictly worse*. The only advantage of D2NN architecture is that the *phase masks* are 1-channel-per-pixel (sparse), reducing parameter count, but the param/byte budget is not the bottleneck under the score formula.

## Specific kernel designs that exploit phase-only modulation
A 1-layer D2NN with a single quadratic phase mask M(x,y) = exp(iπ(x²+y²)/(λf)) is exactly a *lens*, computing the Fourier transform at the back focal plane. Combined with a flat aperture this gives an optical FFT. This is potentially useful for ARC tasks where a global frequency feature is needed — but, again, replacing this with a single fixed Conv whose weights are the explicit Fourier basis is cheaper because (a) MACs are the same but (b) you don't pay the per-element Mul phase modulation.

## ONNX implementation rules
1. Encode each phase mask as two `Constant` initializers (cos φ, sin φ), each [1,10,30,30] float32 ≈ 36 KB.
2. Encode the Fresnel kernel as one truncated complex Conv weight, split into real/imag → out_C=2·c per layer.
3. Use Mul + Add to fuse the complex multiplication: (u_r+i·u_i)·(m_r+i·m_i) = (u_r·m_r − u_i·m_i) + i·(u_r·m_i + u_i·m_r) — costs 4·N² MACs for the Mul stage.
4. Constant-folding will collapse adjacent phase-mask × Fresnel-kernel into a single complex Conv weight if both are constants. **Best practical recipe: fold the entire D2NN stack offline into one fixed real Conv of size [10, 10, k, k], lose the optical interpretation, and just call it "a Conv with a designed kernel."** This is mathematically equivalent and minimises operator count.

## Budget

| Config | MACs | Bytes | Score |
|---|---|---|---|
| 1-layer D2NN, k=5 Fresnel, 30×30 | ~92 K | 73 KB | ~13.5 pts |
| 3-layer D2NN, k=5, 30×30 | ~275 K | 220 KB | ~12.5 pts |
| Folded equivalent Conv 5×5 | ~225 K | 1 KB | ~12.7 pts |
| Folded equivalent Conv 3×3 | ~81 K | 360 B | ~13.7 pts |

## Competitiveness
**Strictly dominated by direct Conv designs.** D2NN buys nothing in MAC count, costs more in bytes (per-pixel masks), and the Fresnel kernel with very few zones at 30×30 (low Fresnel number ƒ = R²/(λz)) collapses to either a Gaussian (≈ blur) or a constant. For any ARC task that benefits from frequency-domain features, a single hand-designed Conv kernel (e.g. the DCT bases in J.2) is cheaper. **Use J.40 only as a pedagogical lens for designing complex-valued kernels, not as a deployed architecture.**

## References
Lin et al. 2018 *Science* 361:1004 (DOI 10.1126/science.aat8084); arXiv:1804.08711; Goodman, *Introduction to Fourier Optics*, ch. 4 (Fresnel/Fraunhofer kernels); Mengu et al. 2020 (D2NN improvements); Zhou et al. 2021 *Nature Photonics* 15:367 (large-scale D2NN); ONNX Conv op spec.

## Brief #02 caveats (cross-cutting)
- **The DFT operator ambiguity.** ONNX issue #4494 (open) confirms onnxruntime ≤ current does not support N-dimensional DFT input properly — only 1-D DFT along axis-1 of a (batch, signal) tensor works. For a 2-D 30×30 DFT one must apply DFT twice with a Transpose between — this doubles the MAC count.
- **Pinned dependency caveat.** The combination `numpy==2.4.4 onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1` is unusual. Validate your build pipeline before relying on any opset-23 features. Safe target opset for this competition: **17–20**, which gives DFT, Greater, Cast, Conv, Sin/Cos, Mul, Add, Slice, Gather, Transpose, Reshape, Identity, Constant, ConstantOfShape, Pad, BitwiseXor, all without subgraph attributes.
- **MAC budgets above assume onnx-tool's accounting**: zero-padding regions count as MACs, elementwise ops at one MAC per element, Sin/Cos/Exp at one MAC per element, DFT at the naive complex-multiplication count.
