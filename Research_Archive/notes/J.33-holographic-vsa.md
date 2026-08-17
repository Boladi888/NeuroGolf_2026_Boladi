# J.33 — Holographic Vector-Symbolic Architecture (HVSA)

## Theory
Plate's HRR (IJCAI 1991, IEEE TNN 1995) binds two d-dim real vectors a, b via *circular convolution* a⊛b. The convolution theorem says a⊛b = IFFT(FFT(a)·FFT(b)). Kanerva's Binary Spatter Code (1994/1996) replaces circular conv by component-wise XOR on binary vectors. Frady, Kleyko et al. (2018+) survey the family ("Hyperdimensional Computing/VSA"). Joffe & Eliasmith 2025 (arXiv:2511.08747) recently applied a VSA solver to ARC-AGI, scoring 10.8 % on ARC-AGI-1-Train and 3.0 % on ARC-AGI-1-Eval — but with an *external symbolic search loop*, not a static graph.

## HRR over FFT in ONNX
The DFT op exists from opset 17 (ax-attribute) and was rewritten in opset 20 (ax-input). It computes y[k] = Σ_n exp(−2πi kn/N) x[n]. A 30-point DFT costs 30·30 = 900 complex MACs ≈ 3,600 real MACs (the real/imag interleaving). For a 30×30 grid we need a 2-D DFT = 30·30·30 + 30·30·30 = 54,000 real MACs. Binding two grids therefore costs 2 × 54,000 (forward FFTs) + 900 (Hadamard) + 54,000 (inverse FFT) ≈ 163 K MACs. Onnx-tool's profile of DFT is "complex multiplication" — i.e. it is *not* shape-only.

## Encoding ARC objects in HRRs
A typical scheme: random 1024-dim "role" vectors {COLOR, POSX, POSY, SHAPE} bound to "filler" vectors representing the value, summed into a single grid descriptor. To encode a 30×30 grid with up to 10 colors we need ≥ 300 bound pairs ⇒ 300 × 163 K = 49 M MACs minimum. **At ln(49 M) ≈ 17.7, the score is 25 − 17.7 = 7.3 pts — strictly worse than every Conv baseline.**

## Why VSA does not fit static-graph ARC
1. The unbinding step a* ⊛ (a⊛b) ≈ b requires a *cleanup memory* — a content-addressable lookup against a codebook. In a static ONNX graph the codebook is a constant `MatMul` (Gemm) of size [d, K] where K is the codebook size; for d=1024 and K=100 codes this is 102.4 K MACs and 410 KB of bytes — already 28 % of the size cap.
2. Binding/unbinding *commutes through addition* only approximately, which means HRR computations need iterative cleanup loops that the banned ops list (no Loop, no If) forbids.
3. Output is a noisy vector — it must be decoded back to a 30×30 image, requiring another massive linear projection.

## ONNX implementation rules (if you must)
1. If you encode HRRs at d=128 instead of 1024 and use only 10 role vectors, the FFT cost shrinks to ~4 K MACs per binding. But the resulting representation is too noisy for ARC's pixel-perfect scoring.
2. Use `Mul` instead of `DFT+Mul+IDFT` for "Hadamard-binding" Frequency-Holographic Reduced Representation (FHRR) à la Plate 1994 complex unit-magnitude vectors — saves the DFT round-trip but still requires a 2·d real `Mul`.
3. Use Kanerva BSC (binary, XOR-bound) only if quantising — but ONNX does not permit bool MAC accounting tricks; the BitwiseXor op exists from opset 18 and is allowed.

## Budget

| Config | MACs | Bytes | Score |
|---|---|---|---|
| FHRR binding at d=128 (1 role) | ~256 | 1,024 | ~19.5 pts |
| Full HRR ARC-encoder (10 roles, d=1024, FFT) | ~5 M | 50 K | ~9.5 pts |
| BSC binding at d=4096 (XOR + popcount) | ~4 K | 512 | ~16.7 pts (but no useful ARC solver fits) |
| DFT-based circ-conv 30×30 grid | ~163 K | 14 K | ~13.0 pts |

## Competitiveness
**Not viable.** The static-graph + log-cost regime penalises exactly the dimensions HRRs need to be reliable. The Joffe-Eliasmith VSA-ARC result is achieved with an *outer search loop over symbolic programs* — exactly what NeuroGolf bans. Only consider HRR-style ideas as a 1-D motif if you have a task that asks for binding *two* small role-filler pairs (e.g. "swap two specific colors"); even then, a Gather+Scatter recipe is cheaper.

## References
Plate 1991 IJCAI, 1995 IEEE TNN 6(3):623; Kanerva 1996 ICANN, 2009 *Cognitive Computation* 1:139; Frady et al. 2021 arXiv:2111.06077 ("Survey on Hyperdimensional Computing"); Joffe & Eliasmith 2025 arXiv:2511.08747; ONNX DFT spec opset 17 (https://onnx.ai/onnx/operators/onnx__DFT.html).
