# J.27 — Loopy Belief Propagation / MRF for NeuroGolf 2026

## Summary
Loopy Belief Propagation (LBP) is **conditionally viable** for a narrow class of ARC denoising / hole-filling / symmetry-completion tasks, where 3–5 LBP iterations on a 10-state Potts model can be fit inside ~3–6M cost (≈ 10 pts). The structure is naturally expressible without `Loop` (just unroll the iterations). The lightest sound formulation is **min-sum / log-domain Potts BP with neighbour Conv 3×3**, which collapses to "Conv 3×3 + ReLU + Min-style ops" and avoids ReduceSum altogether. WFC is too symbolic for ONNX; Ising is a useful 2-state special case. LBP is the right tool *only when* the task is genuinely a constraint-satisfaction problem and not a pointwise transform.

## Background
A grid MRF assigns each pixel x_i a label from {0..K−1} with energy E = Σ_i D_i(x_i) + Σ_{(i,j)∈N} V(x_i, x_j) where D is the unary "data" term (from the input image) and V is the pairwise "smoothness" term (often a Potts model: V(a,b) = 0 if a=b else λ). LBP iteratively passes messages between 4-neighbours:
  m^{t+1}_{i→j}(x_j) = min_{x_i} [ D_i(x_i) + V(x_i,x_j) + Σ_{k∈N(i)\{j}} m^t_{k→i}(x_i) ]
After T iterations, the MAP label is argmin_{x_i} [D_i(x_i) + Σ_{k∈N(i)} m^T_{k→i}(x_i)].

Convergence: Felzenszwalb & Huttenlocher (CVPR 2004; PAMI 2006) show that for grid MRFs, "the number of iterations is proportional to image diameter" — i.e. for a 30×30 grid, ~30 iterations of standard BP, or O(log diameter) with their multi-grid variant. For denoising-only with strong unary evidence, **5–10 iterations** typically suffice (their Fig 6: low-energy solution in "just a few iterations per level"). Sudderth & Freeman (IEEE Sig. Proc. Mag 2008) confirm 5–15 iterations are usually enough on a 320×240 stereo grid.

For Potts pairwise potentials on a grid, each BP iteration can be implemented as a **convolution of message tensors** with a fixed neighbour kernel: messages along ↑↓←→ are independent, so a single Conv 3×3 with the cross-shaped initializer + a min-reduction along the label axis implements the message update.

## ONNX encoding
Represent the K-label belief at each pixel as one of the K=10 channels (this exactly matches NeuroGolf's [1,10,30,30] layout — one channel per ARC color).

In **min-sum / log-domain BP** (legal — no Loop / Scan / NonZero / Compress):

Per iteration:
1. Compute neighbour-sum of beliefs: `nb = Conv3x3(b, K_cross)` where K_cross is the 3×3 kernel `[[0,1,0],[1,0,1],[0,1,0]]` (cross-shape, no centre), groups=K so each label channel sums independently. → fixed weight, depthwise.
2. Add unary evidence: `b' = Add(D, Mul(λ, nb))`.
3. Optional Potts smoothness: subtract a per-label minimum to keep values bounded — `m = ReduceMin(b', axis=channel)` then `b' = Sub(b', m)` (ReduceMin is in `ai.onnx`, OK).
4. Optional softmax-like normalization with `Softmax(axis=1)` (allowed). Avoid normalization if you want to keep things lean — log-domain min-sum is unnormalized.

After T iterations the MAP label is `Argmax(b, axis=1)` (NeuroGolf forbids Argmax? — Argmax is in core ai.onnx and is not on the banned list). For NeuroGolf scoring, since the check is `(result > 0.0).astype(float)`, you only need the correct *channel* to be positive and others negative; you can compute it via `Sub(b, ReduceMax_with_keepdim)` and rely on the correct channel being the only zero/positive one.

Alternative **sum-product** would need Mul + ReduceSum + Softmax — still legal but more elementwise ops to track in `value_info`.

This is BP with a **uniform Potts pairwise potential**. Anisotropic edge potentials require a more expressive Conv (full 3×3 with K² params). For most ARC tasks (color-swap, symmetry-completion) the Potts assumption is correct.

## Cost analysis
Per BP iteration on the [1,10,30,30] tensor:

- Conv 3×3 **depthwise** (groups=10, K_cross fixed): 4·1·10·900 = 36,000 MACs (only 4 nonzero kernel weights, but ONNX MACs counts the dense 3×3 unless pruning is supported; effectively **9·10·900 = 81,000 MACs**, 90 params, 36,000 B mem).
- One Add of unary D (a constant initializer, foldable): adds ~36,000 B intermediate.
- Optional ReduceMin + Sub: ~36,000 B more intermediate.

Per iteration ≈ **81,000 (MACs) + 90 (params) + 72,000 (memory for two intermediates) = 153,090**.

- **5 iterations:** 5 × 153,090 ≈ 765,000 cost. ln ≈ 13.55. Score **≈ 11.45 pts**.
- **10 iterations:** ≈ 1,531,000. ln ≈ 14.24. Score **≈ 10.76 pts**.
- **20 iterations:** ≈ 3,062,000. ln ≈ 14.93. Score **≈ 10.07 pts**.

Felzenszwalb-style multi-grid BP (downsample, BP at low res, upsample) is *not* useful here because NeuroGolf has fixed 30×30 IO and an extra Resize op pair adds cost without compressing iterations meaningfully on such a small grid.

If you also need a non-Potts Conv 3×3 with full per-label-pair weights (K² = 100 weights per kernel position): cost per iteration jumps to **810,900 + 36,000 ≈ 847,000**, so 5 iterations = 4.2M → ln 15.26 → **9.74 pts**.

## ARC task fit
LBP is the right tool *iff* the task can be cast as MAP inference on a grid MRF:
- **Denoising / "remove noise pixels"** — strong unary D = 1{input ≠ noise color} with Potts smoothness. 3–5 iterations suffice.
- **Hole-filling** — set D=0 on hole pixels, D=large for "wrong" colors elsewhere. 5–10 iterations.
- **Symmetry completion** — encode symmetry by tying messages between symmetric pixels via Gather/Reshape (legal); a few iterations propagate the constraint. This is more sophisticated and probably the most novel use here.
- **Connected-component coloring under constraints** — works in principle but is dominated by simpler flood-fill (J.22).

LBP is **not** suited to: object recognition, counting, geometric transforms (rotation/reflection — better done by Transpose/Reverse), tasks where the rule is "apply transform T to objects of color C".

## Comparison with alternatives
- **vs. Wave Function Collapse (WFC):** WFC is constraint-satisfaction with backtracking and per-cell entropy heuristics — fundamentally not expressible without Loop/If, both banned. WFC's "soft" relaxation *is* essentially LBP. So LBP > WFC under NeuroGolf constraints.
- **vs. Ising model:** Ising = K=2 special case of Potts. With NeuroGolf's natural 10-channel layout, K=10 Potts is the same cost (it's all in the channel dimension). Ising buys nothing unless the task is genuinely binary.
- **vs. J.22 Reaction-diffusion:** RD propagates intensities; LBP propagates label probabilities. For "fill region" RD wins (cheaper per step, fewer iterations). For "decide which label each pixel should be given evidence and smoothness," LBP wins.
- **vs. J.4 ESN:** LBP encodes a real inference algorithm; ESN encodes random features. LBP > ESN for any constraint task.
- **vs. J.39 Cyclic CA:** CCA is unconditional (no input evidence guides the dynamics). LBP > CCA when you have evidence.
- **vs. BP-NN literature:** Kuck et al. NeurIPS 2020 ("Belief Propagation Neural Networks") show learned BPNN-D converges 1.7× faster than vanilla BP on Ising, but learned weights aren't available in NeuroGolf — vanilla LBP's hand-set Potts is the practical baseline.

The message-passing structure on a grid does **not** require Loop: parallel updates ↑↓←→ are exactly a Conv 3×3 with the cross kernel, and unrolling T iterations is a fixed-depth network. Felzenszwalb's "bipartite trick" (only update half the messages each iteration) halves cost — useful here.

## Verdict & recommended next experiments
**Verdict: viable for the denoise/complete task family.** Try LBP if and only if your candidate ARC tasks naturally read as "given noisy/partial evidence, output the most consistent grid."

Recommended experiments:
1. **5-iteration Potts-LBP** with cross-shape depthwise Conv 3×3 + ReduceMin + Sub. Cost ~770k, ~11.45 pts. Test on denoising/hole-filling tasks.
2. **3-iteration LBP + final hand-tuned 1×1 Conv readout** to convert log-beliefs to one-hot. Cost ~600k, ~11.6 pts.
3. **Symmetry-completion LBP** using Gather to tie messages across the symmetry axis. Risky to implement but unique value if it works.
4. **Skip** sum-product BP with Softmax; min-sum is leaner.
5. **Skip** non-Potts pairwise (full Conv 3×3 per iteration) — pushes you below 10 pts.
6. **Skip** learned BPNN-D — you can't train in-graph; pre-bake means you'd just be hand-tuning Potts λ anyway.
7. Use the bipartite half-update trick: alternate update of horizontal messages and vertical messages to halve per-iteration cost.

## Sources
- Felzenszwalb & Huttenlocher 2004/2006, "Efficient Belief Propagation for Early Vision" (CVPR 2004; IJCV 2006). Key: O(nk) BP, multi-grid for log-iteration, bipartite half-update.
- Sudderth & Freeman 2008, "Signal and Image Processing with Belief Propagation," IEEE Signal Processing Magazine.
- Szeliski et al. 2008, "A comparative study of energy minimization methods for MRFs," PAMI.
- Geramanis, github.com/NickGeramanis/denoising-inpainting-lbp — practical denoise/inpaint LBP reference.
- Natekar, github.com/parthnatekar/Loopy-Belief-Propagation — parallel 4-direction message passing implementation matching the Conv-3×3 pattern.
- Kuck et al. 2020, "Belief Propagation Neural Networks," NeurIPS 2020.
- Mooij & Kappen, "Sufficient conditions for convergence of LBP," arXiv:1207.1405.
- Wave Function Collapse: Gumin (mxgmn/WaveFunctionCollapse), Heaton 2018 explainer.
- ONNX ReduceMin / Softmax / Argmax operator specs, onnx.ai/onnx/operators.
