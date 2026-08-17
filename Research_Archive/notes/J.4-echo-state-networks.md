# J.4 — Echo State Networks for NeuroGolf 2026

## Summary
Echo State Networks (ESNs) are almost certainly **not viable** as a base architecture for NeuroGolf and are at best useful as a one-shot feature-expansion preamble in front of a tiny readout. The "fixed random recurrent reservoir" idea collides head-on with NeuroGolf's cost model: every reservoir Conv step costs the same 810k MACs as a 3×3 identity Conv (≈11.39 pts on its own), and an ESN needs *several* such steps before the random projection has done anything useful. The "free random features → linear readout" pattern only pays off when the readout is non-trivial; for ARC tasks the readout *is* the whole problem, so a single hand-designed Conv with the right weights almost always beats a random reservoir + 1×1 readout.

## Background
An Echo State Network is the canonical reservoir-computing architecture (Jaeger 2001). The recurrence is
  x_{t+1} = tanh(W_in u_{t+1} + W x_t),  y_t = W_out [x_t; u_t]
with W_in, W and (importantly) the recurrent reservoir matrix **W** drawn at random and frozen. Only the linear readout W_out is trained.

Three properties matter here:
- **Echo State Property (ESP):** The reservoir state must asymptotically depend only on input history, not on initial state. The classical sufficient condition is that the spectral radius ρ(W) < 1 for tanh nonlinearity (Jaeger 2001; the "spectral radius indicator" remains valid for large reservoirs — IEEE 2013). In practice ρ ≈ 0.9–1.0 ("edge of chaos") is best for memory.
- **Memory capacity** is bounded above by reservoir size N (Jaeger 2002, Scholarpedia). For a linear reservoir the bound is attainable; nonlinear reservoirs typically achieve a fraction of N.
- **Sparsity.** Classical ESNs use sparse W (~10% connectivity) for compute savings, not for capacity — the relevant theory is invariant under unitary rotations of W.

For grids, the spatial analog is a **convolutional reservoir** (Tong & Tanaka 2018; Ma et al. ConvESN 2017): replace the random dense W with random fixed Conv weights and a tanh/ReLU. Two-or-more-stage stacks are possible (multi-reservoir / deep ESN, MDPI 2024).

## ONNX encoding
A minimal "spatial ESN" on the [1,10,30,30] tensor:

1. **Input projection W_in** — Conv 1×1 from 10 → C_res channels. Random fixed weight initializer.
2. **Reservoir step (×T):** Conv 3×3 (C_res → C_res, padding 1, fixed random weights scaled to spectral radius < 1) followed by `Tanh` (or `Relu`). Optionally an `Add` of a skip connection from the previous state for leaky integration.
3. **Readout W_out** — Conv 1×1 (C_res → 10).

All ops are in the "" / "ai.onnx" domain, all element-wise, no subgraph attributes — fully legal under NeuroGolf's banned-op list.

Spectral-radius enforcement is **purely a pre-bake step** done in numpy: draw W ~ N(0, σ²I), compute ρ via power iteration, divide W by ρ/0.95. The resulting weights become a fixed `initializer` blob in the ONNX file. Constant-folding doesn't help here (the multiplications happen on activations, not on weights), so the spectral-radius cost is paid in full at inference time.

Sparse Conv: ONNX has no first-class sparse Conv kernel; sparsity reduces *params* (zeros encoded) but **does not reduce MACs** under the NeuroGolf metric (which counts dense Conv MACs, just like onnx-tool by default — onnx-tool docs explicitly state zero-padding is not ignored from MACs statistics). So sparsity ≠ free.

## Cost analysis
Baseline calibration (natural-log scoring): Conv 3×3 identity (10→10) = 810,000 MACs + 900 params = 810,900 cost → 25 − ln(810,900) = 11.39 pts. ✓ matches the spec.

Per **reservoir step** at the native C=10 channel count:
- Conv 3×3 (10→10): 810,000 MACs, 900 params, output activation 10·30·30·4 = 36,000 B memory.
- Tanh: 0 MACs (counted as elementwise), 0 params, 36,000 B output.
- One step ≈ **810,900 + 36,000 ≈ 846,900** added to total.

A 3-step ESN with W_in (1×1 10→10) + 3×reservoir + W_out (1×1 10→10):
- W_in: 90,100
- 3 × Conv3×3 step ≈ 3 × 846,900 = 2,540,700 (counting 2 intermediate value_infos at 36,000 B and 1 final feeding into W_out at 36,000 B)
- W_out: 90,000 MACs + 100 params + 0 mem (named output) = 90,100
- **Total ≈ 2,720,900**, ln ≈ 14.82, score **≈ 10.18 pts**.

A 5-step reservoir is ≈ 4.4M cost → ln ≈ 15.30 → ≈ **9.70 pts**.

If you push to C_res=32 channels for richer features (Conv 1×1 10→32, then Conv 3×3 32→32 ×3, then 1×1 32→10):
- W_in 1×1 (10→32): 10·32·900 + 320 = 288,320
- Conv 3×3 (32→32): 9·32·32·900 = 8,294,400 MACs + 9,216 params + 32·30·30·4 = 115,200 mem ≈ 8.4M per step
- 3 such steps ≈ 25.3M
- W_out 1×1 (32→10): 32·10·900 + 320 = 288,320
- **Total ≈ 26M**, ln ≈ 17.07 → **≈ 7.93 pts**. Strictly worse than a single Conv 3×3 (11.39 pts).

**The "for free" hope fails:** random weights are foldable only when they apply to a known constant, which the input is not. Conv MACs are paid on the activation, not on the weight blob.

## ARC task fit
ESNs are general-purpose nonlinear feature extractors. ARC tasks where this *might* help:
- **Object detection / connected-component priors** that need a few hops of nonlinear local mixing (e.g. detecting "isolated single pixels" vs "filled blocks").
- **Texture / parity tasks** (e.g. checkerboard detection) where a randomized 3×3 nonlinear filter bank gives almost any local feature in one or two steps.

But the cost-effective alternative is *always* a single hand-tuned 3×3 Conv with the exact handcrafted feature you wanted, at 11.39 pts vs ≤10 pts for any meaningful ESN. The only regime where ESN is interesting is when you genuinely *don't know* what feature to extract and you want to brute-force search W_out — but you can't search in NeuroGolf because there is no training-time feedback inside the ONNX graph.

## Comparison with alternatives
- **vs. J.22 Reaction-diffusion** (this folder): RD is also a fixed-weight unrolled Conv stack. RD has *interpretable* weights (Laplacian + reaction polynomial) so the same number of layers buys you genuine spatial spread, while ESN's random weights are wasted. RD ≥ ESN for ARC purposes.
- **vs. J.27 Loopy BP:** LBP iterations cost about the same per step (Conv 3×3 + Mul + ReduceSum) but actually solve a *defined problem* (MAP inference). ESN's random reservoir doesn't solve anything by itself. LBP > ESN whenever the task is denoise/complete.
- **vs. J.39 Cyclic CA:** CCA uses Conv 3×3 + comparisons + Where; same cost per step, but the dynamics are categorical and align with ARC's discrete colors. CCA > ESN for any task with discrete state evolution.
- **vs. J.42 Turing patterns:** Turing patterns are a *constrained* RD instance with two channels and a known instability — strictly more useful than a random reservoir of the same size.

In short, every other listed J-topic uses its conv-stack budget for interpretable structure; ESN spends the budget on randomness.

## Verdict & recommended next experiments
**Verdict: skip ESN as a primary architecture.** It cannot beat a single hand-tuned Conv on the cost/expressiveness frontier under NeuroGolf's metric.

If you still want to test it (one short experiment, then move on):
1. **Single-step "1-Conv ESN"**: random fixed Conv 3×3 (10→10) with ρ=0.95 + tanh + Conv 1×1 (10→10) readout. Cost ≈ 901,000 → ≈ 11.29 pts. Compare against a hand-tuned 3×3 Conv solving the same family of tasks (which scores 11.39 pts and is more accurate). If the random-feature version *ties on accuracy* across many tasks, keep the recipe in your toolkit; if it loses on any, abandon ESN.
2. **Skip** dense reservoirs at C_res > 10 — the math above shows they are dominated.
3. **Skip** sparsity tricks — they cost MACs the same as dense conv under the metric.
4. Do *not* bake spectral-radius enforcement into the graph; do it at weight-export time in numpy.

## Sources
- Jaeger, "Echo State Networks" — Scholarpedia (memory-capacity bound C ≤ N).
- Yildiz, Jaeger, Kiebel, "Re-visiting the Echo State Property," Neural Networks 2012.
- IEEE 2013, "The spectral radius remains a valid indicator of the ESP for large reservoirs."
- Schubert & Gros 2021, "Local homeostatic regulation of the spectral radius of ESNs," Front. Comput. Neurosci.
- Wringe, Trefzer, Stepney, "Reservoir Computing Benchmarks," arXiv:2405.06561 (2024).
- Ma et al., ConvESN, 2017; Tong & Tanaka, "Reservoir Computing with Untrained CNNs," ICPR 2018.
- "Energy-Efficient Edge ... Multi-Reservoir ESN," MDPI Sensors 24(11):3640 (2024).
- Rapid training echo convolution network, Information Sciences 2024.
- ONNX Conv operator spec, onnx.ai/onnx/operators/onnx__Conv.html.
- onnx-tool/Profile.md (zero weights still count as MACs).
- NeuroGolf 2026 task spec (this brief): empirical baselines for Conv 1×1/3×3/5×5 identity costs.
