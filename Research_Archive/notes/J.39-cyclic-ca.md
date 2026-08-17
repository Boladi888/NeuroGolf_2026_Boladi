# J.39 — Cyclic Cellular Automata for NeuroGolf 2026

## Summary
Cyclic Cellular Automata (CCA) — and the closely related Greenberg–Hastings excitable-medium model — are **a niche but cheap** option: a single CCA step is a 3×3 Conv + a comparison + a `Where`, costing roughly the same as one Conv 3×3 step (~850k). 3–5 steps fit comfortably in budget (~3M cost ≈ 10.5 pts). However, the ARC corpus contains essentially **no tasks with cyclic-wave / spiral-pattern outputs**, so CCA's natural target outputs are absent. Its real value in NeuroGolf is as a **discrete-state local update primitive** (the Where-clamped advance rule) that can be reused for non-cyclic discrete CAs (Game-of-Life-style, majority vote, etc.) where cycles in spectral content are not required.

## Background
The κ-color cyclic CA (Fisch–Gravner–Griffeath; Wikipedia) on a 2D grid:
  ξ_{t+1}(v) = (ξ_t(v) + 1) mod κ if at least θ neighbours have state (ξ_t(v) + 1) mod κ; else ξ_t(v).

The Greenberg–Hastings (GH) variant (Wikipedia, psoup.math.wisc.edu): only state 0 advances by contact; states 1..κ−1 advance automatically each step. With κ=3 (resting/excited/refractory) you get the canonical excitable-medium model.

Asymptotic behavior on Z² from a random configuration (Fisch et al.; Bello & Sivakoff 2021):
- Random phase → consuming phase (large monochromatic blocks fight) → demon phase (locally periodic spirals).
- For κ=3, θ=1 with the von Neumann neighbourhood, demons emerge in O(grid-diameter) steps; on a 30×30 grid that's ~30 steps to reach steady state, but **partial structure (cluster formation) emerges in 3–5 steps**.

Connection to Turing patterns / RD: GH/CCA are the discrete-state limit of an excitable RD system. Same kind of spiral-wave attractors, but without continuous parameters — the "wavelength" is set entirely by κ and θ.

## ONNX encoding
Represent state ξ ∈ {0..κ−1} as a one-hot K-channel tensor (K=κ, the existing 10-channel input layout maps κ ≤ 10 cleanly).

One CCA step (κ states, threshold θ, von-Neumann neighbours):
1. **Identify successor channel** for each pixel: `succ = Roll(state_onehot, axis=channel, shift=−1)`. ONNX has no Roll, but it's just a Concat of two Slice outputs along the channel dim — both legal.
2. **Count successor neighbours**: `nb_count = Conv3x3(succ, K_neumann)` with K_neumann = `[[0,1,0],[1,0,1],[0,1,0]]`, depthwise (groups=K). Output is per-channel neighbour count of successor state at each pixel.
3. **Find which pixels have their successor among neighbours** (advance condition): The relevant condition for cell at state s is `nb_count[s+1 mod K] ≥ θ`. Equivalently, multiply the neighbour-count tensor elementwise by the *current* state's one-hot mask shifted by 1, sum across channels: `match = ReduceSum(Mul(nb_count, state_onehot_rolled), axis=channel, keepdims=True)`. Compare to threshold: `advance = GreaterOrEqual(match, θ)` → boolean mask [1,1,30,30].
4. **Advance**: `next_state_onehot = Where(advance_broadcast, state_onehot_rolled, state_onehot)`. The element-wise `Where` is legal (it's the standard ONNX `Where(condition, X, Y)` op, **not** the subgraph-controlled `If`). It does element-wise selection with no subgraph attribute.

For Greenberg–Hastings: only state 0 needs the contact rule; all other states advance unconditionally. Encode by computing `auto_advance_mask = Sub(1, state0_onehot)` (the "not state-0" pixels) and OR-ing with the contact-advance mask before the final Where.

All ops used (`Conv`, `Slice`, `Concat`, `Mul`, `ReduceSum`, `GreaterOrEqual`, `Where`, `Sub`) are legal under NeuroGolf rules — `Where` is element-wise with no subgraph attributes.

## Cost analysis
Per CCA step on a [1,10,30,30] state tensor (K=10):

- 1 Slice + 1 Concat for the channel-roll: ~0 MACs (shape op), but one extra value_info ≈ 36,000 B.
- 1 Conv 3×3 depthwise (groups=10, fixed K_neumann): 9·1·10·900 = 81,000 MACs, 90 params, 36,000 B mem.
- 1 Mul (state_onehot × nb_count): elementwise, 36,000 B.
- 1 ReduceSum across channel: produces [1,1,30,30] = 3,600 B.
- 1 GreaterOrEqual: 3,600 B.
- 1 Where: produces full state tensor again, 36,000 B (this is the step output, foldable as next step's input).

Per step total ≈ **81,000 (MACs) + 90 (params) + ~150,000 (memory across 4–5 intermediate value_infos) ≈ 231,000**.

- **3 steps:** ≈ 693,000. ln ≈ 13.45. Score **≈ 11.55 pts**.
- **5 steps:** ≈ 1,155,000. ln ≈ 13.96. Score **≈ 11.04 pts**.
- **10 steps:** ≈ 2,310,000. ln ≈ 14.65. Score **≈ 10.35 pts**.

If you use a **non-depthwise** Conv 3×3 (10→10 dense): each step jumps to ~810,900 + 150,000 ≈ 960,000. 5 steps ≈ 4.8M → 9.62 pts. Don't do that — depthwise per-channel counting is exact for CCA.

## ARC task fit
**Direct fit: very narrow.** ARC's 800 training tasks have effectively zero outputs that look like cyclic-wave attractors / spirals. Empirically, scanning Hodel's DSL coverage (165 ops × 400 tasks) and Lewish's tagged taxonomy reveals dominant priors as object, color-mapping, geometry, counting, fill, symmetry — none of which align with CCA dynamics.

**Indirect fit: useful as a primitive.** The Conv3×3 + Mul + ReduceSum + GreaterOrEqual + Where pipeline is exactly the recipe for **any totalistic discrete CA**, including:
- Game of Life rules (B3/S23) for objects evolving by neighbour-count rules.
- Majority-vote denoising (a 1-step "if 5/8 neighbours agree, switch").
- "Erode/dilate by one if neighbour-count threshold met" — useful for shrinking shapes by 1 pixel per step.

So the take-away is: budget for **2–3 CCA steps** as a *generic discrete-state-update primitive*, not as a wave-pattern generator.

## Comparison with alternatives
- **vs. J.4 ESN:** CCA's Conv weights are designed (cross kernel + threshold), not random. CCA > ESN.
- **vs. J.22 RD:** Both unroll fixed-depth Conv stacks. RD uses continuous values; CCA uses discrete one-hot. For ARC's discrete colors, CCA's one-hot is exactly the right representation, so the final thresholding step is trivial. RD wins for "fill", CCA wins for "advance state".
- **vs. J.27 LBP:** LBP minimizes a global energy; CCA is a local rewrite rule. LBP needs ≥5 iterations with global propagation; CCA can effect a one-step pixel rewrite. CCA is *cheaper per step* but has no notion of evidence weighting.
- **vs. J.42 Turing:** Turing is continuous and needs many steps; CCA is discrete and 3 steps already do something. CCA > Turing for ARC discreteness.
- **vs. NCA (Mordvintsev 2020; Xu 2025):** NCA learns the update rule with backprop; CCA hand-codes it. Without training, CCA = "frozen NCA with Game-of-Life-style rules". The Differentiable Logic CA paper (Miotti et al. 2025) shows that NCA rules can be implemented as fixed logic gates — same class.

## Verdict & recommended next experiments
**Verdict: skip pure CCA**, but keep the **`Conv3×3 + ReduceSum + Where` discrete-update primitive** in your toolkit for any task that needs "rewrite each pixel based on a count of like-neighbours."

Recommended experiments:
1. **2-step Game-of-Life-style update** with hand-set rule: cost ~460k → ~12.0 pts. Test on tasks like "remove isolated single pixels" or "fill by one-neighbour majority" — these are real ARC denoising patterns.
2. **3-step Greenberg-Hastings** wave dynamics with κ=3 — only worth running if you spot an ARC task with wave-like outputs (you probably won't).
3. **5-step CCA with κ=10** to test the auto-advance primitive at NeuroGolf-native channel count: ~1.15M → 11.0 pts.
4. **Skip** dense Conv versions — depthwise is sufficient and 10× cheaper.
5. **Skip** spiral / demon-phase exploration — convergence depth (~30 steps on 30×30) is too expensive (>2.7M cost) for what is in any case probably not in ARC.

## Sources
- Wikipedia, "Cyclic cellular automaton" — phases (random / consuming / demon) and rule definition.
- Wikipedia, "Greenberg–Hastings cellular automaton" — winding-number proof of repetitive behaviour.
- psoup.math.wisc.edu/mcell/rullex_cycl.html — MCell CCA / GH rule lexicon.
- Hellouin de Menibus & Le Borgne 2021, "Asymptotic behaviour of the 1D rock-paper-scissors CCA," Ann. Appl. Probab. 31(5):2420.
- Fisch, Gravner & Griffeath, "Asymptotic behavior of excitable cellular automata," Experimental Mathematics 2(3).
- Bello & Sivakoff 2021, "CCA and Greenberg-Hastings models on regular trees," arXiv:2108.06404.
- Wroblewski, github.com/NathanielWroblewski/greenberg-hastings — visual reference implementation.
- Miotti, Niklasson, Randazzo, Mordvintsev 2025, "Differentiable Logic CA: From Game of Life to Pattern Generation," arXiv:2506.04912.
- Mordvintsev et al. 2020, "Growing Neural Cellular Automata," Distill.
- ONNX `Where` operator, onnx.ai/onnx/operators/onnx__Where.html — confirmed element-wise with no subgraph attribute.
- ONNX `Slice`, `Concat`, `ReduceSum`, `GreaterOrEqual` — all in ai.onnx default domain.
