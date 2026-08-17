# J.22 — Reaction-Diffusion Morphogenesis (Gray-Scott) for NeuroGolf 2026

## Summary
Gray-Scott reaction-diffusion is a **partially viable, narrow-use** approach: a 3–5-step unrolled Gray-Scott "fill/spread" module is buildable inside the cost cap (~3–5M cost ≈ 10 pts), but the canonical Gray-Scott regime (spots/stripes/labyrinths) needs 5,000–20,000 timesteps to stabilize on a 30×30 grid and is unreachable. For ARC tasks that look like "let activator pixels diffuse outward until they fill an enclosure," reaction-diffusion is dominated by simpler diffusion-only or NCA-style approaches that don't need the nonlinear `uv²` term. Use the Gray-Scott Laplacian + linear-leak structure, but skip the actual GS reaction.

## Background
The Gray-Scott PDE (Pearson 1993) on a 2D grid:
  ∂u/∂t = D_u ∇²u − u v² + F (1 − u)
  ∂v/∂t = D_v ∇²v + u v² − (F + k) v
Discrete update with Δt = 1, five-point Laplacian:
  u_{t+1} = u_t + D_u L(u_t) − u_t v_t² + F (1 − u_t)
  v_{t+1} = v_t + D_v L(v_t) + u_t v_t² − (F + k) v_t
where L is the 3×3 stencil [[0,1,0],[1,−4,1],[0,1,0]] (or the nine-point [[0.05,0.2,0.05],[0.2,−1,0.2],[0.05,0.2,0.05]]).

Pattern classes (Pearson; MROB taxonomy at mrob.com/pub/comp/xmorphia): spots, stripes, labyrinths, U-skate, mitosis. They emerge after **thousands** of forward-Euler steps with Δt ≈ 1 on a 256×256 grid; informally on a 30×30 grid the relevant "front-propagation" dynamics (a chemical front sweeping the grid) need at least **30 / max-front-speed ≈ 30–60 timesteps**, and full pattern stabilization needs many hundreds. CSAIL MIT's reference uses 5,400 steps; Munafo's stable patterns also require thousands.

For "fill region" / "spread color" tasks, you don't need pattern formation — you need **front propagation**, which only needs the Laplacian (D_u ∇²u) plus a saturating nonlinearity, i.e. a Fisher-KPP equation (a one-species RD). A 5-step unroll with kernel 3×3 already propagates a front by 5 cells, enough to fill many ARC bounded shapes.

## ONNX encoding
One Gray-Scott step using only legal ops (Conv, Add, Mul, Sub, Constant):

```
# Inputs: u, v as separate channels of a [1,2,30,30] tensor (or two channels in [1,10,...])
Lu  = Conv(u, K_lap)                        # 3×3 fixed Laplacian, depthwise (groups=1, c_in=c_out=1)
Lv  = Conv(v, K_lap)                        # same
v2  = Mul(v, v)
uv2 = Mul(u, v2)
u_diff = Mul(Lu, D_u_const)                 # scalar broadcast
v_diff = Mul(Lv, D_v_const)
u_feed = Mul(Sub(1, u), F_const)            # F * (1 - u)
v_kill = Mul(v, F_plus_k_const)             # (F+k)*v
u_new = Add(Add(u, u_diff), Sub(u_feed, uv2))
v_new = Add(Add(v, v_diff), Sub(uv2, v_kill))
```

Each step uses only `Conv`, `Mul`, `Add`, `Sub` — all legal. Better: pack u, v in two channels of one tensor and replace the 2× Conv calls with one Conv 3×3 with `groups=2` (depthwise) producing both Laplacians at once.

Laplacian Conv weight = a fixed initializer `[[[[0,1,0],[1,-4,1],[0,1,0]]]]` (per channel). Constant-folding cannot eliminate this Conv because it operates on the live activation.

For a "spread color" simplification, drop v entirely:
```
u_{t+1} = clip(u_t + D · Conv3x3(u_t, K_lap) , 0, 1)
```
which is the 1-species Fisher-KPP-without-reaction. This is what most ARC "fill" tasks actually need.

## Cost analysis
Per channel-pair Gray-Scott step on the [1,10,30,30] tensor (2 of the 10 channels are u, v; the other 8 carry color identity / one-hot):

- Conv 3×3 depthwise (groups=2, 2→2): 9·1·2·900 = 16,200 MACs, 18 params, mem 2·30·30·4 = 7,200 B.
- 4 elementwise Mul + 3 Add + 2 Sub: ~8 elementwise ops on a 2-channel tensor, each 1,800 elements. Elementwise ops contribute ~0 MACs but **do** generate intermediate value_info entries that count toward memory unless folded.
- Per step, ~5 intermediate tensors × 7,200 B = 36,000 B memory contribution; total per step ≈ **16,200 + 36,000 + 18 ≈ 52,200**.

A 10-step Gray-Scott unroll on the 2 channels: ≈ 522,000 cost. ln ≈ 13.16, score **≈ 11.84 pts**. Better than Conv 3×3 identity (11.39).

A 10-step **simplified single-channel diffusion-only** unroll (just one Conv 3×3 + one Add per step, on 10 channels):
- Per step: Conv 3×3 (10→10 depthwise, groups=10): 9·1·10·900 = 81,000 MACs + 90 params, mem 36,000 B → ≈ 117,090.
- 10 steps ≈ 1,170,900. ln ≈ 13.97. Score **≈ 11.03 pts**.

Full **dense** 10-channel Gray-Scott (couple all 10 color channels through reaction terms): per step is at least one Conv 3×3 (10→10) at 810,900 + many elementwise on 36,000-B tensors. **10 steps ≈ 9–11M cost** → ln ≈ 16.2 → **≈ 8.8 pts**. Marginal.

50–100 step "true" Gray-Scott regime: ≥ 50 × 52,200 ≈ 2.6M (best case, depthwise/2-channel) → ln 14.77 → **≈ 10.23 pts**, but 50 steps is **still nowhere near** convergence on a 30×30 grid (reaction-diffusion patterns need many hundreds to thousands of steps), so the model would not be functionally correct.

## ARC task fit
Reaction-diffusion *as front propagation* is well suited to:
- **Flood fill of enclosed regions** (e.g. ARC task `00d62c1b` — fill insides of shapes) — a saturating diffusion of a marker channel converges in ~grid-radius steps (~15 on 30×30).
- **"Spread color from seed pixel"** tasks — same as above.
- **Drawing connections / paths between marked endpoints** — the fronts collide along approximate shortest paths.

It is **not** suited to:
- Pattern *generation* (zebra stripes, leopard spots) — wrong regime, needs thousands of steps to stabilize and these patterns essentially never appear in ARC.
- Tasks requiring counting, object identity, or color permutation.
- Tasks where the *number* of fill steps depends on the input — NeuroGolf forbids Loop, so depth is fixed.

The Xu & Miikkulainen 2025 NCA-on-ARC paper specifically singled out enclosed-fill tasks like `00d62c1b` as NCA-friendly; the same applies to RD with even fewer parameters since the kernel weights are hand-set instead of learned.

## Comparison with alternatives
- **vs. J.42 Turing patterns:** Turing patterns are RD restricted to the activator-inhibitor regime that produces stable spots/stripes — i.e. a *strict subset* of Gray-Scott parameter space. Same op cost per step, but Turing requires the linear instability regime (long convergence). For ARC, plain RD with strong diffusion (front propagation) dominates Turing.
- **vs. J.4 ESN:** RD's Conv kernel is the *exact* Laplacian — no waste — so the same MAC budget yields actual spatial spread. RD strictly dominates ESN.
- **vs. J.27 Loopy BP:** LBP tackles "global consistency under given evidence" (denoising, ambiguous-pixel resolution); RD tackles "spread from a seed". Different tasks; both useful in their niche.
- **vs. J.39 Cyclic CA:** CCA is discrete-state and produces wave/spiral patterns; RD is continuous. For ARC's discrete-color outputs, CCA is more aligned, but RD's Conv-only structure compresses better.
- **vs. Neural Cellular Automata** (Mordvintsev et al. 2020; Xu & Miikkulainen 2025 ARC-NCA): NCA = learnable RD with hidden channels (~8k params for the canonical model). Under NeuroGolf, you **cannot** train inside the model — so you can't actually learn NCA weights at submission time. Hand-set RD weights are an NCA where you've already "frozen" the kernel.

## Verdict & recommended next experiments
**Verdict: viable but only as 5–15-step single-channel saturating diffusion**, not full Gray-Scott. The `uv²` reaction term is too expensive per step for the budget, and full GS pattern formation needs >>100 steps anyway.

Recommended experiments (in priority order):
1. **5-step depthwise-Conv diffusion + Clip** (one Conv 3×3 with groups=10, fixed Laplacian, then Add(input) and Clip(0,1) per step). Cost ≈ 600k → ~11.6 pts. Try on enclosed-fill tasks (`00d62c1b`-family).
2. **2-channel "activator + barrier" RD** with hand-set k, F to make a front that stops at obstacle pixels (color ≠ background). Cost ≈ 500k–1M for 10 steps → ~11–12 pts.
3. **Skip** the full Gray-Scott uv² nonlinearity unless you find an ARC task that genuinely requires it.
4. **Skip** any parameter regime aiming at spot/stripe formation — convergence depth is unreachable.
5. Important detail: the final `(result > 0.0).astype(float)` thresholding means you can let RD drive a marker channel above 0 anywhere it should be on, and below 0 elsewhere; you don't need exact saturation, which removes the need for many extra steps.

## Sources
- Pearson 1993, "Complex Patterns in a Simple System," Science 261:189.
- Munafo, "Reaction-Diffusion by the Gray-Scott Model: Pearson's Parameterization," mrob.com/pub/comp/xmorphia.
- MIT CSAIL amorphous group, "Gray Scott Model of Reaction Diffusion."
- Algosome, "Reaction Diffusion: The Gray-Scott Algorithm" (Laplacian 3×3 stencil).
- Munafo 2010/2014, "Stable localized moving patterns in the 2-D Gray-Scott model," arXiv:1501.01990.
- Adamatzky 2018, "Generative complexity of Gray-Scott model," arXiv:1610.09097.
- Hua/Hua, "Cellular Automata II: Gray-Scott Reaction-Diffusion Model" (minhhua.com).
- Mordvintsev et al. 2020, "Growing Neural Cellular Automata," Distill.
- Xu & Miikkulainen 2025, "Neural Cellular Automata for ARC-AGI," arXiv:2506.15746.
- Wikipedia, "Discrete Laplace operator" (3×3, 9-point stencil weights).
- ONNX Conv spec, onnx.ai/onnx/operators/onnx__Conv.html.
