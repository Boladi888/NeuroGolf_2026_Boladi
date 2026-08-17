# J.42 — Turing Pattern Morphogenesis for NeuroGolf 2026

## Summary
Turing-pattern morphogenesis (an activator-inhibitor RD system in the diffusion-driven instability regime) is **not a viable primary architecture** for NeuroGolf: stable Turing patterns need O(10²–10³) timesteps to converge from a homogeneous + small-perturbation initial condition on a 30×30 grid, which is far beyond any cost budget (each step ≈ 100k cost; 100 steps ≈ 10M cost ≈ 8.7 pts at best, *if* it converges). Turing patterns are a strict special case of reaction-diffusion (J.22) — same op recipe, more constrained parameter regime, longer convergence — so anything Turing can do, J.22's Gray-Scott can do at least as well per step. ARC contains no clear "stripe-from-spot" or "spot-from-noise" tasks anyway.

## Background
**Turing 1952** showed that a system of two morphogens with reaction kinetics that has a stable spatially-uniform steady state can become *spatially* unstable when one species (the inhibitor) diffuses sufficiently faster than the other (the activator). The classical conditions (Murray; Caltech BE150 lecture):
1. Stable kinetics in absence of diffusion.
2. Activator self-activation (∂f/∂u > 0).
3. Inhibitor cross-inhibits the activator (∂g/∂u > 0, ∂f/∂v < 0).
4. **D_v / D_u > critical ratio** (typically 5–10 for Gierer-Meinhardt; the requirement D_v >> D_u is the famous "local activation, lateral inhibition" principle — Meinhardt 2000).

Wavelength scales as λ_pattern ∝ √(D_v / D_u · 1/|kinetics|). Spots vs stripes is set by the *cubic* term in the reaction polynomial; stripes appear in supercritical Hopf-near regimes. Wavelength of a Turing-type mechanism regulates the morphogenesis of meshwork patterns (Nature Sci Rep 2021) — small wavelength → dense; large → sparse.

**Convergence time** to a stable Turing pattern is governed by the slowest unstable eigenvalue of the linearized RD operator; typical numerical simulations on 100×100 grids use 10⁴–10⁵ Euler steps (BioRxiv 2025 "Fundamental limits on patterning"; ipython-cookbook 12.4 uses ~9000 steps with Δt=0.001). Even on a 30×30 grid, you cannot reasonably converge a Turing pattern in <100 steps without an aggressive timestep that risks numerical instability (CFL-bound for explicit Euler with the Laplacian: Δt ≤ Δx²/(2·max(D))).

Turing patterns vs Gray-Scott (J.22): Gray-Scott is one specific reaction polynomial (cubic autocatalysis with feed/kill); Turing is the *diffusion-driven instability principle* — a class of (f, g, D_u, D_v) systems that includes Gray-Scott, Gierer-Meinhardt, Schnakenberg, Lengyel-Epstein, etc.

## ONNX encoding
Two-channel state s = (u, v). Per Turing step (Gierer-Meinhardt-flavored, D_v >> D_u):

```
Lu = Conv3x3(u, K_lap)                   # depthwise, 5-pt or 9-pt Laplacian
Lv = Conv3x3(v, K_lap)
diff_u = Mul(D_u, Lu)
diff_v = Mul(D_v, Lv)                    # D_v ≈ 5–20× D_u
# Reaction (Gierer-Meinhardt: f = u²/v − μ_u·u, g = u² − μ_v·v)
u2 = Mul(u, u)
inv_v = Reciprocal(v)                    # care: v must stay > 0; clamp with Add(v,ε)
react_u = Sub(Mul(u2, inv_v), Mul(μ_u, u))
react_v = Sub(u2, Mul(μ_v, v))
u_new = Add(u, Add(diff_u, Mul(Δt, react_u)))
v_new = Add(v, Add(diff_v, Mul(Δt, react_v)))
```

All ops legal: Conv, Mul, Sub, Add, Reciprocal are in core ai.onnx and don't carry subgraph attributes. `Reciprocal` is potentially unstable; use `Div` from a small-floor `Add(v, ε)` instead.

For a **Schnakenberg model** (no Reciprocal needed):
  ∂u/∂t = D_u ∇²u + a − u + u²v
  ∂v/∂t = D_v ∇²v + b − u²v
which is purely Conv + Mul + Add + Sub. Cleaner.

Canonical 2-Conv kernel weights (5-point Laplacian): same as J.22 — initializer `[[[[0,1,0],[1,-4,1],[0,1,0]]]]`. For better isotropy use 9-point (Wikipedia "Discrete Laplace operator"): `[[0.25,0.5,0.25],[0.5,-3,0.5],[0.25,0.5,0.25]]`.

## Cost analysis
Per Schnakenberg-style step on 2 channels (u, v) packed into 2 of the 10 channels:

- 1 Conv 3×3 depthwise (groups=2, fixed Laplacian for both u,v): 9·1·2·900 = 16,200 MACs, 18 params, 7,200 B mem.
- 4–6 elementwise ops (Mul, Add, Sub, u²·v cubic term): each ~7,200 B intermediate. Say 5 intermediates ≈ 36,000 B.
- Per step ≈ **52,000 cost**, similar to single-channel diffusion.

To produce a *visible* Turing pattern on 30×30 from random small perturbations:
- Linear analysis says e-folding time is ~1/(unstable eigenvalue), typically ~10–30 in dimensionless units. So ~30 steps to grow the instability to O(1) amplitude, then another ~50–200 to lock the pattern.
- **Realistic minimum: ~50 steps. Realistic stabilization: ~100–500 steps.**

Cost for 50 steps ≈ 2.6M → ln 14.77 → **10.23 pts**.
Cost for 100 steps ≈ 5.2M → ln 15.46 → **9.54 pts**.
Cost for 200 steps ≈ 10.4M → ln 16.16 → **8.84 pts**.

But: at 50 steps the pattern is **not stabilized**, so the `(result > 0.0).astype(float)` thresholding will produce noisy / drifting output that almost certainly won't match `expected_output` exactly. **Functional correctness is the binding constraint**, not cost. ARC scoring is binary: incorrect = 0.

## ARC task fit
Searching across ARC-AGI-1's 800 tasks (Hodel DSL coverage, Lewish 2025 taxonomy review, Concept-ARC, ARC-Heavy):
- **No clear "convert spots to stripes"** tasks. The closest analogues are tasks where regular periodic textures appear, but those are usually given as input (a periodic texture to recognize / extend) — not generated from random initial conditions.
- A few tasks have output grids with regular dot patterns or stripes, but the *rule* is always combinatorial (e.g. "place a dot in every cell where row + col is even"), not pattern-formation.

So: **Turing-pattern morphogenesis has effectively no native ARC task family.** Its theoretical interest is high (it's the canonical model of biological morphogenesis), but ARC was not designed with morphogenesis in mind.

The **only** scenario where this might pay off: an ARC-GEN-100K or private holdout task explicitly contains "from a noise input, output the canonical Turing-stripe pattern." This is unlikely.

## Comparison with alternatives
- **vs. J.22 Gray-Scott RD:** Turing patterns ARE a special case of RD. Gray-Scott in the right regime *is* a Turing system (Pearson 1993 explicitly catalogues Turing-pattern subregions of GS parameter space). They share the exact ONNX recipe. Use J.22's framework; don't treat Turing as separate. The only thing this section adds is a *classification* of which (D_u, D_v, kinetics) regions produce stripes vs spots, useful only if you know an ARC task targets one.
- **vs. J.4 ESN:** Turing has interpretable kernels; ESN has random. Turing > ESN per step.
- **vs. J.27 LBP:** LBP needs evidence; Turing is unconditional pattern formation. Different problem classes.
- **vs. J.39 CCA:** CCA is the discrete-state cousin of Turing/RD with comparable cost-per-step but faster apparent dynamics (3–5 steps see structure). For ARC's discrete colors, CCA > Turing.
- **vs. NCA (Mordvintsev et al.; "Differentiable Programming of Reaction-Diffusion Patterns" Mordvintsev–Randazzo–Niklasson ALIFE 2022):** trained NCA can implement Turing-style patterns with shallower depth than naive Euler RD; but you can't train in NeuroGolf, so the depth advantage is lost.

## Verdict & recommended next experiments
**Verdict: skip Turing patterns.** They are dominated by Gray-Scott RD (J.22) at every step count, and ARC has no obvious task family for them. Treat this section as the **classification reference** for choosing parameters within J.22, not a separate architecture.

If you must run experiments:
1. **Schnakenberg 30-step** as the cheapest test of pattern formation: ~1.6M cost → 10.7 pts. Verify on a synthetic "make stripes from noise seed" task. Almost certainly won't match an ARC output exactly.
2. **Skip** all Reciprocal/Div-bearing models (Gierer-Meinhardt) — numerical instability without `Clip` or `Min` will cause divergence at step counts much shorter than convergence.
3. **Skip** stripe/spot pattern targets unless you discover one in the private holdout.
4. **Cross-reference into J.22**: use Turing wavelength formulas (λ ∝ √(D_v/D_u) — Caltech BE150 lecture) only as a guide for *how strong* to make the diffusion ratio if a pattern-relevant task appears.

## Sources
- Turing 1952, "The Chemical Basis of Morphogenesis," Phil. Trans. R. Soc. B.
- Murray, *Mathematical Biology* — Turing instability conditions.
- Caltech BE150 lecture 21, "Turing patterns," be150.caltech.edu/2019/handouts/21_turing.html — wavelength of fastest-growing mode.
- Meinhardt & Gierer 2000, local activation / lateral inhibition.
- Gambino et al. 2014, "Turing instability and pattern formation in an activator-inhibitor system with nonlinear diffusion," arXiv:1403.0351.
- Wavelength of a Turing mechanism, Sci. Rep. 11, 7321 (2021) — meshwork pattern morphology vs wavelength.
- "Pattern formation mechanisms of self-organizing reaction-diffusion systems," PMC7154499 — D_v >> D_u requirement.
- BioRxiv 2025.05.27.656324, "Fundamental limits on patterning by Turing-like RD mechanisms" — convergence-time scales.
- ipython-books 12.4 simulation parameters (Δt=0.001, T=9, ~9000 steps).
- Mordvintsev, Randazzo, Niklasson 2022, "Differentiable Programming of Reaction-Diffusion Patterns," ALIFE 2022.
- mrob.com/pub/comp/xmorphia — Pearson's classification of Gray-Scott regimes including Turing-like subregions.
- Wikipedia, "Discrete Laplace operator" — 5-point and 9-point stencil weights.
- ONNX Conv / Mul / Add / Sub / Reciprocal operator specs.
