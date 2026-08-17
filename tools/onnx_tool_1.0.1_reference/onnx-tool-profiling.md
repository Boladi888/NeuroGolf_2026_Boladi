# onnx‑tool 1.0.1 — profiling and constant folding

- **URLs:** `https://github.com/ThanatosShinji/onnx-tool` (README), `https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md`, `…/data/ConstantFolding.md`, `https://pypi.org/project/onnx-tool/`.
- **MAC convention:** `1 MAC = a*b + c`. Element‑wise ops (Add/Sub/Mul/Mod/Floor/Cast/Bitwise…) count as ≥ 1 op per element; Profile.md states "Treat other instructions as 1 MAC, like cmp, add, and mov" though they are lighter physically.
- **Memory:** "Layer memory usage = Weight tensor memory + Output tensor memory. The same weight tensor will be only count once." — i.e. weights are folded into the `params` total once; activations contribute to per‑node `memory`.
- **Params:** "The total size of weight tensor. The same weight tensor will be count once like memory."
- **Constant folding (`Model(..., constant_folding=True)`):** replaces constant sub‑graphs by constant tensors. Empirically reduces graphs by 30–60 % node count on real models. **Implication for NeuroGolf:** any sub‑graph whose inputs are all initialisers (e.g. a Morton index built from Range/Add/Mul/Mod that depends only on constants) collapses into a single initializer at measurement time, paying only its byte cost in `params` and zero MACs.
- **Project mapping:**
  - Pre‑bake all permutation/index tables as initializers; do **not** synthesize them with `Range`/`Mul`/`Mod` at runtime — the latter pays MAC and memory per element until folded, but folding succeeds only when *every* upstream is constant.
  - The locked probe (`Conv1×1=90 100`, `Conv3×3=810 900`, `Conv5×5=2 252 500`) tells us that the `params` byte‑weight of a 1×1 Conv with 10 in/10 out (100 floats = 400 B) is dwarfed by per‑element memory; therefore in the cost formula the activation memory term dominates for any element‑wise op chain on `[1,10,30,30]`.
