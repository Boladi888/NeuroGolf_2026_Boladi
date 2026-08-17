# onnx‑tool constant folding behaviour (1.0.1)

- **URL:** `https://github.com/ThanatosShinji/onnx-tool/blob/main/data/ConstantFolding.md`.
- **Mechanism:** `Model(rawonnx, constant_folding=True)` walks the graph, replaces any node whose every input is a constant (initializer or already‑folded) with its computed output tensor as a new initializer. Reported reductions on real models: 7 689 → 3 343 nodes; 1 139 → 110 nodes.
- **Critical detail:** folding "depends on `onnx_tool.Node`'s `value_infer` implementation" — only ops with a registered shape/value inference rule fold. Standard arithmetic + Gather + Reshape + Transpose + Slice all have value_infer, so any pure‑constant compute graph that uses these will fold to a single initializer.
- **NeuroGolf consequences:**
  - You may build a Morton/Gray/RNS table either as a literal initializer or as `Range → arithmetic → Cast`. Both reach `params` only after folding.
  - Any computation involving the *input* tensor will not fold; design rules of thumb: keep all index tables on the constant side, use Gather/Slice to consume them.
