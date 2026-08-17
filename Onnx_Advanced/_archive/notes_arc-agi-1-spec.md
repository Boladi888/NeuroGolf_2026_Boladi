# ARC‑AGI‑1 dataset and priors

- **URLs:** `https://github.com/fchollet/ARC-AGI`, `https://arxiv.org/pdf/2505.11831` (ARC‑AGI‑2 paper, Section 1.1 reviews ARC‑AGI‑1), `https://arcprize.org/arc-agi`.
- **Format facts:** 400 training + 400 evaluation tasks; grids are integer matrices with values 0–9; max 30×30; the test‑taker must reproduce the output grid exactly.
- **Core priors invoked:** object persistence, goal‑directedness, elementary number sense (counting), basic geometry & topology (connectivity, symmetry).
- **Hodel DSL (re‑arc / arc‑dsl):** ~160–165 primitives; `objects()`, `colorfilter()`, `fill()`, plus rotate/mirror/count primitives. Multiple secondary sources confirm tasks are dominated by symmetry, mirroring, flood‑fill / connected‑component, gravity, and counting.
- **Vision‑ARC paper (`https://arxiv.org/html/2511.14761v1`):** explicitly identifies an "AND/OR/XOR" cluster of generalised‑logic tasks at one corner of its task embedding; verifies that bit‑plane / Boolean tasks are a recognised but small ARC sub‑family.
- **Project mapping for J.01/J.13/J.14:** symmetry‑completion, XOR‑style "spot the difference" / "overlay two grids" tasks, parity tasks, and flood/gravity tasks are the only families where bit‑level / coordinate‑level cleverness can plausibly outperform 1–2 Conv layers.
