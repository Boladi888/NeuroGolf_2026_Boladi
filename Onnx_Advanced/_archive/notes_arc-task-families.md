# ARC task taxonomies relevant to bit / index / parity tricks

- **URLs:** `https://arxiv.org/html/2512.07109` ("Neural Affinity Framework", procedural taxonomy with 97.5 % rule‑based classification accuracy); `https://medium.com/99p-labs/...` and `https://www.agemo.ai/resources/summer-of-arc-agi` (Hodel DSL summary, 160 primitives); `https://arxiv.org/html/2511.14761v1` (Vision‑ARC, identifies AND/OR/XOR cluster).
- **Sub‑families that map onto our five techniques:**
  - **XOR / parity / two‑grid overlay** — explicit cluster in Vision‑ARC's task embedding; canonical inputs are two halves of a grid that are XOR‑combined to produce a difference map. (J.01)
  - **Symmetry completion** — input is half of a symmetric design; output completes it. Implementable as `Slice → Transpose (mirror) → Add` or as `BitwiseXor` of original with mirror. (J.01, J.13)
  - **Flood fill / connected components / gravity / row‑fill / column‑fill** — adjacency‑heavy; benefit from row/col operations on a 1‑D linearisation. (J.13, J.14)
  - **Counting / area comparison / position arithmetic** — small integer ranges; the home turf for RNS in *hardware* but flat in onnx‑tool. (J.16)
  - **Approximate / probabilistic targets** — *do not exist* in ARC because of exact‑match scoring; this kills J.18.
