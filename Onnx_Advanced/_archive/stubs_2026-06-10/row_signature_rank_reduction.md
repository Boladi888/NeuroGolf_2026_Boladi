# row_signature_rank_reduction.md

This note treats **row-signature / rank-reduction** as a reusable family for ARC-style grids \(X \in \{0,\dots,9\}^{N\times M}\), with \(1 \le N,M \le 30\). The practical ONNX-static recipe is: compute a compact per-row signature, reduce or reorder rows with `TopK` or exact pairwise comparisons, then assemble output rows with `Gather` or a dense equality/permutation mask. ARC’s official format is a grid of integers 0–9 with grid size capped at 30×30, and ARC solution practice already includes DSL primitives for sorting, deduplication, horizontal/vertical splitting, and row/column compression even though “row-signature” is not a formal benchmark taxonomy.

**ARCHIVED (Phase-2)** — extractions live in techniques.md (Wins integrated); full text: `_archive/row_signature_rank_reduction.md`.
