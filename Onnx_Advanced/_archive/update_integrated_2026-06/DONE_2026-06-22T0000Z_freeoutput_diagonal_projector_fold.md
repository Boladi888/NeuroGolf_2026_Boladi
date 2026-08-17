# Free-output render: diagonal Einsum projectors do NOT fold to element-wise Sum

## What I discovered
A "feature-rank reduction" trap for the #109/#119 free-output outer-product render
family (the `bpr,bpw,bos,ps->borw` shape used by Aditya's task025 team-best).

The final render Einsum `bpr,bpw,bos,ps->borw` consumes two big feature tensors
`row_feat[1,P,R]` and `col_feat[1,P,W]` (here P=26, F16, 1,560 B each = the
single biggest cost block) plus a projector `slot_projector[P,S]` that is a pure
0/1 BLOCK-SUM matrix grouping the P features into S render slots. It LOOKS like
the projector just sums groups of features, so the obvious "win" is to pre-sum
the descriptor tensors into the S-slot space with a variadic `Sum`, shrinking the
feature tensors from [1,P,R] to [1,S,R]. On task025 this dropped cost 14,267 ->
12,311 (a real-looking beat).

**It is WRONG. Visible 0/266.** The render is a DIAGONAL contraction: it forms
`row_feat[p]·col_feat[p]` on the SAME index p, then sums within each projector
group. So per slot the output is a SUM of S' rank-1 products with SPECIFIC
cross-orientation pairings (for task025: row.line·col.line + row.v_left·col.adj_left
+ row.v_right·col.adj_right + row.adj_left·col.h_up + row.adj_right·col.h_down).
Folding to `row6[s] = Σ_descriptor row_feat`, `col6[s] = Σ_descriptor col_feat`,
then `row6[s]·col6[s]` gives `(Σrow)(Σcol)` = ALL cross-terms, not just the
diagonal pairs. The descriptor types pair a specific row-descriptor with a
specific col-descriptor; that pairing is the rendering mechanism. The P-axis is
the IRREDUCIBLE rank of the renderer.

## The general rule
A 0/1 block-sum projector in a multi-operand Einsum can be folded into the
operands ONLY when each operand is contracted on a DIFFERENT (independent) index.
When two operands share the contracted index p (`...p...,...p...,p s->...`), the
projector groups DIAGONAL products `op1[p]·op2[p]`; you cannot distribute it into
per-operand sums without introducing spurious cross-terms. Quick test before
attempting any projector-fold: does more than one operand carry the projector's
contracted axis? If yes, the fold changes the math — gate visible before trusting
the lower cost.

## Where
- task025, beta_p21 worker-3, 2026-06-22.
- Build/dead-end script: `The_400_Tasks/Task_by_Task/task025/prototypes/build_task025_slotfold_v1.py`
- Baseline = LIVE = Aditya team-best, byte-identical (SHA a1a8458...), static=profile 14,267.

## Why it matters
This is the most tempting remaining lever on task025 (the 26-feature tensors are
22% of the cost) and it produces a cost that BEATS team-best while being silently
incorrect — a classic VISIBLE-FAIL trap. Documenting it stops the next worker
from re-deriving it and trusting the static cost. Generalizes to any free-output
render where a projector collapses a shared-index outer product.

## What's missing from the library
techniques.md #109/#119 cover free-output Einsum renders and rank gates, but I did
not find an explicit note that a shared-contracted-index projector cannot be
folded into per-operand sums. Suggest adding this as a one-line caveat to the
#109 free-output / #119 conditional-shift discussion (or wherever the
`bpr,bpw,...->borw` outer-product render is documented).
