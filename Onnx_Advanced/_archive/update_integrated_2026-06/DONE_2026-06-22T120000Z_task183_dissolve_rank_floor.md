# task183 dissolve floor — free-output Einsum loses when the per-channel mask is a high-rank data-dependent scatter

## What I discovered (transferable, extends #109/#89/#116)

The "dissolve the materialized one-hot into a scorer-invisible in-kernel Einsum temp" lever
(#89/#188/#109) has a **rank gate** that is the deciding factor, not the output size:

- A free-output multi-term Einsum `output[k,r,c] = Σ_t K[t,k]·R[t,r]·C[t,c]` is cheap ONLY when the
  per-channel output mask decomposes into **few rank-1 terms** (T small): separable row⊗col stripes
  (task188), solid rectangles / box-moves (task234). Cost = `(T·10 + 2·T·W)` selector bytes.
- When the mask is an **arbitrary data-dependent scatter** whose matrix rank reaches the full grid
  dimension on a meaningful corpus fraction, T collapses to **one term per cell** (T = H·W), and the
  selectors `R,C [H·W, W]` are **F32-LOCKED** (ORT rejects mixed F32-input/F16-operand Einsum — the
  #109 carrier tax). The selector bytes then DWARF the materialized one-hot you were trying to delete.

**Measure the rank FIRST** (cheap: `np.linalg.matrix_rank` over ~400 fresh ARC-GEN masks). If the mask
is commonly rank>1 / near-full-rank, the free-output dissolve is dead before you build it. This mirrors
the #110 "ray-fill rank gate" verdict already in the INDEX top-10, now confirmed for the one-hot-dissolve
class too.

## Where (task183, NeuroGolf)

- Rule: cyan cells in inner size×size (`size∈{2,4,6}`) recolored by quadrant to 4 data-dependent corner
  colors; non-cyan → black 0; output compact size×size.
- The cyan scatter rank over 400 pairs: `{1:91, 2:74, 3:75, 4:51, 5:65, 6:44}`, mean 3.14, full-rank-6
  in 11%. So T=36 (per-cell) is forced.
- Priced dissolve routes vs the current 360 B `onehot6 [1,10,6,6]`:
  - free-output Einsum to 30×30: R,C[36,30]F32 + K[36,10]F32 = 10080 B
  - free-output Einsum to 6×6 then Pad: R,C[36,6]F32 + K[36,10]F32 = 3168 B
  - #116 direct OneHot, index padded [1,1,30,30] U8 = 900 B (+ off-grid sentinel)
  - #116 OneHot at [1,1,6,6] = 360 B (IDENTICAL to current Equal-tail — no gain)
- Verdict: 839 (memory 760 + params 79) is the forced floor. Team-best re-scored 839, profile==static==839,
  visible 265/265, truth-full 200000/200000. No sub-839 candidate; no-candidate session.
- Build/proof detail: `The_400_Tasks/Task_by_Task/task183.md` "2026-06-22 w-183 dissolve floor proof".

## Why it matters

task183's whole prompt was built around the dissolve hypothesis (#89/#109/#116). The corpus rank measurement
falsifies it in ~2 minutes and converts an open "attack the 360 B tail" into a closed forced-component proof.
The reusable rule: **#109/#89 free-output dissolve requires a low-rank (few rank-1 terms) per-channel mask —
gate it with a corpus rank histogram before pricing selectors. An arbitrary scatter forces T=cells and
F32-locked selectors that lose to the materialized one-hot.**

## What's missing from the library

#109 mentions "low-rank outer product" qualitatively and #110 has a ray-fill rank gate, but there is no
explicit "measure mask rank on the corpus before choosing free-output Einsum vs materialized one-hot" decision
procedure with the F32-selector-vs-one-hot byte law for the *one-hot output dissolve* case (as opposed to the
ray/interval render in #110). Suggest extending #109 (or a one-line by_symptom route under "I want to dissolve
a materialized one-hot") with: "rank-gate the per-channel masks first; T ≥ max corpus rank; selectors are
F32-locked at `2·T·W·4 + T·10·4` B — if that exceeds `10·H·W` the materialized one-hot wins."
