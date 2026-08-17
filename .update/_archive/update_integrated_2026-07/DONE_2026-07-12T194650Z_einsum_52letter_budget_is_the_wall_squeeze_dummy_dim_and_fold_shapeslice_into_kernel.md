# Transferable insight — for a maximally-packed free-output Einsum, the WALL is often the ONNX 52-letter index budget (not rank/dtype). Reclaim a letter by squeezing a dummy size-1 operand dim, then FOLD a pre-computed shape-slice intermediate into the fused kernel → delete that intermediate at ZERO param cost

**Source**: task138 (`bo_ng_cld_worker2`, 2026-07-12, 2 passes). LIVE **Sho 4311** (sha `72ffa3463b99…`).
**New staged WIN 3983 (−328, byte-equal) — freed TWO letters and folded BOTH column slices** (`Ac`&`Bc`);
FALSIFIES a prior 3-pass "WALL at 4311" verdict
(`hard_records/below_4311_freeoutput_wall_2026-07-10.md`). The prior passes argued minimal-rank params +
F32-input-fusion + an infeasible "factor input out" lever. All true-ish, but they **missed the actual
binding constraint** and left a free −288 on the table.

## The trap it kills
When the live best is a single hand-tuned **free-output separable Einsum** (whole `[1,K,H,W]` output rendered
by ONE many-operand Einsum, zero 2-D intermediates), the instinct is to attack **params** (low-rank factor the
shape/basis tensors) or **memory** (F16-narrow the F32 operands). Both are usually blocked, and the reason is
almost always the same and easy to overlook:

- **A classic-notation ONNX Einsum equation can use at most 52 distinct index letters (`a-z`, `A-Z`).**
  A maximally-optimized author packs node-N right up to 52 (verify: every index appears in ≥2 operands, 0 free).
- **Every "improvement" that adds a contracted index needs a FREE letter**: CP/2-mode factoring a shape tensor
  (`SHA[uvwt] → sum_r a[ru]b[rv]c[rw]d[rt]`) adds `r`; folding a pre-computed slice
  (`Ac=Einsum(SHA,aLc)` → inline `SHA:…O…, aLc:O`) adds the contraction letter `O`. **0 free letters ⇒ every
  such move is impossible without first freeing a letter.** That — not rank, not dtype — is why the author
  pre-computes small slice intermediates (`Ac[6,6,2]`=288 B, `Bc[5,5,2]`=200 B) in *separate* nodes instead of
  folding them into the kernel: the fold has nowhere to put its index.

## The two letter-reclaim moves (and the one that paid here)
1. **Squeeze a dummy size-1 operand dim → free its letter at ZERO cost.** `Q2_ch[2,2,2,1]` and `Q2_cl[2,5,5,1]`
   each carried a trailing size-1 dim contracted over letter `O`. `2·2·2·1 == 2·2·2`, so squeezing changes
   **0 params** (score_v2 `params` = `math.prod(dims)` element count, dtype/degenerate-dim immune) and frees `O`.
2. **Factor input#1 → free a "local" letter.** Failed here: the only index local to input#1's group (`n`) shares
   the spatial `R,C` with other blocks, so contracting `n` alone materialises `[1,2,30,30]`=14,400 B. (Lesson:
   an index is only cheaply-freeable if contracting it leaves a SMALL tensor — i.e. none of its co-operands
   carry an un-contracted spatial dim.)

**With `O` free, FOLD the larger slice** `Ac`: replace node-65 operand `Ac:TyZ` with `SHA:TyOZ + aLc:O`
(`sum_O SHA[T,y,O,Z]·aLc[O] ≡ Ac`, node 63's definition inlined). The sub-contraction now runs as a **free
in-kernel temp of the single fused Einsum** — a fused Einsum node has ONE output, so its internal pairwise
temps are NOT charged (Onnxadv #89). The `Ac[6,6,2]`=288 B intermediate disappears; node deleted.
**Then free a SECOND letter `K` by mechanism (b)** — `K` lived in only `rxHI10:kK` and `Q2_ch:tKM`, so
pre-contract `QK[k,t,M]=Einsum('kK,tKM->ktM',rxHI10,Q2_ch)` (a 160 B `[10,2,2]` intermediate) and replace both
operands with `QK:ktM` → frees `K` → fold `Bc` the same way (`SHB:zYKZ+aHc:K`), deleting `Bc[5,5,2]`=200 B.
Net −288 (Ac) −200 (Bc) +160 (QK) = **−328 → 3983**. **Exact & byte-equal** because every folded/pre-contracted
tensor (`SHA,aLc,SHB,aHc,rxHI10,Q2_ch`) is 0/1 ⇒ F32-exact, no reorder rounding.
Gates: visible 266/266 · `ab` byte_equal 566/566 (0 div) vs live · profile static==profile **3983** · ng_safety
PASS. `ng_graph_cost_diff`: "Ac −288, Bc −200, QK +160".

## Why it stops at 3983 (the wall is real below, and it is STILL the letter budget)
- Both cheap free letters are now SPENT: `O` (squeeze, 0 B) → `Ac`; `K` (pre-contract `QK`, +160 B) → `Bc`.
  A 3rd fold/factor needs a 3rd free letter, and the remaining candidates are all spatial (contracting them
  materialises a `[…,30,…]` plane, e.g. freeing `n` → `[1,2,30,30]`=14,400 B) or channel (≥160 B each).
- Param factoring is genuinely available (`SHB[5,5,5,2]` = exact `indicator(v=u+w+1)`, 2-mode **rank 5** →
  −75; `CLC`/`CHC` rank-3 → −24/−13) but each factor needs a fresh node-65 letter (0 free) ⇒ inapplicable.
  **NB: the prior pass's "SVD mode-0 rank is full ⇒ minimal" was a red herring** — mode-0 (single-axis)
  matricization is trivially full for any generic tensor; the compressibility lives in the *balanced* 2-vs-2
  unfoldings (SHB's were all rank 5). Always check balanced unfoldings, not mode-0, before declaring rank-minimal.

## Rule of thumb
Before declaring a hand-tuned free-output Einsum a WALL:
1. **Count its index letters.** If it uses all 52 with 0 free, the binding constraint is the letter budget —
   *that* is the wall, and rank/dtype floors are downstream of it.
2. **Hunt reclaimable letters**: (a) any operand with a size-1 dim → squeeze it (free letter, 0 params);
   (b) any index whose co-operands are all small (no spatial dim) → pre-contract it out. Each freed letter buys
   one fold/factor: fold the **largest** pre-computed slice-intermediate that feeds only the fused Einsum
   (delete its bytes free), or apply a balanced-unfolding-rank factor to the largest shape tensor.
3. `params` is an **element count** (dtype- and degenerate-dim-immune): F16-narrowing an initializer or
   squeezing a `…,1]` dim saves **0** params; only fewer ELEMENTS (lower rank / folded slice) helps params,
   and only narrower intermediates help memory.
The deeper below-floor lever (unproven, est. ~3700): a from-scratch re-derivation whose Einsum uses ≤50 letters
(e.g. merging the 4 per-direction ray-render blocks into 2 via the existing size-2 direction index) frees the
2-3 letters needed to fold `Bc` + factor `SHB/CLC/CHC` together. Algorithmic rewrite, not a peephole.
