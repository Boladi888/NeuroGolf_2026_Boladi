# task138 — the real wall is the ONNX 52-letter Einsum budget (and how 4023 broke it once)

**Recorded:** 2026-07-12 by `bo_ng_cld_worker2` (Claude lane), task138 only.
**Result:** staged **4023** (−288, byte-equal to live Sho 4311) by folding `Ac` into node 65 using one
reclaimed einsum letter. Below-4023 is walled by the **52-letter budget**, not by rank/dtype floors
(the prior passes' arguments). This record supersedes the "WALL at 4311" verdict in
`below_4311_freeoutput_wall_2026-07-10.md` (that verdict was wrong — 4023 exists and is staged).

## Live bar
`ng_current_bar --task 138` → manifest **4311**, Sho, sha `72ffa3463b99f3a41b11a2d1ee431771769274fa99ec94a416ddfc022b3520a9`.

## Exact cost structure of 4311 (static==profile)
memory 2830 + params 1481. Node 65 is a **56-operand fused free-output Einsum** whose equation uses
**all 52 index letters (a-z, A-Z), zero free** (verified: every index appears in ≥2 operands; no danglers).
That budget is *the* design constraint of Sho's graph — it explains every "why not just…" below.

## The one slack Sho left, and the 4023 win
- `Q2_ch[2,2,2,1]` and `Q2_cl[2,5,5,1]` have a **dummy trailing size-1 dim** both contracted over letter
  `O`. Squeeze → `[2,2,2]`/`[2,5,5]`, frees `O` at **0 param cost** (params = element count, unchanged).
- Column-direction shape-slice `Ac = Einsum('uvwt,w->uvt', SHA, aLc)` (a 288 B F32 `[6,6,2]` intermediate,
  node 63) exists ONLY because there was no free letter to fold it into node 65. With `O` free, replace the
  node-65 operand `Ac:TyZ` with `SHA:TyOZ + aLc:O` (`sum_O SHA[T,y,O,Z]·aLc[O] == Ac`). Exact (SHA, aLc are
  0/1; aLc one-hot). `Ac` intermediate vanishes → memory 2830→2542. Node 63 deleted, Einsum 8→7.
- Builder: `prototypes/build_task138_foldac.py`. Gates: visible 266/266, ab byte_equal 566/566 (0 div) vs
  Team_Best, profile static==profile==4023, ng_safety PASS. sha `31e4d8045fbed9ef6e3cc034a2383fa9ae6baf6a08debdbb8c791b6245d66b59`.

## Why below 4023 is walled (all trace back to the 52-letter budget)
1. **Fold `Bc` (200 B, the SHB/aHc column-hi slice) — needs a 2nd free letter, none cheap.**
   - `O` was the only zero-cost free letter (used for `Ac`; folding the bigger of the two is optimal).
   - Freeing `n` (only index local to input#1's `sOHT/inRT/inCT` group) FAILS: its partners `R,C` are shared
     with the border-render blocks, so contracting `n` alone yields `H[b,t,R,C]=[1,2,30,30]` = **14,400 B**.
   - Freeing a channel letter `K` or `M` (e.g. `sum_K rxHI10[k,K]·Q2_ch[t,K,M]`→`[k,t,M]`) costs **~160 B**
     to save `Bc`'s 200 B ⇒ ~**−40 B net**, and it perturbs the channel path (byte-equality risk). Declined.
2. **Rank-factor the shape/coupling params — real structure, but each factor needs a fresh node-65 letter.**
   - `SHB[5,5,5,2]` = exact `indicator(v=u+w+1)` shift; **exact 2-mode rank 5 → 175 params (−75)**.
   - `CLC[2,2,6,6]`, `CHC[2,2,5,5]`: exact rank-3 `(0,1)|(2,3)` unfolds → −24 / −13.
   - `SHA[6,6,6,2]`: rank 11 in all balanced unfolds — does NOT compress (2-mode LOSES: 432→528).
   - Each factored operand introduces a contraction index → needs a free letter in node 65 (0 free) ⇒
     **cannot be applied** to the current graph. (The prior "SVD mode-0 rank is full" argument was a red
     herring — the tensors ARE low-rank in the balanced unfoldings; the blocker is letters, not rank.)
3. **"Factor `input` out → F16 render (~3700)" is infeasible, not merely out-of-scope.** `input` appears
   **bilinearly** (operands `bnRC` and `bmrc`). input#2 is a separable *placement* whose contribution is
   output-sized; any split either materialises a 36,000 B `[1,10,30,30]` plane (Onnxadv #89) or leaves
   input#2 in node 65 (still F32-locked). No small F16-safe pre-contraction of input#2 exists.
4. Plane-based families (index-plane 3600 B, CumSum-scan 7200 B) remain worse — see the 07-10 records.

## The ONE real remaining lever (unproven; large effort)
A **from-scratch re-derivation of node 65 that uses ≤50 letters** (leaving 2-3 free) would unlock:
fold `Bc` (−200 B) + factor `SHB`/`CLC`/`CHC` (−112 params) ⇒ est. **~3700** (a further ~8%). Candidates to
cut letters: the 4 border-render blocks each burn ~7-8 distinct letters (`selR1/R2/C1/C2` + their own
`CHC/CLC/rxHI30/rxLO30`); if two opposite borders can share a residue chain, several letters free up.
This is an algorithmic rewrite (must re-pass 200k truth + raw nan/inf scan), not a peephole. NOT attempted
here — the 4023 byte-equal fold is the disciplined, shippable win for this pass.
