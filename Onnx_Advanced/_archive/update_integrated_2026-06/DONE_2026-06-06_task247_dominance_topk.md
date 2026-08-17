# task247 — promotable findings (2026-06-06, worker-7)

Shipped-class result: LIVE 18,184 → **543 byte-equal-FULL** (v5, ab 200,269/200,269 div=0,
truth 200k/200k, vis 269/269; staged `task247_v5_eqassembly_cost543/`). −97.0%, +3.51 LB
raw-ln predicted. Generator-source-certified lemmas (task_a3325580.py). Full lineage +
fusion-aware floor proof: `task247.md`.

## Promote to techniques.md (#46-family extension) — **exponential positional dominance: first-occurrence ORDER as a free-input Einsum scalar**

1. **"Order colors/objects by leftmost (topmost/first) occurrence" needs NO occupancy plane,
   NO ArgMax, NO pairwise-rank matrix.** One free-input Einsum with an exponential weight
   vector: `f = Einsum('bchw,w->bc', input, b^-j)` ([1,10] = 40 B + 30 params). Then
   `leftmost(c) < leftmost(d) ⇔ f(c) > f(d)` — the leading column dominates the whole tail.
   - **Margin condition**: per-column (per-line) cell count ≤ B requires base `b ≥ B+2`
     (tail bound `B·b^-(L+1)·b/(b−1) < b^-L`). task247: B=10, b=16 ⇒ 1.5× exact gap ≫ F32 eps.
   - **F32 range guard**: b=16 over 30 columns bottoms at 2^-116 (normal ✓); b=32 hits
     denormals at j=29 (FTZ risk — don't). Weights 2^(-4j) are exact powers of two.
   - Replaced in task247: colocc [1,10,1,30] 1,200 B + ArgMax 80 + Cast/Transpose/Less
     pairwise machinery → 40 B + 30p. Same trick works for topmost (`'bchw,h->bc'`),
     rightmost/bottom (ascending weights), and "first nonzero scanline" selectors.

2. **TopK as rank-selector kills the BOOL→F32 rank-sum Cast.** The classic pairwise rank
   (`Less` matrix → And(ismax) → Cast F32 → ReduceSum) pays 324 B for the Cast alone
   ([1,9,9,1]·4 — ORT CPU ReduceSum takes float/double/int32/int64 only, no U8/BOOL).
   `TopK(Where(ismax, f, 0), k=K_cap)` returns the winners already sorted by the dominance
   scalar = rank order; slot k is real iff vals[k] > 0 (Greater mask). vals+idx+mask = 39 B.
   Static-k initializer (task054 grader precedent). Measured: TopK route 606 vs pairwise 1,115.
   - **Don't** sort unmasked f (losers steal slots — mask BEFORE TopK), and **don't** fold
     count into a combined sort key (`cnt·1000 + f` — F32 mantissa can't hold both scales).

3. **Tie caveat** (the one semantic divergence vs ArgMax-leftmost): equal leftmost columns —
   ArgMax-based LIVE gives both the same rank; dominance-f breaks the tie by deeper columns.
   Gate on full-corpus ab (task247: zero ties in 200,269); rotate-adversarial divergence is
   exactly this class on off-contract inputs.

## Promote to techniques.md — **bg-dominance lemma: ch0 isolation can be FREE on bounded-M tasks**

4. **When the rule bounds the max object-color count M (output-height cap, officials, or
   generator source), bg is provably the strict global count max** — `nonbg ≤ 9·M ⇒
   bg ≥ 100−9·M > M` whenever M < 10 (10×10 grids; scale for other canvases). Consequences:
   - `M = TopK(c10, k=2).values[:,1]` (bg owns slot 0) — 28 B incl. the dead-but-counted
     indices output — beats Slice+ReduceMax (40 B) and every masked alternative;
   - `Equal(c10, M)` is ch0-safe with NO slice — the winner mask can live on the raw
     [1,10] counts;
   - keep the one-hot iota at values 1..9 so an idx-0 ghost could not paint even
     off-distribution (belt+braces).
   task247: deleting both ch0-drop Slices this way took 606 → 563. The all-zero scorer
   dummy stays safe (vals=[0,0] ⇒ M=0 ⇒ row gate all-false).

## Promote to techniques.md — **u8 color-code canvas + final-Equal beats one-hot tails (Equal-assembly, small-patch form)**

5. **When the output content is a small patch (here 9×3) assembled then Pad-ded to canvas,
   carry it as a u8 color-CODE canvas, not one-hot.** `Cast(TopK-idx → u8)` + nested
   `Where(valmask, ·, 0)` + `Where(rowgate, ·, 0)` → [1,1,R,K] u8 (R·K bytes); ONE
   `Equal(canvas, chids[1,9,1,1])` expands to the [1,9,R,K] BOOL pre-Pad content. Code 0 =
   "empty" matches no chid 1..9, so slot/row gating doubles as the padding sentinel
   (task398-v4 pattern at patch scale). task247: −20 vs the Equal(iota,idx)+And one-hot tail.
   Counter-case: full-canvas content → task256's broadcast-Add output class instead (priced
   here: ≥660 F16 for a 9×3 patch — loses; wins at 30×30).

## Smaller notes

- **Einsum reduction beats ReduceSum at opset 13+ by params**: equation is an ATTRIBUTE
  (`'bchw->bc'` = 0p) while ReduceSum-13 axes is an INPUT (+2p). Same 40 B output off the
  free input.
- LIVE autopsy reminder: two identity Gathers (`[0..9]` on axes 2,3) = 8,000 B of dead
  weight survived multiple polish passes because spread-scan patterns don't check
  Gather-index-is-identity. Candidate for a new spread pattern (`gather_identity_elide`).
