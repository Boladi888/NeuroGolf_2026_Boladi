# task299 — sub-50 route DEAD (WALL at 50)

- **Live bar:** cost **50**, sha256 `63937e8cccb8a0981bf1e4606b6807083bfab548f949e29a4243abf836467c24`
  (manifest, owner Aditya, 2026-06-19). `ng_current_bar.py --task 299` authoritative.
- **Decode is `(output>0)` threshold** (`neurogolf_utils.run_network`), NOT argmax.
  Per active cell EXACTLY one channel must be >0 (the correct color); all others ≤0.
  LIVE truth = 3000/3000 (class-check pass), so truth-100% is the gold gate.

## LIVE cost-50 mechanism (single Einsum, memory 0)
`equation = bhry,uh,bgxc,g,uk->bkrc`, operands `[input, row_selector, input, col_selector, coeff]`
- `row_selector[2,10]` = `[ones(10); e_2]` (20). Reads row profiles: R[0,r]=6 (active
  count, also zeroes padding), R[1,r]=#color2 in row r.
- `col_selector[10]` = `ones - 4·e_8` (10). C[c]=6 on non-8 cols, 6-4·s8 (<0) on the 8-col.
- `coeff[2,10]` (20). Maps the 2 row-modes → 10 output channels.
- `out[k,r,c] = C[c]·(R0·coeff[0,k] + R1·coeff[1,k])`. The −4 sign-flip on C picks
  ch8/ch4 on the 8-col; coeff cancels bg-bleed on the 2-row (needs s2=2, always true).
- Cost = 20+10+20 = **50**, memory 0 (no materialized intermediate; output not charged).

## Attempted sub-50 route: merge row_selector≡coeff → shared T[2,10] (cost 30)
`prototypes/build_m30.py` → `task299_m30.onnx`. static=profile=**30** (memory 0).
Hand-derived + verified 21/21 under **argmax**, but **visible 0/21** under the real
`>0` threshold decode. **PROVEN INFEASIBLE:**
- View each channel as a 2-vector t_k = T[:,k]. bg rows (ρ=6·t0) are evaluated at BOTH
  a non-8 col (sign +) and the 8-col (sign −), forcing base[k]=t0·t_k to be both ≤0
  and ≥0 ⇒ **t0·t2 = 0**.
- On the 2-row (ρ=4·t0+2·t2): base[ch0]=ρ·t0 = 4|t0|²+2(t2·t0) = **4|t0|² > 0**, so the
  background channel is positive alongside ch2 ⇒ two positives ⇒ threshold decode fails.
- Independent of integer box; holds for any shared T. So the 20-param merge cannot exist.

## Other sub-50 routes ruled out
- **col_selector ≡ a row of row_selector/coeff:** col_selector must detect channel 8 with
  a sign flip; nothing else in the graph touches ch8 (rows detect ch2), and an all-ones
  row cannot flip. Irreducibly separate, and dense-charged at its full [10]=10.
- **Shared selector S[2,10] for BOTH row (contract y) and col (contract x) reads:** forces a
  full W[2,2,10]=40 output map (rank-2 in both factors) ⇒ 20+40 = 60 > 50. The asymmetric
  rank-2×rank-1 (−4 trick) is strictly cheaper.
- **rank-1 × rank-1 (both factors scalar):** a single per-cell scalar m=R·C has only 2 sign
  classes ⇒ ≤2 distinct threshold outcomes, but the task needs 4 distinct channels
  (bg/2/8/4 over the 2×2 row×col state table). Provably impossible.
- **Materialized-intermediate routes** (v4_freeout 811, bool_concat 923): any 6×6 channel
  read is ≥144 B F32 (x2+x8 = 288 floor); all ≫ 50. Only the memory-0 single-Einsum family
  lives under 100.

## Verdict
50 is the **forced floor** for the memory-0 single-Einsum family: rank-2 bilinear required,
minimal encoding = row_selector[2,10] + col_selector[10] + coeff[2,10] = 20+10+20 = 50, and
every pairwise merge is infeasible (proofs above). **No strictly-sub-50 candidate exists.**
Do NOT re-explore the merge/shared-selector families. One remaining (low-confidence)
hypothesis: a 2-op route (Einsum + one tiny elementwise) that injects the column sign-flip
without a stored 10-vector — but every such split materializes a ≥120 B per-column
intermediate, so it starts already above 50.
