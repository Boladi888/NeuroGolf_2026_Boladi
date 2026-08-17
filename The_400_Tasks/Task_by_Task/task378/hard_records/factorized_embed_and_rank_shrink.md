# task378 hard-record — factorized-family deep shaves are dead/out-of-scope

- Live bar at session (2026-07-03): **2654** (memory 1936 + params 718), owner Boladi,
  sha `d30e61b3d2e68fd4f30df196ed557c28ca019b7754fc3d613aad0810eec10cfe`
  (K-NeuroGolf-OnnxlyFans manifest.csv). This session shipped **2614** (−40 B) via a
  pure plumbing shave (see `staged_candidates/task378_factorizedfinalgraphsurgery_cost2614/`).
- Family: factorized final-output Einsum
  `bzc,qz,bmk,qm,kij,qu,qv,bur,bvs,ir,js,rR,sS->bcRS` → renders `[1,10,30,30]` directly.

## Routes attempted and rejected (do NOT re-explore without a rule re-derivation)

1. **Remove/shrink `embed12x30_f16` (360 params = 50% of params).** DEAD.
   It is an `[I12|0]` identity used twice (row axis rR, col axis sS). Its only job is to
   keep every working tensor 12-wide so the sole 30×30 tensor is the FREE graph `output`
   (score excludes `input`/`output`). Rendering the Einsum at `[1,10,12,12]` then `Pad`
   to 30×30 exposes a 1440-elem F16 = 2880 B intermediate; net +2528 B. Precompute
   `basis30 = basis12 @ embed` is impossible: `r` is shared by `row_gate (bur)` AND
   `basis12 (ir)` before the embed contracts it, so embed acts on the product, not on
   basis alone.

2. **Shrink the factorization ranks** (q=8, u=v=6, z=i=j=3): `monomial_map [6,3,3]=54`,
   `row_map/col_map [8,6]=48 ea`, `qcoef [8,5]=40`, `basis12 [3,12]=36`, `qcolor [8,3]=24`.
   These are one-hot selectors over the hand-derived render (const + ring + 4 diagonal
   rays over a degree-2 coordinate polynomial). Ranks are pinned by the rule. Shrinking =
   a full re-derivation of the factorization → out of scope per prompt ("not a floor-proving
   task", "expected wins tens/low hundreds, not a new architecture"). Not attempted; flagged
   as the WALL.

3. **F16 the scoring plumbing** (`row_moment`/`col_moment`/`masked_score`, ~480 B of F32
   [1,10] tensors). DEAD: `neg_score=-100000` overflows F16 (max 65504); `score_scale=4096`
   risks flipping the structural-color ArgMax tie-break. Must stay F32.

4. **Keep index chain 4D to save another 40 B.** DEAD: 4D `first_row` collides with 2D
   `row_moment` in `Sub` → broadcast `[1,10,1,10]`. Cast-then-squeeze-back (the shipped
   shave) is the clean −40.

## One remaining hypothesis (unverified, low-confidence)
The gate machinery (`row_gate`/`col_gate` F16 [1,6,12]=144 ea + 6 BOOL [1,6,12]=72 ea =
720 B memory) is the single biggest memory block. It is minimal for 2 comparisons × 2 axes.
Only a route that reduces the u=v=6 interval count (a rule re-derivation) touches it.
`pool_values` (MaxPool Y, 40 B) is dead but mandatory — no ONNX op gives a per-channel 2D
argmax flat index without materializing a flattened [1,10,900] input (36 KB, far worse).
