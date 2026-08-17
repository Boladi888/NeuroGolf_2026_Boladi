# task047 hard record — perfect-25 ceiling audit + rank-2 floor (2026-07-07, bo_ng_cld_worker2)

**Live bar:** cost **60**, sha `6b007e6d6665d31a941916cb0bdb5b7eb482a88e99164ab1e7db49cdc15b3a3c`
(Vamshi, manifest 2026-07-02). Graph = single Einsum `nchx,tc,ndyw,td,ot->nohw`,
inits F[3,10]=30 + C[10,3]=30, free output. `profile_real_cost` = static 60 = profile 60
(memory 0 + params 60, 265 real inputs). This is a rank-3 bilinear cross-draw, **not** a
perfect-25 identity task — the prompt's "team_best 60 -> floor_est ~1" premise is a rumor.

## DEAD ROUTE 1 — perfect-25 zero-param direct-output family (cost <=1)
- Built `cand_identity` (Identity) and `cand_transpose_hw` (Transpose perm=[0,1,3,2]) —
  the only attr-only / no-payload ops that preserve the [1,10,30,30] color contract.
- `ng_gate visible --task 47`: **Identity 0/265**, **Transpose_hw 0/265**. One visible miss = dead.
- **Why impossible (cardinality proof, not just "tried"):** the ARC input has exactly 2 non-bg
  cells (one 8, one 7); the output is a full row+col cross ≈ 30 non-bg cells. Identity /
  Transpose / flip / rotate / any index permutation only *repositions* cells — it preserves the
  multiset of cell values, so it can emit at most 2 non-bg cells. It can NEVER equal a 30-cell
  cross. No no-payload direct-output op is visible-correct. Perfect-25 route is closed.

## DEAD ROUTE 2 — rank-2 bilinear Einsum (F[2,10]+C[10,2] = cost 40)
Decode (`ng_harness/encode.decode_to_grid`) requires **exactly one color channel > 0 per cell**
(sign pattern, not argmax): winner logit > 0, all other 9 channels <= 0.
- Reduce structure: every row/col is type P / E8 / E7. Shared F ⇒ each type has one feature
  vector u in R^t; feature(r,c) = u_r (elementwise*) u_c; logit[o] = C[o,:]·feature.
  6 distinct feature points; target color table PP→0, E8E8→8, E7E7→7, {PE8}→8, {PE7}→7, {E8E7}→2.
- **Analytic infeasibility:** the three *diagonal* combos PP, E8E8, E7E7 give feature points that
  are element-wise **squares** ⇒ all lie in the closed first quadrant of R². They must decode to
  three *different* colors (0, 8, 7), i.e. be separated into three different `g·x>0` half-planes
  through the origin. Three distinct directions inside a ≤90° cone cannot be trichotomized by
  origin half-planes: the middle-angle point's two neighbours are each <90° away, so any half-plane
  containing the middle point also contains at least one neighbour (β must be both <0 and >0 —
  contradiction). Hence **rank ≥ 3 is forced**.
- **Numerical confirmation** (`prototypes/rank2_feasibility.py`, squared-hinge margin=1, L-BFGS,
  seeded restarts): t=2 best loss **3.99** across 180 restarts (never improves off the floor);
  t=3 best loss **2.5e-3** across 120 restarts (converges — the deployed 60-graph is existence proof).

## FLOOR ARGUMENT — 60 is the architectural floor
- t = 3 forced (above). Input channel axis c = 10 forced by the [1,10,30,30] contract ⇒ F = [3,10]
  = 30 params (pre-slicing channels would materialize a memory intermediate). Output channel axis
  o = 10 forced by the output contract ⇒ C = [10,3] = 30 params (zero rows for the 6 unused colors
  are still counted in the dense tensor; storing C[4,3] + scatter into channels 0/2/7/8 forces a
  ScatterND/Gather that materializes a [1,10,30,30] ≈ 9000 B memory intermediate — dwarfs the 18 B
  param saving). Einsum output is free. ⇒ **cost 60 = 30 + 30 is the forced floor** for the bilinear
  route, and every non-Einsum alternative reintroduces ≥ thousands of bytes of counted memory.

**Verdict: WALL at 60.** No sub-60 candidate exists in the bilinear family; perfect-25 (<=1) is
categorically impossible for this cross-draw transform. One remaining (low-confidence) hypothesis
if ever revisited: a route that reads only channels 7/8 with a shared constant-bias trick — but the
always-30 one-hot column-sum that supplies the background bias is exactly what F[3,10] already
bundles for free, so splitting it out has so far only *added* params. Do not re-run perfect-25 or
rank-2 here.
