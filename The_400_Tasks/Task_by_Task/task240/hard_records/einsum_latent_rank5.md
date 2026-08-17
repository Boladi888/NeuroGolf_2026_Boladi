# task240 — DEAD ROUTE: rank-5 latent refit (225p) — 246 is the forced floor

- **Live bar at proof (2026-07-03):** cost **246**, owner Aditya, authoritative source = manifest.
  sha256 `f74284bfc0e2bdeafe0a843f06c6fbe7e9e4ed0e36b0c7f8a3dfd24b05c1f39c`. Live=team_best=workbook all agree 246.
- **Attempted route:** shave the latent color-mixing core from separable-rank-6 (m=6, 96p) to rank-5
  (m=5, 75p) while keeping the forced spatial basis `basis5[30,5]=150` — a 225-param single-Einsum (−21).
- **Failed gate:** never reached threshold-correct. Threshold-aware (sign-margin) optimizer, 11 inits
  incl. a warm-start (drop `core6`'s weakest eigendirection): **best 212/266 official examples, 768 cell
  sign-violations on the full corpus**; m=6 reaches **exactly 0**. Cannot even fit a 48-example subset
  (best 156 violations) → a **capacity wall, not an optimization-budget wall**. No `ng_gate --full` run:
  no ONNX ever cleared visible, so nothing to truth/battery (WORKER_PROTOCOL gate hygiene).

## Why it's dead (forced-floor argument)

The live 246 graph is a **single zero-intermediate Einsum**
`bohw,hx,wy,xk,yl,kp,kl,lq,rp,sq->bors` (name `task240_v14_k5_m6_bilin`). It realizes a per-channel
linear spatial operator T:[30,30]→[30,30] (the channel/color axis `o` passes through untouched). Decode
gate = `(out > 0)` sign-threshold vs one-hot target, so only the **sign pattern** must match (on-cell
margin measured 0.0199; off-cells exactly 0.0).

1. **Single-op is forced.** Any 2nd op materialises a charged intermediate ≥ `[1,10,19,19]` F32 =
   14,440 B ≫ 246 (memory is charged, graph I/O is not). So the minimal graph = one Einsum whose whole
   cost is initializer params. (Conv can't do the non-translation-invariant fold/reflect; a single Gather
   needs a 900-entry index map and can't sum folded orbits.)
2. **Spatial basis ≥ 150 forced.** T's marginal rank along each 30-axis (h,w,r,s) is **exactly 5**
   (measured). Any factor touching a 30-axis needs ≥5 columns → `[30,5]=150` min. Maximal sharing (one
   basis for all 4 legs — valid: T is symmetric under r↔s,h↔w, diff 1e-4) hits exactly 150.
3. **Latent core = 96, separable-rank-6 forced.** T is separable-rank **exactly 6** under the (rh)(sw)
   grouping (σ₆=1.03, σ₇≈1e-9 — a clean gap, not a soft tail). The coupling `core6` is **full-rank-6**,
   symmetric, eigenvalues ±{1.52,1.49,0.92,0.83,0.57,0.34} (none near 0). m=4 has only 16<25 rank-1 terms
   (impossible); m=5's residual is ~1.0 magnitude = 50× the 0.02 sign-margin → can't preserve signs, as
   the empirical optimizer confirms. Every alternative latent parameterisation tried lands ≥96: eigen-
   diagonalise→156, absorb-S-into-basis→396 (breaks the 30-axis sharing), self-adjoint `O=Sᵀ`→**T is not
   self-adjoint** (diff 0.59). C is not exactly low-rank (rank 6) so no exact shave.

**⇒ 246 = 150 (basis, forced) + 96 (latent, separable-rank-6 forced) is the forced floor.**

## The prompt target was a phantom
Prompt aimed 246→~185 by "dissolving the color_f32 [1,1,19,19]=1444 B plane to U8 via dilated-Conv."
That plane + the canvas30 belonged to the **v1 / 1706 einsum-fold lineage**, which the live v14 einsum
already dissolved to **memory=0**. There is no 1444 B plane in the live graph, so 185 has no referent.
floor_est 185 is invalid; the true floor is 246.

## One remaining (low-probability) hypothesis for a future batch
The only unexploited symmetry is per-**color** structure: the einsum treats all 10 channels identically
(`o` passthrough). If some color channels are provably never markers/arms (e.g. a background channel that
is a pure complement), a channel-restricted operator *might* trim latent DOF — but that needs a Slice
(intermediate memory ≫ 246) unless expressible as another free Einsum axis. Very unlikely to pay; do not
re-run the rank-5 latent shave — it is a proven capacity wall.
