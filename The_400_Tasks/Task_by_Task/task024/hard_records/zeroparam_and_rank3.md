# task024 — DEAD ROUTES (batch53, bo_ng_cld_worker3, 2026-07-07)

Live bar at audit: **cost 72**, sha256 `e56f9abe3f7eb1114d19deb96b74b5761573f7184450eefabbafbf476fd68fef`
(manifest, owner Vamshi, 2026-07-02). Authoritative via `tools/ng_current_bar.py --task 24`.

## The task (derived — was `Rule: unknown`)
Data-dependent **cross-fill**. Per non-background cell in the input:
- color **1** or **3** → paint that cell's **entire row** with its color;
- color **2** → paint that cell's **entire column** with color 2;
- at a row∩col intersection the **row-fill (1/3) overwrites the col-fill (2)**.
Verified on all 3 train + 1 test pairs. NOT a whole-grid geometric transform.

## Decode contract (critical — cost the earlier confusion)
`ng_harness/session.run_raw` decodes the [1,10,30,30] output by **per-channel
threshold `(out > 0.0)`**, NOT argmax. The gate then `array_equal`s that against
the one-hot of the expected grid. So at every in-grid cell the TRUE colour channel
must be `>0` and **all 9 other channels `≤0`**. Padding cells auto-satisfy (out≡0
there because RowCount/ColCount over an off-grid row/col are 0).

## DEAD ROUTE 1 — zero-param / cost-≤1 "perfect-25" family (the prompt's premise)
Identity and Transpose(perm 0,1,3,2) one-node graphs, [1,10,30,30] I/O.
- `ng_gate visible --task 24`: Identity **0/266 FAIL**, Transpose **0/266 FAIL**.
- Reason: input ≠ output in every pair (sparse points expand into full lines);
  non-square grids (10×8, 10×11, 12×11) also break transpose on shape.
- A no-payload direct-output op cannot synthesize new row/col content ⇒ **cost ≤1
  is forced-impossible**. The perfect-25 rumor is FALSE. VISIBLE-FAIL AXIOM: dead.

## DEAD ROUTE 2 — lower-rank free-output Einsum (72→48 or 72→28 shave)
The live 72 graph is ONE `Einsum` `qs,qd,ndhx,rs,rb,nbyw,sc->nchw` with
`basis(4,10)=40 + row_transform(4,4)=16 + col_transform(4,4)=16 = 72` params
(basis reused 3× → counts once; score_v2 = 72). Latent rank `s = 4`.
Tried refitting the SAME equation at rank k (params = 10k + 2k²):
- **k=2 (28 params)**: analytic-grad fit, 98.72% cell / **0% exact** → hard
  capacity wall.
- **k=3 (48 params)**: well-converged argmax fit reaches 100% TRAIN but only
  **99.9833% VAL** exact (fails ~1/6000 held-out, min logit margin −5.1). Under the
  REAL per-channel `>0` decode a quick fit gave **0% train-exact** at 700 examples.
- **k=4 (72)**: exact — reproduces LIVE. This is the shipped floor.

### Why k=3 cannot win (rigorous)
`{per-channel-`>0`-correct cells} ⊆ {argmax-correct cells}` (true ch >0 & others ≤0
⇒ true ch is the unique positive ⇒ the max). Hence per-channel exact-accuracy
≤ argmax exact-accuracy. The best-case argmax k=3 fit is **< 100%** on held-out
(99.98%), so k=3 can never reach 100% under the harder real decode ⇒ fails the
100%-truth gate ⇒ never shippable (traps §14e learning-artifact). Structurally the
`>0` decode needs **4 tied rank-1 spatial patterns** — `r1⊗1` (ch1), `r3⊗1` (ch3),
`nonfill⊗c2` (ch2), `nonfill⊗noncol` (ch0) — which pins latent rank at 4. A constant
col-profile (needed for the `⊗1` terms) also can't be synthesized exactly from
count-features at k=3.

## VERDICT: WALL at cost 72
The rank-4 free-output Einsum is the forced structural floor. No sub-72 legal
candidate via lower rank; zero-param cost-1 is impossible. The old 25-node
boolean `And/ReduceMax/Slice` route priced 110 (worse) — the Einsum already beat it.
One remaining (low-odds, out-of-scope-per-prompt) hypothesis: fold one 4×4 transform
into `basis` to reach ~56 while keeping rank 4 — untested; the fitted transforms are
non-trivial so absorption is unlikely.
