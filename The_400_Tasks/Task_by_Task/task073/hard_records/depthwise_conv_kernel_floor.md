# task073 — WALL: cost 40 is the forced floor (depthwise-conv kernel-height floor)

**Live bar:** cost **40** (manifest, owner Aditya, sha256 `e4316bf775feba0bc9dc82c0472b0b990aefde5fb40341dc51ca6d6f221e9fac`), refreshed 2026-06-19.
Recorded by `bo_ng_cld_worker1` (Claude), batch 53, 2026-07-07.

## The transform (recon — prompt's "perfect-25 zero-cost" rumor DISPROVEN)
Per-column gravity, NOT a whole-grid identity/transpose/flip/rotate. 5×5 grids, colors {0,1,5} only.
Every input column is one of exactly two archetypes:
- **type-a**: bg,bg,`1`,`5`,`5` (a `1` at row 2 with a `5` below, base row all-5). Output: the `1` falls to the base → bg,bg,bg,`5`,`1`.
- **type-b**: bg,bg,bg,bg,`5` (base only). Output: unchanged.

LIVE = a **single depthwise Conv**, group=10, weight `(10,1,4,1)`=40 params, pads `[3,0,0,0]`, kernel `[4,1]`:
ch1 `[0,1,0,0]` (shift 1s down 2), ch5 `[0,0,-1,1]` (`out5[i]=in5[i]−in5[i−1]`, deletes the base 5 the 1 replaces), ch0 `[-1,1,0,1]` (bg extension), other 7 channels zero-kernels (paid but their all-zero input → 0 output regardless). Cost 40 = 40 params, **0 intermediates**.

## Decode (the trap that killed the shortcut)
Gate decodes **per-channel `(out>0)`**, NOT argmax (`tools/ng_harness/session.py:run_raw`, compared `np.array_equal` vs `encode_grid(output)`). So ALL 10 channels must match the one-hot at every cell — ch0 (background) must be `>0` exactly at output-bg cells, including the **vacated cell** (row 2 in type-a, where the `1` was).

## Attempted routes & exact failures
| Route | Cost | Gate result | Why dead |
|---|---:|---|---|
| Depthwise Conv **K=3** (ch0=in0; ch1/ch5 = LIVE arithmetic) | 30 | `visible 0/15 FAIL` | ch0 cannot flip the vacated cell to bg within K=3 |
| Depthwise Conv **K=2** | 20 | not gated (proven infeasible) | see forced-floor proof |
| Multi-op / cross-channel ch0 fix (Conv→mid→…) | **36,030** | `score_v2` mem=36000 | any added op ⇒ one counted `[1,10,30,30]` intermediate = 9000 elem × 4 B |
| Non-depthwise Conv (group<10) | ≥80 | — | weight = 10·(10/group)·K; K still ≥4 |

## Forced-floor proof (why K=4 / cost 40 is minimal)
1. `score_v2` counts params **per element**: cost = 10·K for weight `(10,1,K,1)`, 0 memory (verified: K=2→20, K=3→30, K=4→40). A win needs a *correct* kernel at K≤3.
2. ch0 must satisfy `(out0>0)==bg-mask(output)` for both archetypes. Using taps `{i−2,i−1,i}` (the only backward-reaching K=3, since size-preservation forces pad_top≤K−1): type-a row2 needs `a+b>0` while type-b row4 needs `a+b≤0` (same expression) → **contradiction**. Dilated K≤3 fails identically (taps must be uniformly spaced; the required offsets are `{0,−2,−3}`, gaps 2 then 1 — non-uniform).
3. Separating the archetypes needs `+self(i)` (type-b row3 vs row4), `−weight at i−3` (cancel type-b base-row false positive), `+weight at i−2` (activate type-a vacated cell). Offsets `{0,−2,−3}` ⇒ a contiguous kernel spanning `i..i−3` ⇒ **K=4**.
4. **Brute-force confirms**: exhaustive integer search (weights −2..2), all pads/dilations — NO solution at K=2 or K=3 (contiguous or dilated); K=4 has 16 solutions (incl. LIVE).
5. Any non-single-op or non-depthwise alternative pays a `[1,·,H,W]` intermediate (≥36k) or a larger weight. Single depthwise K=4 = 40 dominates.

**Verdict: WALL. cost 40 is the forced floor; do not re-explore the Conv-kernel-shrink or multi-op families.**

## One remaining (low-confidence) hypothesis
Only if a truth-corpus audit ever showed the `1` is ALWAYS at a FIXED row across the entire hidden 200k set could ch0 collapse to a fixed constant plane — but that plane would be a `[1,1,30,30]` initializer (≥ hundreds of params) or a counted intermediate, so it still loses to 40. Treat as closed.
