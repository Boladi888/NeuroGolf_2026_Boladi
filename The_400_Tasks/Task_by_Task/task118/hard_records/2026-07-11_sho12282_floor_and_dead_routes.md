# task118 hard record — Sho 12,282 is the structural floor (2026-07-11, bo_ng_cld_worker1)

**Supersedes the ~12,600 floor claim in `structural_floor_and_convshift_zeropad.md`** (that proof was for the
OLD directional-tie-break family). Sho's 12,282 uses a SIMPLER selection with no directional tie-break and
disproves the old floor. This session proves 12,282 is the true floor for the exact rule.

- **Live bar:** 12,282 (manifest, authoritative; `ng_current_bar.py --task 118`), sha
  `853d7e92044414765ace7f061ffca8a59da1e8f4876c0750e7760beeb91584f8`, owner **Sho**, 2026-07-09.
- Independently reproduced byte-equal this session: `prototypes/build118.py --variant repro` →
  267/267 vs expected, 267/267 raw byte-equal vs LIVE, cost 12,282. So we fully understand the LIVE graph.

## LIVE graph decode (Sho 12,282) — every tensor is forced
Carrier trick: `W1=[·,·,65,·,·,64,·,·,·,·]` 1×1 `Conv` projects one-hot → color2=65/color5=64/else 0, then
`Cast`→U8 `vc`. The two count `QLinearConv`s use **input zero-point 64**, so `(vc−64)` = +1 (color2) / 0
(color5) / **−64 (background)** → any background in the cross footprint saturates the count to 0 = a
**complete-cross detector**. Radius picked by scalar `L = max(s3) > max(s2)`. Center = `score >= Max(MaxPool(score), thr)`
(local-max AND ≥ threshold, in ONE compare). Cover = `QLinearConv(centers, selected_stencil)`; paint = covered AND color-5.

Per-tensor cost (memory 12,153 + params 129), all @ 25×28 crop unless noted:
| tensor | B | forced because |
|---|---|---|
| v (F32 read) | 2,800 | one-hot F32 input → any channel-reducing read is ≥ F32 [25×28] (Conv 2,800 < ArgMax-INT64 5,600 < Cast-whole-input 9,000) |
| vc (U8 carrier) | 700 | Cast of the read |
| s3, s2 | 1,400 | radius decision needs BOTH maxes (L balanced 137 r3 / 130 r2; `mx3>mx2` = 267/267, `mx3>=mx2` only 256) |
| score | 700 | chosen count (Where blend), feeds local-max + center |
| mp, mpt, cen, cenu | 2,800 | local-max + threshold-floor + bool + U8-cast for cover conv |
| pcount | 700 | cover conv |
| fiv, gtp, S | 2,100 | 2 independent conditions (color-5, covered) AND-ed |
| Spad | 900 | bool [1,1,30,30] — final Where mask must broadcast to [1,10,30,30] |
| params+sten+scalars | 182 | W1/col8/k2/k3/pp/pax/... all minimal for the op shapes |
**= 12,282.** Naive ablations all fail (free local checker over 267): no-localmax 233/267, no-threshold 0/267, no-color5 0/267.

## DEAD ROUTE 1 — tighter crop cr=3 (25×27), cost 11,882 (−400). Do NOT ship.
- `build118.py --cr 3`; 267/267 vs expected AND 267/267 raw byte-equal vs LIVE on the officials — but
- **`ng_gate ab --n 2000` vs LIVE = 3 divergences / 2267, byte_equal=False.** The 3 are REAL hidden failures, not synthetic noise:
  - div#842: a color-5 cell **painted at col 27** (paint bbox on the 267 officials only reached col 26 — the hidden corpus is wider).
  - div#1534 / #1876: right-edge crosses whose completeness flips because `QLinearConv` pads with zp=64 (0 contribution) while a real col-27 background cell contributes −64.
- Not rescuable by an asymmetric-pad "read col-27 but output 27-wide" trick: `fiv` needs vc@27 (extra Slice map) → nets ≥0. Crop is FORCED at 28.

## DEAD ROUTE 2 — bake threshold into QLinearConv INT32 bias (drop `mpt`), cost 11,584 (−698). Do NOT ship.
- `build118.py --feats bias_thr` (B=−(thr−1): −2 for r3, −1 for r2 → weak crosses saturate to 0, dropping the `mpt` map).
- **0/267 vs expected.** Confirmed cause (intermediate probe): `score` is correct (nonzero only at true cross cells),
  but with threshold baked, background has `score=0=MaxPool=0` so `GreaterOrEqual(score,mp)` fires on the whole
  background — 604 "centers", **602 are background false-centers** → cover spreads everywhere → over-paint.
- Excluding background needs a `score>0` map, which cancels the `mpt` saving. **`mpt` is load-bearing** — it uses the
  scalar threshold as a floor that background's 0 can't clear (one map doing local-max reference AND background exclusion).

## PASS 2 (2026-07-11, same worker) — minimal-rule study + exhaustive single-map-drop
- **Minimal-rule data study (numpy, all 267):** paint rules ranked — cover-by-ANY-complete-cross **195/267**,
  cover-by-threshold **232/267**, local-max+threshold (current) **265+/267**. So the local-max center-selection is
  provably ESSENTIAL, not over-built. Any assembly that drops it is a visible-fail.
- **Gated the cheapest genuinely-different assembly** — no-local-max (drops `mp`+`mpt`, 10 maps): `build118.py --feats no_lm`,
  cost **10,882** (−1,400), `ng_gate visible` = **233/267 FAIL**. Confirms: the cheaper assembly is a certain hidden 0.
- **Exhaustive single-map-drop (all 11 detection maps s3,s2,score,mp,mpt,cen,cenu,pcount,fiv,gtp,S):** each is load-bearing.
  Every fusion is impossible under the element-count cost model: compare→cast (cen+cenu), pool→threshold-floor (mp+mpt),
  2-condition-AND (gtp+S), and radius-needs-both-maxes (s3+s2+score) each inherently require 2 named tensors.
- Read/carrier re-confirmed minimal: F32 project (2,800) < ArgMax-INT64 (5,600) < Gather/Slice-2ch (7,200) < Cast-whole-input (9,000).

## VERDICT: WALL at 12,282 (airtight).
Every tensor forced; the cost basis (F32 read 2,800 + 11 maps×700 + Spad 900 + params) is pinned on all three axes
(read dtype/size, map count, crop). The ONLY sub-12,282 assemblies drop a provably-necessary map → visible-fail.
BoBrain `floor_est ~1,200` is unreachable (F32 read alone = 2.3× it). Sub-12,282 requires abandoning LIVE-equivalence
(lookup/Cherry13b, off-limits). Artifacts: `prototypes/t118_nolm.onnx` (sha `dfe60cf3…`, 233/267 evidence).
