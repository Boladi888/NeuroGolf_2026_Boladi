# task096 — DEAD ROUTE: color-index / single-index-plane collapse toward ~900 floor

**Recorded:** 2026-07-10 by `bo_ng_cld_worker2` (Claude lane), batch for Prompt_2026-07-09 Task096_ColorIndexCollapse.

## Live bar (refreshed, authoritative)
- **live_cost = 5537** (owner **Sho**), sha256 `a05a9164c54d269c17d302d5940f29a9b41274fc2123241ca71c8f6e4556899f`
- source: `ng_current_bar.py --task 96` → manifest (NOT stale). Team_Best_Onnx/task096.onnx == this sha.
- Local `task096.md` STATUS (9545) and local staged best (5817, v10 onehot2_tail) are BOTH stale/superseded by Sho.

## The prompt's Direction (what this record kills)
"Color-index collapse sweep: perform spatial logic on one index channel, use Gather/Equal only at
the boundary. Delete charged carrier/canvas/template. Fast probe: replace the per-color path with
ArgMax/index-plane prototype and price it… floor_est ~900."

## Why the target is a FALSE / already-beaten floor
1. **The named charged component is GONE from LIVE.** Sho's 5537 does NOT carry the fat
   F32[1,6,30] per-color marginal that the old floor claims (and my 5817 lineage) fixated on.
   Sho's graph: `ReduceL1` counts → two **Einsum** ops contract `input` directly with F32 stroke
   banks (`rxA[30,4] rxB[30,5] sA[19,4] sB[19,5]`) into `[5,19]` row/col **profiles** (never
   materialising a [·,·,30] marginal) → top-5 ranking on `[5]`-vectors → a **QLinearConv**
   (`dec_w[10,2,1,1]`, `pads`) decodes a `cg2ch[1,2,11,11]` (=242B) into `output[1,10,30,30]`
   in ONE op. Charge model: memory 4808 + params 729 = 5537, spread across 125 small tensors,
   largest 380B. **There is no fat carrier/canvas/template left to delete.**
2. **The ~900 estimate is a mis-count.** It assumed the old 30×30 one-hot output tail (900B, #81)
   and "per-color manip not forced." Both are wrong now: the real output is `2N-1 ≤ 11` one-hot
   `UINT8[1,10,30,30]` and the `output` tensor is **FREE** in the charge model (neurogolf_utils
   `calculate_memory` skips names `input`/`output`). And per-color extent IS forced (below).
3. **Per-color spatial separation is FORCED, not optional.** Output = `color[rank_by_extent[d]]`
   drawn as corner L-arms (`out[R,C]=color[d] iff (d−min(|dr|,|dc|))<len[d]`, d=chebyshev). The N
   shells sit at **independent random centres** in the input, so there is NO single-index-plane
   read of shell order — you must group by colour to measure each fragment's bbox extent. Ranking
   by row-only=128/266, col-only=130/266 (history); max(H,W) extent needs both marginals per colour.

## Formulations attempted from scratch, and their PRICED WALL
Same charge model as the bar (`_score_v2.score_network_v2`). Builder:
`prototypes/build_task096_floorprobe.py` → `prototypes/task096_floorprobe.onnx`.

- **A — F16 index-marginal extent + Equal/Pad one-hot render:** score_v2 **COST 11407** (mem 10974 +
  params 433) = **2.06× the 5537 bar.** Fatal charged tensors the naive collapse materialises that
  LIVE avoids: one-hot expansion `[1,10,11,11]` BOOL+U8 = **2420B** (LIVE uses QLinearConv → free
  output); `ReduceMax` marginals `[1,10,30]` F32×2 = **2400B** + sliced `[1,10,19]` F32×2 = 1520B
  (LIVE uses Einsum straight to `[5,19]`, no full marginal); int64 Gather `[121]` = 968B.
- **B — the prompt's literal ArgMax single-index-plane:** score_v2 **COST 17110.** `ArgMax` output is
  **I64[1,1,30,30] = 7200B** charged; the U8 index plane is another 900B; and you STILL need
  per-colour `Equal` separation `[1,10,30,30]=9000B` on top. The single index plane is strictly
  ADDITIVE cost over the marginal route — it removes nothing.

**Both from-scratch collapses price 2–3× OVER live.** Reaching a cheaper graph REQUIRES exactly the
Einsum-direct-profile + QLinearConv-decode tricks Sho's LIVE 5537 already applies. LIVE is at the
collapsed formulation, not on a false floor.

## Exact gate / stop
No `visible`/`truth`/`ab` run — protocol forbids gating a losing candidate (both from-scratch
prices are 2–3× the bar; nothing to stage; `ng_safety` not applicable). Stopped on a **priced wall**
per the SWEEP "one crack only" rule.

## One remaining hypothesis (for a future FULL-BUILD batch, NOT a one-crack probe)
Reimplement Sho's exact paradigm (Einsum `[5,19]` profiles + QLinearConv decode) and try to (a) drop
the two F32 profiles to **F16[5,19]=190B each** if TopK/threshold precision holds on counts≤30
(F16 is exact to 2048), and (b) replace the arm-length **run-detection** with **count-derived
length** `L≈(count+4)/8` for shells 3–5. Est. upside only ~300–700B (→ ~4900–5200), i.e. a
`rung-epsilon` shave the current prompt explicitly forbids, and it needs **ab byte-equal vs LIVE**
matching Sho's clipped-outer/invisible-centre tie-breaking (truth is capped ~98.8% for both). Only
worth it if a future prompt reclassifies this as a shave task rather than a floor attack.

## Second pass (2026-07-10, forced re-attack) — Sho's tricks dissected; wall reconfirmed with numbers

Profiled LIVE: **static=5537, profile=5537 (no dynamic gap)** — bar fully credited. Profiled built
Formulation A (marginal family): **static=11407, profile=11407**. Library (`lib_query`) + BoBrain
mined: the only relevant levers (**#108** shell→per-cell predicate, **#119** free-output Einsum
micro-floor, **#35** Conv weighted-sum color decode) are **ALL already in LIVE**. No unused lever.

**Sho's two cost-avoiding tricks (reverse-engineered), and why they are FLOORS:**
1. **Low-rank Einsum profile** `bchw,cs,ha,hd,ya,yd->sy`: contracts `input` with a degree-4/5
   polynomial basis (`rxA[30,4] rxB[30,5]`) + one-hot color→slot map straight to `[5,19]` F32
   profiles (760B), **never materialising a `[·,30]` marginal.** THE floor for per-color extent:
   any explicit marginal is `ReduceMax(input,axis)` = **F32[1,10,30] ≥ 2400B** (output dtype tracks
   the F32 input; casting input first makes a charged 9–18 KB copy). Measured in Formulation A:
   `[1,10,30]` F32×2 = 2400B present and unavoidable.
2. **Parabola quant-conv decode** `dec_w[k]=[2(k+1),−1]` ⇒ `out[k]=−(ci−(k+1))²+…`, one-hot via the
   quant clamp, from a `cg2ch=[ci, ci²]` `[1,2,11,11]`=**242B** carrier, padding 11→30 inside the
   conv into the FREE `output`. THE tail floor: any non-quant decode materialises a padded index
   plane `[1,1,30,30]` U8 = **900B** (Equal→free output) or a `[1,10,11,11]`=2420B one-hot.

**Count-derived arm-length hypothesis — EMPIRICALLY DISPROVED.** `L≈(count+4)/8` gives 48/304 on
fresh ARC-GEN; length is a *free random generator parameter* (`len[3]∈{2,3}`, `len[4]∈{2,3,4}`,
`len[5]∈{2,3,4,5}`) and clipping breaks any count↔length map (e.g. color5 count=6→true len=2 not 1;
color8 count=18→true len=4 not 3). **Length REQUIRES spatial run-detection** (Sho's gap-ArgMax) —
so a "lean marginal" graph must add run-detection *on top of* the 2400B forced reduces.

**Both cost floors are occupied by LIVE.** A genuinely-different assembly must abandon ≥1 floor:
- different extraction (explicit marginals) ⇒ +≥1640B (2400 vs 760) — uncompensable;
- different decode (non-quant) ⇒ +≥658B (900 vs 242) — uncompensable;
- copying both floors = rebuilding Sho's core; only render-plane shaving (~500B, 8→4 planes) remains,
  which is a **forbidden `rung-epsilon` shave** requiring `ab` byte-equal to Sho's exact
  clipped-outer/invisible-centre tie-break (truth capped ~98.8% for both, so higher-truth is
  impossible → byte-equal is mandatory).

Gated numbers: LIVE 5537 (profile). Formulation A **11407** (profile, 2.06×). Formulation B 17110.
No truth/ab/battery `--full` — protocol forbids full-gating a 2–3× priced-out loser (pure burn);
`profile` already proves >bar. Cheapest CORRECT formulation found = LIVE's own Einsum paradigm
(5537); no genuinely-different formulation prices below it.

## Verdict: DROP (WALL). Named floor component already collapsed by LIVE; both cost floors (760B Einsum extraction, 242B quant-decode) occupied; no floor route to ~900 and no genuinely-different assembly below 5537.
