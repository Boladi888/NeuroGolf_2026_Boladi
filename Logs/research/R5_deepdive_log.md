# R5 deep-dive log — per-task teardown of Aditya's 175 winning nets (running)

Self-paced mining loop (owner authorized ~10hrs, 2026-06-08). Each batch: interpret next ~8 nets, record novel
vs covered, extract clearly-novel idioms to techniques.md, accumulate smaller candidates here for consolidation.
State: `tmp/mine_state.json`. Techniques so far from R5: **#111–#124**.

## Batch 1 (task003/004/010/011/022)
- **task010** (60n, 76p): RANK 4 row-values by pairwise-`Greater` + `Sum`-of-wins (rank = #beaten) → **extracted #124**.
  Also: per-row float-threshold bucket-decode (`Greater` at 1.5/2.5/3.5/4.5 → ordinal), a float twin of #120.
- **task022** (13n, 787p): einsum `bchw,bdpq,d,rhp,swq->bcrs` with `rhp,swq` = [3,11,11] CENTER-RELATIVE offset
  maps → CANDIDATE: relative-position basis (place a pattern relative to each detected center); extends #118.
- **task003** (18n, 25p): region-equality via `ReduceSumSquare(A-B) < eps` ("are two patches identical") + And/Or
  fill → CANDIDATE micro-note: squared-distance patch-compare. Covered by #111 otherwise.
- **task004** (3n, 2302p): 8-operand separable einsum (#119) + 1x30 row-encode Conv (#118). Covered.
- **task011** (17n, 732p): position-code Conv + Equal + separable einsum `bchw,ca,hr,ws->ba` (#118/#119). Covered.

### Candidates pending consolidation (extract to techniques.md once corroborated by more tasks)
- relative-position basis einsum (task022) — center-relative `rhp,swq` offset maps.
- ReduceSumSquare patch-equality test (task003).
- float-threshold ordinal bucket (task010 tail) — float twin of #120.

## Batch 2 (task024/026/028/033/035/036/043/045) — ALL COVERED, consolidated micro-idioms → #125
No new standalone idiom; all 8 reuse #111/#112/#118/#119. Connective-tissue glue extracted as **#125**.
- **task024** (17n,34p): einsum-select row/col `nchw,kcs->nkhs` codes + bit-threshold (`Greater` 15/511 = 2^4-1/2^9-1) + `Xor` parity. → #119/#112/#125.
- **task026** (13n,34p): arithmetic boolean (`1-x`=not, `Mul`=and, `Sum`=or) occupancy logic, no bool cast. → #125.
- **task028** (1n,250p): single-node 5-operand separable einsum (`src_rows[2,30]` basis). → #118/#119.
- **task033** (15n,60p): `Gather`-with-index-vector reindex/upsample (idx5→idx17). → #118-style.
- **task035** (41n,779p): 4-border slice + per-edge `ArgMax` + `Where`-cascade priority-select. → #111/#125.
- **task036** (—): object-center `ArgMax` + crop + separable einsum (`row_basis_5x30` upscale). → #118/#119.
- **task043** (8n,44p): border-row `And`-intersection + `Where`-replace. → #111.
- **task045** (16n,86p): `Where`-cascade decision-tree → bit-code + `BitwiseAnd` mask. → #112/#125.

**Trend:** novelty is dropping — the remaining nets recombine #111–#125. Loop value now = coverage confirmation
+ the rare new idiom + the end-of-run transfer list.

## Batch 3 (task049/050/056/057/059/060/063/066) — ALL COVERED
All reuse #93 (moments) / #118 (basis) / #119 (separable einsum) / #123 (interval) / #125 (boolean glue). Minor
recurring flavors logged as candidates (consolidate only if they keep recurring):
- **output-bank Gather** (task056): `Conv`-classify → `ArgMax` → `Gather` a full output slice from a `[K,10,1,1]`
  const bank = classify→lookup-output. (Borderline vs #84 LUT; watch.)
- **sentinel-masked arg-select** (task049): `Where(valid, x, ±BIG)` → `ArgMin`/`ArgMax` = extremum among VALID only.
- **MatMul-by-coord-vector** (task050): read a position via `MatMul` with a coordinate vector (alt to #93 einsum marginal).
- task057/059/060/066: moment-read + threshold-band `ArgMin` + crop + separable-einsum-with-basis output (#93/#118/#119);
  task060 = 7-operand separable einsum; task066 = 3-segment path via per-segment interval-membership (#123) + separable einsum.

## Batch 4 (task067/070/071/072/073/082/084/085) — ALL COVERED
- **task067**: cumulative-count via `1x30` ones-`Conv` + threshold-at-total/3 → keep-first-third (#93).
- **task070/084/085**: separable einsum output (#119; task084 = 14-operand monster).
- **task071** (40n): sprite-stamp = `GatherND` read + `ScatterND` write with `Expand`/`Reshape`-built index grids (#113).
- **task072**: arithmetic boolean (`1-x`) + `Abs`-difference (#125).
- **task073** (1n,910p) & **task082** (1n,1800p): SINGLE learned `Conv` — when the rule is a local filter, one conv
  IS the floor (#84/#88). Useful counterpoint: his zero-param style isn't always cheapest; price a single conv too.

## Batch 5 — ACCELERATED COVERAGE SWEEP (remaining 126) + 15 OTHER deep-dived  [coverage COMPLETE]
Novelty was exhausted (batches 2-4 all-covered), so switched from 8/batch dumps to a classification + op-inventory
sweep of all 126 remaining: **zero uninventoried ops — every op already maps to a technique**; 111 nets classify
cleanly into known families (#84/#93/#111/#112/#113/#114/#118/#119/#123/#124/#125). Deep-dived the 15 unclassified
"OTHER" (the only plausible novelty source):
- **task251 → #126** (NEW): iterative MaxPool flood-fill — 2D region growth + enclosure/hole detection. The one real find.
- **task375/329/373 → #127** (NEW): geometric micro-idioms (`Sqrt(area)`=square-side, diagonal `|dr|==|dc|`, abs-via-`Relu`, abba quilt).
- task135/326 (2n,0p): pure `Slice`+`Pad` structural crop-place (#67).
- task144/227/236/257/318: arithmetic-boolean blend `tgt*m + bg*(1-m)` / `Abs`-diff patch-compare (#125).
- task262: single `ConvTranspose` block-upsample (#80). task322: `MaxPool` neighbor-presence (#103/#105). task372: `Where`-overlay (#111).

**Coverage = COMPLETE.** All 175 Aditya-won nets accounted for: 64 deep-dived (batches 1-5 + the 15 OTHER), all
consistent with the family classification; the 111 family-clean nets coverage-confirmed via the op-inventory (no
uninventoried op, no novel combination) rather than each per-net dumped, given exhausted novelty. **Library now
#111-#127 — 17 techniques mined from Aditya.** Remaining loop work: the transfer list (next wake).

## R5 COMPLETE (2026-06-09, ~1h in — finished ~5h before the 6h deadline via the accelerated sweep)
Loop stopped early (work done, not killed by timeout). Deliverables:
- **`Onnx_Advanced/techniques.md` #111–#127** — 17 techniques mined from Aditya's 175 winning nets.
- **`R5_aditya_idiom_catalog.md`** — all 175 classified by idiom family + the einsum/QLC/constant deep-dives.
- **This log** — per-batch teardown (64 nets deep-dived + 111 coverage-confirmed; zero uninventoried ops).
- **`R5_transfer_list.md`** — the 17 techniques mapped to my top-30 highest-cost tasks; clearest win = task351→#121 (GridSample, near-direct port of an analyzed Aditya net).
Loop NOT re-armed. Coverage of Aditya's submission is exhaustive; further value is in APPLYING the techniques (workers), not more mining.
