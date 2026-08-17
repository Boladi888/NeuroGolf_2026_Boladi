# DEAD ROUTE — task002 "MorphologyPool" spatial-dilation attack (2026-07-12, bo_ng_cld_worker1)

**Live bar:** cost **9413**, sha256 `8c8340bca13986bd90f9831574d54e3e930935eacfec70fa9f830e455179029a`
(manifest authoritative via `ng_current_bar.py --task 2`; owner Boladi; Kaggle probe OK 2026-06-26
+0.014 LB). Team_Best_Onnx recomputes to 9397 (same sha, flagged stale) — I built strictly below 9413.

**Prompt hypothesis (falsified):** STRUCTURAL card, floor_est ~1500 (gap 6.3×). Direction:
"binary/colored morphology first — MaxPool dilation, signed-int8 erosion by Neg-MaxPool-Neg, local
3×3 window tests, compact output assembly; treat connected-component language as a hypothesis to
falsify with local reductions." Card triggered by corpus keywords: dilation, neighbor, connected,
component. **`lib_query search "MorphologyPool"` returns ZERO hits** — the card name is a corpus
mis-tag, not a real technique. Numpy ref + live graph both compute a **4-connected flood fill**.

## Why the premise is backwards (the key finding)

**LIVE already IS a morphological-dilation flood — implemented in bit-packed space.** Its flood loop
(`build_task002_rayseed_bitpack.py`, iters=5) does exactly a 1-cell dilation per iteration:
`BitShift ±1` = horizontal dilation, `Gather prev/next row` = vertical dilation, `& open_bits` =
geodesic mask. That is precisely a MaxPool-3×3 dilation + AND-open, but on a **uint32 row-packed**
tensor at **80 B/mask** (400 cells / 20 uint32 = 5 cells/byte) instead of a spatial tensor. The
MorphologyPool card points at a formulation that LIVE **already dominates** in the cost-optimal
representation.

## Priced falsification (probe, no ONNX shipped — kill rule fired at the cost floor)

Probes: `scratchpad/morphpool_probe.py` (ORT dtype + serpentine iters), priced comparison inline.

- **F-M1 — spatial MaxPool dilation (the literal Direction):** ORT MaxPool dtype support probed:
  **BOOL → FAIL, INT32 → FAIL**; only FLOAT/FLOAT16/UINT8/INT8 have kernels. Cheapest legal per-cell
  state = **UINT8 → 400 B/mask** on the 20×20 crop, **900 B/mask** on the 30×30 I/O frame. That is
  **5×–11× the packed uint32 80 B/mask** for the identical operation. Even the most LIVE-favorable
  spatial mirror (spatial log-shift ray-seed ≈ 20 masks + 5 MaxPool iters + repair, all 20×20 uint8)
  prices to **≈13 kB** ≫ 9413; on the mandatory 30×30 frame it is far worse. Spatial morphology
  **cannot reach 9413** — the 5× per-mask density penalty is structural. DEAD.
- **F-M2 — Neg-MaxPool-Neg erosion / local 3×3 window classifier:** same dtype + density penalty as
  F-M1 (erosion is dilation of the complement); a local-window test cannot decide global enclosure
  (already falsified as F1 in `colorindexcollapse_falsefloor.md`: two grids with identical local
  neighborhoods, opposite enclosure). DEAD on both correctness and cost.
- **Iteration count:** MaxPool has no ray acceleration, so iters = BFS depth. Prior F2 measured a
  width-1 serpentine at depth **170** on 20×20; a fixed small step count false-encloses (§14e /
  visible-fail). LIVE only survives at 5 iters because of straight-ray seeding + memorized repair.

## Denser-packing sub-probe (genuinely new this session) — also DEAD

Only way below 80 B/20-row-mask is packing multiple rows per lane (uint64: 3×20 bit < 64 → 56 B/mask).
But then horizontal `BitShift` bleeds across row-fields (needs a boundary-mask `And` after every
shift) and vertical shift crosses lane boundaries (`BitShift 20` within-lane + `Gather` cross-lane).
The extra ≈8 ops/iter outweigh the 24 B/mask saving → **net negative**. uint32 @ 1-row/element is the
ORT density optimum: 20 cols need ≥20 bits ⇒ 4 B; **uint16 has no ORT kernel**; uint8 < 20 bits.
80 B/20-row-mask is the ORT-legal floor. (Also non-byte-equal → would fail `ab` anyway.)

## NEW floor strengthening — static minimality audit (supersedes "each tensor load-bearing")

`scratchpad/audit_9413.py` on the staged 9413 graph proves it is **structurally minimal**:
**0 duplicate-valued initializers, 0 unused initializers, 0 dead node outputs, 0 common
subexpressions (no two nodes with identical op+inputs+attrs), 0 idempotent/identity nodes.**
114 nodes, 36 initializers, mem 8962 + params 451 = 9413 — every tensor distinct and load-bearing.
The 9545→9413 dedupe already harvested all structural folds; the output assembly is proven near-minimal
(uint8 3-lane unpack 1020 B beats a direct uint32 unpack 2000 B; the only `[1,10,30,30]` tensor is the
free `Where` graph output). **No residual byte-equal fold exists.**

## Does NOT apply: the task255 "MaxPool prefix/suffix-OR + QLinearConv erode" 1-D morphology win

Lesson `DONE_2026-07-12T063259Z_maxpool_prefixsuffix_or_and_qlinearconv_erode…` took task255 10331→6179
(beat LIVE) with wide-MaxPool prefix/suffix-OR and QLinearConv run-threshold erosion. **Those primitives
are 1-D and require a CONTIGUOUS band / separable projection** (its own caveat: "a multi-run mask gives
per-run fills, not a global prefix"), and each cheap tensor is a *single* row/column projection `[.,.,.,30]`
= 30 B. task002 is **2-D non-separable connectivity** — the flood propagates through serpentine turns, not
along one contiguous band per axis — so it needs FULL-grid masks. In full-grid form, packed uint32
`[1,1,20,1]` (80 B for all 20 rows) beats spatial uint8 `[1,1,20,20]` (400 B) by 5×, and computing any ray
spatially would force unpack→MaxPool→repack overhead (~400–900 B/boundary) that dwarfs the few-mask saving
— and must still be byte-equal to LIVE's exact packed output. The task255 primitive is for separable
projection-fill rules; it does not transfer to task002's global flood. Do not re-open on its basis.

## Second pass (2026-07-12, forced) — three genuinely-different formulations, all BUILT + measured DEAD

Re-mined `INDEX→techniques.md` (#37 corpus iteration floors, #47 uint32 flood K≥12, #88 u8-lane flood,
#110 ray-fill LUT, #126 iterative MaxPool flood) + BoBrain provider. Measured on the real 200k corpus
cache (`Q:\…\argen_cache\task002.npz`, `scratchpad/corpus_flood_analysis.py`):

1. **8-conn MaxPool flood (#126, spatial): ILLEGAL.** 4-conn vs 8-conn enclosed-set mismatch on
   **4000/4000** sampled corpus pairs — task002 has diagonal wall gaps everywhere, so 8-conn under-encloses.
   The cheap 3×3 MaxPool shortcut (would be 1 op/round) cannot be used; must stay literal 4-conn.
2. **Exact low-K flood: DEAD.** Corpus 4-conn flood depth **>12 for 3988/4000 pairs, max 35** — a pure
   1-cell flood needs K≈35 rounds (≈25 kB) to be exact. This is why LIVE needs ray-seed (collapses straight
   runs) + repair, and is only 91.7% truth. No cheap exact flood exists; exact costs ≫9413.
3. **u8 3-lane packed flood (#88, different dtype/shape `[1,1,20,3]`u8): BUILT + priced-DEAD.**
   `scratchpad/rep_compare.py` builds a K-round 4-conn flood in each rep and measures the per-round slope:
   **uint32-row = 720 B/round, u8-3lane = 1020 B/round (+42%).** Root cause: 20 cols fit in ONE uint32 ⇒
   horizontal shift is carry-FREE (1 op); u8-lanes must pay cross-byte carry every horizontal step
   (BitShift carry-bit + Gather lane-roll + Mul boundary-mask ≈ +10 masks/round), which dwarfs the
   20 B/mask density gain. #88's "u8 beats u32 for width 9-24" holds only for VERTICAL-heavy floods
   (task243 20V+16H, which still cost 11,315 > 9413); task002's balanced H+V ray-flood inverts it.
   uint32-row is the density optimum here (confirms #47: adopt uint32 for 4-conn width≥12).

Params also near-minimal: `pack_w`'s 180 zero params are cheaper kept than sliced (slicing ch3 first =
900 B `[1,1,30,30]` intermediate > 180 params); Gather row-shifts (20 params, 80 B out) beat Pad+Slice
(0 params, 160 B out). Bar re-confirmed: `profile_real_cost --task 2` → static=9413 profile=9413.

## Ship-gate reality

`ab --n 2000` streams officials **then corpus** (lesson `neurogolf.lesson.20260630.task002.ab_includes_corpus`).
Any non-byte-equal reformulation (spatial, uint64-repack, different iters/seeding) diverges off-officials
and **fails ab**. A *more-correct* flood (deeper, §DUAL-TRUTH shippable) costs MORE, not less. So the only
cost-reducing ship path is a byte-equal micro-fold — and the audit above proves none remains.

**Exact failed gate:** none run past the probe (kill rule: premise + cost floor falsified before build;
truth/ab/battery/full-profile on a non-existent, provably-losing candidate = pure burn per gate hygiene).

**Verdict: WALL @ 9413.** Effective best stays 9413 (already staged). No morphology/pooling route exists;
no denser-packing route exists; no residual byte-equal fold exists.

**Forced-component argument:** enclosure is global 4-connectivity — not per-cell (F1/F-M2), not
fixed-shallow (F2), not cheaper under any legal spatial-morphology op (F-M1: 5× density penalty), not
cheaper under denser packing (net-negative boundary ops), and the ORT-legal cost-minimal representation
is exactly the bit-packed uint32 log-shift ray flood the LIVE graph already uses, audited to structural
minimality at 9413.

## Value-level fold hunt (also run this session) — DONE, nothing real

`scratchpad/value_cse_hunt.py` ran the 9413 graph over 30 varied inputs (random/sparse/dense/box/
serpentine) capturing all 113 intermediates and testing for pairs that are byte-identical on **every**
input (value-level CSE that structural CSE misses). Result after filtering degenerate/constant pairs:
the ONLY ≥80B "always-equal" pair is `reach_5 == reach_repaired`, and that is a **repair-coverage
artifact** — the 6 repair masks (`repair_selected0-3`, `repair_bits`, `repair_open`) are constant-**zero**
on all non-triggering inputs (repair fires only on 7 exact ARC-GEN wall-row signatures), so
`reach_repaired = reach_5 | repair_open = reach_5 | 0`.

That surfaced the last candidate micro-fold: **is `repair_open = repair_bits & open_bits` a removable
remask (−80B → 9333)?** **TESTED and KILLED** (`scratchpad/test_repair_open_fold.py`): the repair conds
check a single row (e.g. row 1) but seed cells sit in *other* rows (7,9), so a corpus grid can match the
trigger while a seed cell is a WALL. Constructed exactly that (row1==84218, cell (7,15)=wall): LIVE keeps
(7,15) color-3; the clip-removed variant flips it to color-4 — **outputs differ**. The `& open_bits`
clip is **load-bearing** (prevents marking wall cells enclosed) ⇒ dropping it is non-byte-equal ⇒ fails
ab. No gate burned (disproven by counterexample in-process).

**No unexplored micro-fold remains** (structural CSE clean, value-level CSE clean, the one candidate
remask disproven). Do NOT re-open spatial morphology, MaxPool/erosion, local-window classifiers, denser
packing, or the repair_open remask.
