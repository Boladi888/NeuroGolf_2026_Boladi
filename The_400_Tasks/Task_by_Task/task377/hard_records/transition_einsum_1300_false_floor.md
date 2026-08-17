# DEAD ROUTE — task377 "TransitionEinsumCount 4248→1300" is a FALSE FLOOR (DROP)

- **Recorded:** 2026-07-11 by `bo_ng_cld_worker1` (Claude lane).
- **Live bar at analysis:** **4246** (manifest, authoritative via `ng_current_bar.py`, owner Boladi,
  refreshed 2026-07-11T00:40Z). Live SHA `d15d6df5a57b0312660df27c429623d2b598935987de560b637505ee67247c6c`.
  (`profile_real_cost --arcgen 266` on the live file = static=profile=4246, memory 3172 + params 1074.)
- **Prompt target:** 1300 (claimed 3.27× gap, "forced bytes are the source marginal/transition scalars
  and output predicate; F32 Conv-diff carriers are not forced").

## Why the 1300 floor is FALSE (charged component already gone)

The Direction says: "Use ordered transition-count Einsum / source-column reads so the transition rule is
computed directly instead of carried through Conv-diff planes." **This is already the LIVE graph** (`task377_rw7`):
node 0 is `Einsum('bchw,hg,bdgw,cd->bw')` = per-column vertical transition count with `v8_adj_next` F32[30,30]=900
+ `v8_neq_cd` F32[10,10]=100; there is **no Conv anywhere** in the live graph. The charged component the target
attacks (Conv-diff F32 planes) was deleted back at v8 (2026-06-21). Per the 2026-07-10 workflow rule
("if it is already gone … return DROP with the refreshed floor note"), this is a DROP.

## The 1300 estimate rests on a premise that is empirically false

The true rule (onion→bullseye): input = nested rectangular color layers (NOT concentric — inner rects are
offset inside outer ones); output = those layer colors redrawn as concentric square rings, (2L−1)². The ONLY
information in the output is the ordered layer-color sequence c₀..c_{L−1} (L≤5) — everything else is a
deterministic render. The 1300 floor assumes that sequence can be extracted from cheap per-source scalars.
It cannot. Tested every cheap extractor on the full 266-example corpus (train+test+arc-gen):

| # | Independent from-scratch formulation | visible/266 | Kill reason |
|---|---|---|---|
| C1 | sort colors by pixel COUNT (desc) → layer order | **209/266** | colors REPEAT at depth (test[0]: outer=2 AND center=2); a per-color statistic can't emit one color at two depths |
| C2 | sort colors by BOUNDING-BOX AREA (desc) → layer order | **240/266** | same repeat problem + overlapping bboxes |
| C3 | CENTROID center + Chebyshev-ring read | non-concentric on **232/266** | layer centers are offset; centroid of the figure ≠ innermost center on 87% of the corpus |
| C4 | middle COLUMN/ROW of figure bbox, RLE the scan line | col 232 / row 230 / **best-of-both 256**/266 | bbox-center line does not reliably cross the offset innermost region |

A single visible miss = certain hidden 0 (traps §VISIBLE-FAIL). **All four die on visible.** Only "the column
with the MAXIMUM number of vertical transitions" is guaranteed to cross the innermost region (it is contained
in every outer rect, so its column crosses all boundaries) — which is exactly what the live transition-count
Einsum + ArgMax computes. The spatial max-transition scan is not a formulation artifact; it is forced by
color-repeat + non-concentricity.

## Forced-floor arithmetic (cost model: params = init element count; memory = intermediate bytes)

- **Output one-hot** `small_oh` BOOL[1,10,9,9] = **810 B** — forced. Output format is BOOL[1,10,30,30] one-hot;
  maxL=5 across the corpus ⇒ 9×9 bullseye pre-pad. The [1,1,30,30] U8 route (900) is worse, so 810 is the floor.
- **Transition selector** `v8_adj_next`[30,30] = **900 params** — forced. It is the row-adjacency operator
  *inside* the Einsum; the Einsum contracts the one-hot input with itself shifted by adj → [1,30] directly,
  never materialising a color plane. Any Slice/Conv shift first forces a per-cell color-index/activity plane
  for all columns: ArgMax→I64[1,1,30,30]=7,200, or ramp-Conv→F32[1,1,30,30]=3,600 — both dwarf 900
  (v13 Lever-1 proof, re-confirmed). The element-count model means dtype tricks don't shrink [30,30]=900;
  only a smaller shape would, and 30 rows is intrinsic.
- **Color-diff** `v8_neq_cd`[10,10] = **100 params**, **ring Gather index** I32[9,9] = **324 B** (Gather
  indices are INT32/INT64-only; 324 is the min), plus the [1,30] Einsum read vectors and the CumSum ring chain.

Minimum for ANY correct formulation ≥ 810 + 900 + 100 = **1810 B**, and the realised formulation floor is
**4156** (v15 below) / 4246 (live). **1300 is unreachable — DROP.**

## EMPIRICAL fast-probe (2026-07-11 pass 2) — the transition skeleton alone > 1300

Per the prompt's Fast-probe kill rule ("price the transition-count skeleton … if it exceeds 1300, stop"),
I truncated the live graph to JUST the forced front-end {transition Einsum → ArgMax cstar → OneHot →
the two column-read Einsums} and profiled it (`prototypes/probe_frontend_only.py` →
`task377_frontend_only.onnx`):

```
task377_frontend_only.onnx: static=1381 profile=1381 (memory 368 + params 1013; 266 real inputs)
```

**1381 > 1300 — the skeleton exceeds the target before a single render byte** (no ring-shell CumSum, no
5×ReduceMax color extraction, no ring index [9,9]=324, no output one-hot [1,10,9,9]=810, no inbox mask).
params 1013 is dominated by the forced `adj[30,30]=900`. Adding the forced render (≥ 810 + 324) gives a
true floor ≥ **2515**, realised at 4156. The 1300 estimate assumed near-free scalar extraction; the
center-finder + color-read alone provably costs 1381. **KILL per fast-probe rule → WALL.**

## Also rejected earlier (do not re-walk — v13 floor proof, 2026-06-22)

- Lever 1 (delete [30,30] adjacency via Slice-shift): LOSES (forces 3,600–7,200 B color planes).
- Lever 2 (free-output Chebyshev distance×color Einsum): LOSES (#109 carrier tax; Einsum is F32/I32-locked
  on ORT CPU, band operands charge F32 and exceed the BOOL one-hot they'd delete; measured 5,387).
- v14 eq-count+ArgMin selector: visible 22/266 (inactive all-zero cells break `29 − equal_count`).

## One remaining hypothesis (unproven, low confidence)

Could the render tail (`v_ring_i` 324 + inbox/Gather chain) be replaced by a single separable free-output
Einsum that writes `output` directly, dodging BOTH the Gather-index and the pre-pad one-hot? v13 Lever-2 says
no for the color render (F32-locked Einsum), but a *BOOL* separable band render was not exhaustively priced
against the exact 810+324 it would need to beat. Expected to lose (same carrier tax). Not the 1300 path.

## Side win found en route (NOT the structural target)

`v15` I32-distance fold on the live graph: the ring index built the F16[9,9] `v_ring_f`=162 only to Cast it
to the INT32 Gather index. All distances are exact integers (0..4), so casting the two 1-D clipped operands to
I32 and taking the Min in I32 is byte-identical and deletes the 162 B F16 plane (adds 2×36 B I32 casts, net −90).
`profile_real_cost` 4156 < 4246, visible 266/266, AB byte-equal, ng_safety PASS. Staged as a `rung−epsilon`
micro-shave — see `staged_candidates/task377_v15_i32ringidx_cost4156/`. This is a shave, not the false-floor
structural win the prompt asked for.
