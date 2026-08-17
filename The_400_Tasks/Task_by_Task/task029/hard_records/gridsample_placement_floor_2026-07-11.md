# task029 — GridSample placement floor / WALL (2026-07-11, batch62 Claude worker2)

## Bar in force
- **Live bar = 5,252** — `manifest.csv`, owner **Sho**, updated 2026-07-09T01:09:17Z,
  SHA `281d36b7545d8f271d6de5788db96a2401266b274fbffcefc6c4600400334d5a`
  (`Team_Best_Onnx/task029.onnx`). No `Staged_Boladi/task029.json` sidecar.
- `ng_current_bar.py --task 29` → `live_cost=5252 authoritative_source=manifest`.
- **This SUPERSEDES the stale local notes** (task029.md STATUS still said LIVE 14,545 and
  "honest floor 7,954 (v28) WALL"). Sho's 2026-07-09 ship **falsifies the 7,954 floor proof.**

## What Sho's 5,252 actually is (inspect_graph on the live ONNX)
A GridSample crop — a *different op family* from the whole v18–v28 Conv/Slice/canvas lineage:
1. **Scalar moments per color** (NOT `[1,10,30]` marginals): `cnt=Σinput` (`bkrc->bk`),
   `m2r=Σ_r rowcount²` (`bkrc,bkrv->bk`), `m2c=Σ_c colcount²` (`bkrc,bkvc->bk`) → 3× F32 `[1,10]`.
2. **Closed-form border test** (INT32): `valid = (cnt²·P == 4·(m2r−m2c)²) AND (cnt>7)`,
   `P = 4(m2r+m2c) − cnt² − 16cnt + 16`. Verified identity: for a hollow rectangle h×w this gives
   `P = 4(w−h)²` and `lhs = rhs = 16(h+w−2)²(w−h)²` **exactly**. This kills the old
   "forced 2,400 B `[1,10,30]` marginals" claim — the marker is found from 3 scalars/color.
3. **`e = Cast(valid)`**, then `rp=Einsum('bkrc,bk->br')`, `cp=…->bc` → marker row/col presence
   `[1,30]`; 4× `ArgMax` → `rmin,rmax,cmin,cmax`.
4. **GridSample** with a `[1,30,30,2]` F16 grid (separable ramp `iota[y]=(2y−27)/29` + scalar
   offset `rmin·2/29`, masked to `bigf=3` past the crop via `Where` → zero-pad) → F32
   `[1,10,30,30]` output = **graph output (free)**.

Cost model = **memory (Σ intermediate bytes, max(static,runtime)) + params (init element count)**;
graph input & output are NOT charged. Profile = static = **5,252 (mem 5,182 + params 70), gap 0%**,
visible 266/266.

## Forced-component floor (exact byte accounting)
| charged tensor | bytes | why forced |
|---|---:|---|
| `grid [1,30,30,2]` F16 | **3,600** | GridSample needs a full float grid (`[N,Hout,Wout,2]`, Hout=Wout=30 fixed by output contract); F16 is the smallest float. Only op that does masked data-dependent 2D translation into a static full-size output in ONE op. |
| 3 moments F32 `[1,10]` + 3 INT32 casts | 240 | Einsum output = input dtype (F32); INT32 required (see F32-algebra dead-end below). |
| 11-op INT32 border test + BOOL sel + `e` | ~500 | minimal DAG for `cnt²P==4d² ∧ cnt>7`; INT32 forced by precision. |
| `rp,cp [1,30]` F32 + 4 ArgMax | ~280 | single-Einsum presence is the cheapest position extraction. |
| grid-build (`g0/gy/g0x/gx`,`gyp/gxp`,`cond`,scalars) | ~640 | slot-trick keeps params tiny while only ONE `[1,30,30,2]` is materialized; `Where` mask required to zero-pad past the crop. |
| params (iota ramps + scalars) | 70 | 2×`[30]` ramps + ~10 scalars; minimal. |
| **TOTAL** | **≈5,252** | |

Independent from-scratch rebuild (`prototypes/task029_p62_gridsample_rebuild.onnx`, re-derived grid
math) → profile **5,252**, **AB byte-equal to Sho 2266/2266 (0 div)**. Ties the floor; a tie is
not stageable. Every tensor is load-bearing — nothing removable keeps the graph correct.

## Dead alternatives (why nothing beats 5,252)
- **Resize / `tf_crop_and_resize`** — roi is only `[8]` (~32 B) vs the 3,600 grid. **BUILT + run on
  ORT (ex0, marker rows[3,10] cols[4,13]): border color 2 + full exterior LEAK into the output.**
  `tf_crop_and_resize` extrapolation zeros only samples mapping *outside the image*; the interior
  boundary (`iy=rmax`) maps *inside* → sampled, not zeroed. Cannot mask the crop. **DEAD (empirical).**
- **Separable 2-op Gather / Gather-flat** — materializes a full `[1,10,30,30]` middle tensor
  (F32 36,000 / BOOL 9,000) or needs a `[1,10,900]` reshape (36,000). **DEAD.**
- **Single 3-operand dual-selection-matrix Einsum** `krc,yr,xc->kyx` — 2×`[30,30]` F32 = 7,200 +
  data-dependent matrix construction (`Equal`+mask) ≈ 16,000 total. Einsum needs uniform dtype, so
  F16 matrices force an F16 input cast (18,000). **DEAD (> 5,252).**
- **GatherND** — indices tensor must be INT64; `[30,30,2]` = 14,400. **DEAD.**
- **Collapse-to-color-index + GridSample(single plane) + Equal** — the collapse alone is ≥3,600
  (Einsum/ArgMax/ReduceSum output F32/INT at `[1,30,30]`), plus grid 3,600 + shifted 3,600 ≈ 12,000.
  This is the old v23 canvas family with a GridSample shift. **DEAD.**
- **Dynamic Slice / Pad (variable output)** — output is hard-fixed `[1,10,30,30]`; grader
  `verify_subset` uses `np.array_equal` (prior don't-do, 2026-06-22). **DEAD.**
- **F32 algebra (drop the 3 INT32 casts, save ~120)** — dense non-marker colors reach
  `lhs=cnt²·P ~ 10¹¹`; at that magnitude F32 spacing ≫ 1 → different integers round equal →
  false-positive markers (`e` picks >1 color → `rp` sums them → wrong bbox). **DEAD (magnitude proof).**
- **Drop the `cnt>7` guard (save ~20)** — a 2-pixel straight line satisfies `cnt²P==4d²`
  (`n=2` is the root of `n²+4n−12=0`) → false marker. Guard is load-bearing. **DEAD.**
- **`[1,9]` color slice (skip background ch0, save ~56)** — 3 extra Slice ops add ~108 B > 56 saved.
  **DEAD (net worse).**

## Second pass (2026-07-11, forced re-attempt) — library floor laws + built the collapse-canvas family
Mined `techniques.md` (crop-and-place §1652: fold-mask-into-Gather-index task384→9,617; negative-index
wrap; collapse-to-color-index §8b) and `cost_floor_reference.md` floor laws:
- **§165 "single-channel U8 color-index canvas = 900 B is the floor for a `[1,10,30,30]` output"** — BUT
  §18: a U8 plane needs a **≥plane-bytes F32/INT cast source**, so the collapse never nets below ~3,600
  (full) / ~2,304 (crop-collapse Conv `[1,1,24,24]`, §23 = the v28 lineage → 7,954, already ABOVE 5,252).
- **§219 "translate+zero-fill floor = length of shifted axis"** — realized by a SINGLE `Gather` whose
  output IS the graph output (memory 0). Applies to a **1-axis** shift; a 2-axis crop-and-place needs a
  second Gather whose intermediate is charged.

**BUILT + gated the genuinely-different collapse-canvas family** (`prototypes/task029_p62_collapse_gather.onnx`,
different dominant intermediate: U8 color-index canvas, not the F16 grid): collapse `Einsum('bkrc,bkxy->brc')`
→ `[1,30,30]` F32, Cast U8, two U8 `Gather` shifts (min-clamped indices), `Where`→sentinel-10 mask,
`Equal`→BOOL output. **visible 266/266; profile = 11,247** (memory 11,182 + params 65) = **2.14× the
5,252 bar**. Breakdown: `colf` F32 `[1,30,30]` = 3,600 (collapse, ties the grid ALONE) + SIX 900-B
U8/BOOL canvases (`colu`,`canvas`,`g1`,`placed`,`imask`,`masked`) = 5,400, because the cost model SUMS
intermediates and every Gather/Where step materializes a fresh canvas. Fold-mask-into-index (§1667) would
drop only the `Where` (−900 → still 10,347). **DEAD — 2× worse.**

Also attempted **collapse + Resize-with-sentinel canvas** (roi ~32 B): ORT `tf_crop_and_resize` places
unreliably (0/266 — same roi anomaly as the pass-1 Resize test), and even at its analytic best
(crop-collapse Conv 2,304 + U8 chain ≈ 4,356 placement) it exceeds GridSample's grid+build (4,240). **DEAD.**

**Why the collapse-canvas family is structurally ~2× GridSample:** GridSample places the 10-channel one-hot
DIRECTLY with output = graph output (free) and ZERO intermediate canvases — paying only the fused grid
(3,600). Every collapse-canvas route must (a) collapse (≥3,600 F32 source), THEN (b) place on the canvas
via ≥2 steps that each materialize a 900-B canvas, THEN (c) expand. Steps (a)+(b) alone exceed the grid.
The ONLY sub-3,600 placement helper is Resize's `[8]` roi, which cannot mask the interior boundary.

## Verdict
**WALL at 5,252.** 5,252 is the structural floor for the byte-fixed `[1,10,30,30]` zero-padded
one-hot output contract. The prompt's `floor_est ~1,200` is unreachable: it assumed "score vector +
selected indices," but the full-canvas zero-padded output *forces* an O(H·W) placement helper, whose
cheapest realization is the F16 `[1,30,30,2]` GridSample grid = 3,600 B (68% of cost). No ONNX op does
masked data-dependent 2D translation with an o(H·W) parameter.

**Only remaining sub-5,252 hypothesis** (unexplored, low probability): a hypothetical op or fused
pattern that produces the masked, translated full-size output without materializing either an
O(H·W) grid/index or an O(C·H·W) middle tensor. None exists in opset ≤21. If a future opset adds a
scalar-offset "roll-with-zero-fill + separable mask" op, revisit.
