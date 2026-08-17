# task079 — broke a "proven" exact floor on the CROP side (2846 → 2481), + 4 ORT levers + a plane-necessity method

Source: worker-079 session 2026-06-19 (t079). Builders `The_400_Tasks/Task_by_Task/task079/prototypes/build_task079_v40_narrowslice.py` and `build_task079_v45_int32trim.py`. Staged `staged_submits/task079_v45_int32trim_cost2481/` (sha256 2F73F197…4D). Beat the prior EXACT floor v32 @ 2846 → **2481** (−365, −12.8%), **byte-equal to v32 over the full 200266-input corpus** (AB div=0), truth 200000/200000, profile==static==2481 (266 arc-gen). v32 re-staged as the safety floor.

The headline lesson: **a floor proof that nails the SCORE side is not a floor proof for the whole graph — re-audit the CROP/output side separately.** task079's 2026-06-18 proof rigorously floored the score side (two F32 occupancy planes, 1120 B) and concluded "no EXACT sub-2846 rung exists." That conclusion was wrong by 365 B: the crop side (a GatherND over the full 30-col canvas + its index tensors) had untouched slack. Lesson: price the score side and the crop/output side as two independent floors; "the discriminator is minimal" says nothing about how you extract and emit the answer.

## Finding 1 (by_symptom / techniques — crop): replace a GatherND-over-full-canvas with a width-restricted dynamic Slice

v32 pulled the winner-color 3-row strip with `GatherND(input, idx[batch,color,row]) → row_strip4 [1,1,3,30]` (360 B) + the INT64/INT32 index tensors to build that index (`row_strip_idx4` 72 + `_i32` 36 + batch/color/row Unsqueeze-Concat tensors ~36). GatherND grabs the **entire** trailing axis (all 30 columns), and you pay for an index tensor.

When the active grid bounds the data (here 14×14, sprite cols ≤13), ONE dynamic Slice does it cheaper:
`strip14 = Slice(input, starts=[color_idx, r_sel, 0], ends=[color_idx+1, r_sel+3, 14], axes=[1,2,3]) → [1,1,3,14]` (168 B), with the index tensors collapsing to two tiny [3] start/end vectors. Net −288 B, **byte-equal** because the 3×3 window search only ever used cols 0..13 anyway (v32's AvgPool neg-pad cropped the 30-strip to 12 windows; the 14-strip with pads=0 gives the same 14−3+1=12 windows). The Slice's output shape is provably constant (ends−starts = start+const per axis) so bake concrete value_info + rely on the scorer's strict-infer rerun (traps §3 / task036 pattern), and profile_real_cost confirmed no GRADER-GAP inflation despite data-dependent channel/row starts.

General lever: **GatherND/Gather that reads a full-canvas slice is a crop-side smell. If the active grid bounds the gathered data, a dynamic Slice restricted to the active extent both narrows the plane and deletes the index-tensor machinery.** Gate byte-equal-vs-parent over the full corpus.

## Sorter note: by_symptom.md — add to the crop/extract section: "GatherND-over-full-canvas → dynamic Slice restricted to active grid width (deletes index tensors + narrows plane); byte-equal when active region bounds the data (task079 v40, −288B)."

## Finding 2 (ort_reality / error→fix): Slice accepts ALL-INT32 starts/ends/axes/steps, but Tind must MATCH across the four

This **corrects a stale task079 note**. The old floor proof said "INT32 GatherND index → ORT-rejected, needs INT64." True for **GatherND** (its index is INT64-only), but NOT for **Slice**: `Slice` runs fine with INT32 `starts/ends/axes/steps` — provided all four share the same Tind. Mixing INT32 `starts` with an INT64 `axes` init throws `Tind bound to different types`. Converting the whole Slice index machinery to INT32 (and noting `color_idx` is already INT32, so the prior INT32→INT64 Cast vanishes) saved 44 B on task079. ArgMax outputs stay INT64 (forced) but get Cast to INT32 immediately.

## Sorter note: ort_reality.md (errors / dtype matrix) — add: "Slice: INT32 starts/ends/axes/steps all OK, but Tind must MATCH across the four (mixing INT32 starts + INT64 axes → 'Tind bound to different types'). Unlike GatherND (index INT64-only), Slice does NOT force INT64 — INT32 halves index bytes."

## Finding 3 (ort_reality / dtype matrix): Gather ACCEPTS BOOL data input

`Gather(BOOL_tensor, indices)` is accepted (verified on pinned ORT), even though `ReduceSum(BOOL)`, `ArgMax(BOOL)`, `Sum(BOOL)` are all rejected. Lever: when you must select-then-reduce a boolean occupancy plane, **Gather the 1-byte BOOL plane first and cast only the small gathered slice to F16**, instead of casting the whole [1,10,14,1] plane to F16 before the Gather. task079 v45: Gather `row_has_b` BOOL [1,10,14,1] → [1,1,14,1] BOOL (14 B), then Cast→F16 (28 B) just for the ArgMax. −15 B.

## Sorter note: ort_reality.md (dtype × op matrix) — add: "Gather accepts BOOL data input (Gather(BOOL,idx) OK); pair with the fact that ReduceSum/ArgMax/Sum reject BOOL — gather the BOOL plane, cast only the gathered slice."

## Finding 4 (ort_reality / tiny_architectures): Conv accepts NEGATIVE pads in ORT (same as AvgPool/MaxPool)

Verified: `Conv(weights, pads=[0,0,-16,-16])` runs in ORT and crops 30→14 in the conv op itself (built `[1,1,14,14]` from `[1,10,30,30]` via a 1×1 weight `[0..9]` color-index kernel; content correct, out-of-region color cropped to 0). So a 1×1 Conv can collapse channels AND crop in one node. (For task079 this still lost — see dead-end below — but the neg-pad-Conv primitive is broadly useful for "collapse + restrict-to-active-grid" in one op.)

## Sorter note: ort_reality.md / tiny_architectures.md — add: "Conv accepts negative pads in ORT (like AvgPool/MaxPool); a 1×1 Conv can channel-collapse AND neg-pad-crop the canvas in one node."

## Finding 5 (method, for traps §FLOOR-PROOF / Algorithm_Library): the corpus collision-test that PROVES a plane is load-bearing

To decide whether a cheaper single-plane (or single-feature) discriminator can be EXACT, don't argue — tabulate. For each corpus example, take the ground-truth winner (from the output) and test whether its **signature** equals any non-winner's at increasing feature richness:
`S1=(pix)`, `S2=(pix,row_count,row_max)`, `S3=(pix, full sorted row profile)`, `S3col=(+full col profile)`. A collision (winner-sig == a non-winner-sig) means NO function of those features can separate them → that feature set cannot be exact. task079 (200000 examples): S1=2576, S2=22, **S3=19**, **S3col=0**. So even the *full* row profile leaves 19/200000 row-indistinguishable winners (concrete: 3 copies of a 4-cell sprite vs 2 copies of a 6-cell sprite — identical sorted row profile {1,1,2,2,3,3}, pix=12 — separated only by col_count 9 vs 6). The column plane is **load-bearing**; no row-only exact discriminator exists. This is faster and more decisive than building-and-gating candidate discriminators.

## Sorter note: Algorithm_Library/rule_discovery_methodology.md — add as a method: "plane-necessity via corpus signature-collision test (winner-vs-nonwinner sig equality at increasing feature richness; 0 collisions = separable, >0 = that feature set cannot be exact). Decides whether a cheaper discriminator plane is droppable. (task079: full row profile → 19/200000 collisions → col plane load-bearing.)"

## Dead-ends MEASURED (the two "radical" reformulations both cost MORE — the no-liveness sum() model punishes per-color 2D planes)

- **Count copies via per-color 3×3 Conv** (instead of pixel-count + occupancy correction): *algorithmically EXACT* (5000/5000 truth, byte-equal output) but **cost 14722** — the depthwise `[1,10,12,12]` F32 conv plane = 5760 B ALONE exceeds the whole 2436-B memory, and it needs the conv plane + Equal-to-max BOOL + present-mask BOOL + F16 cast (~11520 B). (Gotcha if you try it: `Equal(conv, ReduceMax)` lights up all 144 windows for ABSENT colors whose max=0 → must AND with a present-mask `Greater(conv,0.5)`.)
- **Collapse the channel axis first** (ArgMax/1×1-Conv → color-index map, then count): cheapest collapse intermediate is 784 B (neg-pad Conv → [1,1,14,14]) but per-color recovery must re-expand (`Equal→[1,10,14,14]` BOOL 1960 + F16 cast 3920, since ReduceSum(BOOL) is rejected) → ≥**7616 B** total.

Root cause (transferable): **per-color discrimination that needs 2D spatial resolution (a conv window plane, or a re-expanded color axis) is maximally punished by the `sum()`-of-all-intermediates cost model. The cheap path collapses the SPATIAL axis into 1D projections FIRST (row/col occupancy, ~14 elems/color) while keeping colors separated — never materializing a per-color 2D or 10-channel-spatial plane.** When a "count features per color" task tempts you toward a per-color conv or a OneHot re-expansion, price the 2D/10-ch plane first; the 1D-projection-first ordering is almost always the floor.

## Sorter note: traps.md or scoring_internals.md — add to the no-liveness cost-model section: "per-color 2D window planes (copy-count Conv [1,C,h,w]) and re-expanded color axes (OneHot/Equal → [1,10,H,W]) are punished maximally by sum()-no-liveness; collapse the SPATIAL axis to 1D projections FIRST and keep colors separated. (task079: occupancy-projection 1120B exact-floor beats conv-copy-count 14722 and collapse-first 7616, all measured.)"
