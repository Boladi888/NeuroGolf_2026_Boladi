# cost_floor_reference.md — price-shop your algorithm before building

You're about to build an ONNX. Before writing code, estimate the cost floor of your op chain. This file is the price list.

## The formula

```
cost = memory + params
points = max(1.0, 25 - log_e(max(1.0, cost)))
```

Lower cost = higher points. **Cost 0 (e.g., `input → Identity → output`) and cost 1 both yield the 25-point ceiling** via the inner `max(1.0, cost)` guard. At cost=1000, points=18.1. At cost=10,000, points=15.8. At cost=100,000, points=13.5. At cost=1,000,000, points=11.2.

Implication: the gain from 1,000,000 → 100,000 (-900k bytes) ≈ +2.3 pts. From 10,000 → 1,000 (-9k) ≈ +2.3 pts. From 100 → 10 ≈ +2.3 pts. **Log scale means small absolute reductions at low cost are worth as much as huge reductions at high cost.**

## Full-canvas-plane floor taxonomy (2026-06-01 — the irreducible cost class)

For per-cell-rule tasks whose grids reach 30×30, the floor is set by how many **full 30×30 planes** the rule structurally needs — and ORT forces those planes to a 4-byte dtype (a U8 plane needs a ≥plane-bytes F32/INT cast *source*, so dtype-narrowing never nets out). Before targeting sub-2k/sub-3k, count the unavoidable planes AND check the grid max (`corpus.build(N)['in_shapes']`): if grids are < 30, crop to the real max (task070 grids are hardcoded 17×17; task400 are exactly 24×24).

| Task | Irreducible plane(s) | Why it can't shrink | Floor |
|---|---|---|---:|
| task203 | `mid_oh` `[1,10,1,30]` F32 = 1,200 | 10-channel F32 read of the middle row (the recolor permutation is *derived from* it); channel-collapse-first needs the 3,600 full canvas; the old sub-1,200 dynamic-Slice attempt scored `None` because shape value_info was unknown. After task036, re-test fixed-shape dynamic Slice with baked value_info before calling this part final (traps §8b/#146). | 2,188 provisional on unannotated Slice |
| task400 | `canvas` `[1,1,24,24]` F32 = 2,304 | dilated crop-collapse Conv output; ORT forces Conv→F32; grids are *exactly* 24×24; extracting the 5×5 from the 10-ch one-hot instead costs 4,800 | ~3,300 |
| task138 | old canvas floor RETRACTED | frame-raycast crop can be rebuilt as a free-output rank-1-basis Einsum with no crop planes; F32 exp-ramp positional codes replace ArgMax planes. Keep the old canvas note only as a rejected family-specific proof. | 5,403 team-best; 5,283 safe follow-up pending full gate |
| task224 | `border_full_b` `[30,30]` BOOL = 900 | full-canvas condition for the output-overlay `Where`; output is 30×30 byte-equal to LIVE | ~1,700 |
| task137 | two `[1,1,30,30]` planes (axis-select 900 + 3-state canvas 900) | 2D Chebyshev ring needs *both* axes (1 plane); 3-state output ring/bg/padding needs the sentinel-UINT8 canvas (1 plane); grids reach 30 | ~2,500 |
| task383 | Dilated F32 crop/color Conv over 24×24 + F16 B-count + U8 canvas/bridge | bordered-rectangle rule needs per-cell color AND global row/col B-counts. **Updated 2026-06-11: #98 30→24 dilated crop-collapse moved 13,112→10,942** (measured residual floor ~10,884). Earlier canvas+`Equal` BOOL output still matters (make the Equal the graph output, don't Cast it), and MatMul-clearing the F16 B-mask still holds; the old full 30×30 color-decode Conv is no longer the floor. | ~10,900 |
| task251 | fixed-unroll bit-packed flood | R10 is pinned by the full cached 200k convergence tail; 20k/30k samples under-counted (8/9). Each removed round saves ~216B but fails truth (R9 = 199998/200000). | 3,740 R10 |
| task286 | bit-parallel column-packed flood | old grid-tensor geodesic floor retracted: pack each column into UINT32 scalars, hoist segmented run masks, and sweep horizontally on O(W) scalars. Full-log vertical fill loses; horizontal reversals bound the pass count. | 21,558 current |

**Real plane-floor vs beatable "floor":** task203's "2,404 floor" fell to 2,188 (a center-Gather fold + opset-18 `ScatterElements` removed plumbing *around* the plane); task215's "5,730 floor" fell to 110 (a from-scratch rewrite removed the plane *entirely*). The plane-bytes themselves are the wall; everything else is negotiable. Verification checklist in current_strategy.md §captain-session-2026-06-01 #2.

## Calibration: local prediction vs Kaggle reality (regime table — supersedes the dated datapoint ledger)

```
predicted_lb_gain ≈ Σ (log_e(orig_cost_i / new_cost_i))   # across all changed tasks
```

| Regime | Multiplier | Empirical basis |
|---|---|---|
| Single-task wildcard | **1.0× (raw log_e)** | ~21 single-task ships 2026-05-22…23 matched raw prediction within ±0.05; ~10 more 2026-06-01/02 exact |
| Same-task follow-on after a hidden-correct ship | **1.0× (no discount)** | task165, task138 v9 — follow-on gains stayed raw-log additive |
| Small stack (2–7 tasks) | **~1.0× additive** | 3-stack +6.16 exact, 5-stack +9.68, 7-stack exact; 21-stack +20.90 |
| Mid stack (~30 tasks) | **~0.84×** | megastack33: predicted +0.74, actual +0.62 |
| Large megastack (100+ tasks) | **~0.48×** | megastack153 (2026-05-20): predicted +54.20, actual +26.13 (verify cross-contamination, ORT memory pressure) |
| Cascade unlock | step-function exception, RARE | ONE confirmed: task110 v8 +12.73 (first crossing can be 10–70× predicted). Same-task follow-on AFTER an unlock reverts to the log-curve **with undershoot** (v9: +0.21 predicted, +0.07 actual). Do NOT label ordinary additive gains as cascade |
| Zero-public-weight task (10 confirmed: 366/324/133/182/018/157/255/066/076/277) | **0.00 actual regardless of cost** | traps.md §14e + §27 (cherry13b Identity stub is the disambiguator) |
| Hidden-cost-decoupled | partial or zero | task284 full decoupling; task234 ~22% partial — cost-gain survives only where structural assumptions hold on hidden |

**Practical rule**: single-task + small-stack work predicts at `Σ log_e(orig/new) × 1.0`; apply 0.48× only when bundling 100+ tasks; never apply a positive multiplier to zero-weight tasks. Workflow: single-submit each candidate to measure, then stack confirmed-positives (additivity holds). Full dated datapoint ledger (every exact-match ship 2026-05-20…06-02): `_archive/cost_floor_evicted_2026-06-10.md`.

## Tensor byte sizes (the units of memory)

| Shape | F32 (4 B/elem) | F16 (2 B/elem) | INT32 (4 B/elem) | UINT8/BOOL (1 B/elem) | INT64 (8 B/elem) |
|---|---:|---:|---:|---:|---:|
| `[1, 10, 30, 30]` | 36,000 | 18,000 | 36,000 | 9,000 | 72,000 |
| `[1, 1, 30, 30]` | 3,600 | 1,800 | 3,600 | 900 | 7,200 |
| `[1, 10, 15, 15]` | 9,000 | 4,500 | 9,000 | 2,250 | 18,000 |
| `[1, 1, 15, 15]` | 900 | 450 | 900 | 225 | 1,800 |
| `[1, 1, 9, 9]` | 324 | 162 | 324 | 81 | 648 |
| `[10]` | 40 | 20 | 40 | 10 | 80 |
| `[1]` (scalar value_info) | 4 | 2 | 4 | 1 | 8 |
| `[]` (scalar value_info) | 4 | 2 | 4 | 1 | 8 |

**Scalars cost `dtype.itemsize` bytes**, not always 4. `calculate_memory` does `num_elements × dtype.itemsize` where shape `[]` yields `num_elements = 1`. `BOOL []` = 1 B, `F16 []` = 2 B, `F32/INT32 []` = 4 B, `INT64 []` = 8 B. The old "4 bytes minimum" claim only applies to 4-byte dtypes. Use `BOOL`/`UINT8` scalars where possible for the cheapest sentinel/threshold value_info entries.

## Cost contribution per op

These are LOWER BOUNDS assuming the smallest valid intermediate dtype and minimal `value_info` count.

| Op chain | Contributes to `cost` | Notes |
|---|---|---|
| Pure shape-only (`Slice/Gather/Pad/Reshape/Concat/Transpose/Identity/Cast`) on small intermediates | ~1,500 floor | Cost from intermediate value_info bytes + ~30 params for indices |
| One Conv (3×3, no padding, no F16) | +36,000 if intermediate is `[1, 10, 30, 30]` F32; less if narrowed | Conv itself is not a memory hit; the activations are |
| One Conv (1×1, F16 intermediates) | +18,000 for F16 [1,10,30,30] or proportionally less | F16 halves memory |
| ReduceMax / ReduceSum on `[1, 10, 30, 30]` axes=[2,3] keepdims=1 | +40 (`[1, 10, 1, 1]` F32) | Tiny |
| ReduceMax / ReduceSum on `[1, 10, 30, 30]` axes=[2] keepdims=1 | +1,200 (`[1, 10, 1, 30]` F32) | Per-column reduction |
| ReduceSum of bool → INT64 | output dtype is INT64, count bytes accordingly | Cast to BOOL if needed downstream |
| OneHot with `[1, 1, 30, 30]` indices, depth=10 | output is `[1, 10, 30, 30]` F32/F16/INT64; +36,000 F32 or +18,000 F16. **BOOL/UINT8 values are NOT supported by ORT's OneHot kernel** (verified opset 11/13). Use `Equal(label, channel_constants) → BOOL` if you need a BOOL multi-channel mask. | OneHot is 0-MAC but the output tensor counts |
| Where(c, x, y) | output shape = broadcast of all 3; ORT charges max of profiler-observed | `Where(BOOL, BOOL, BOOL)` rejected; `Where(BOOL, INT8/INT16, ...)` rejected; **`Where(BOOL, UINT8, UINT8)` works at opset 13+**; `Where(BOOL, F16, F16)`, `Where(BOOL, F32, F32)`, `Where(BOOL, INT32, INT32)`, `Where(BOOL, INT64, INT64)` all work |
| Equal / Greater / Less | output is BOOL of broadcast shape | `Equal(BOOL, BOOL)` works at all opsets ≥ 1; `Equal(F32/F16, F32/F16)` requires opset 11+ (NOT opset 10). `Greater(BOOL)`/`Less(BOOL)` always rejected. `Greater/Less(UINT8/INT8/INT16)` work at opset 10+ — see `ort_reality.md` (dtype matrix) |
| Constant node with value `[1, 10, 30, 30] F32` | params: 9,000 element count | NOT bytes; element count |
| Initializer `[1, 10, 30, 30] F32` | params: 9,000 | Same — element count |
| Initializer `[1, 10, 30, 30] BOOL` | params: 9,000 | Element count is SAME as F32; dtype doesn't help params |
| `[1, 10, 30, 30]` BOOL initializer in `memory`? | 0 | Initializers don't count in `memory`. They count in `params`. |

## Cost bands (verified by shipped tasks)

### Cost ≤ 100 (theoretical floor)
Two distinct sub-cases:
- **Pure structural identity** (`input → Identity → output`, `input → Transpose → output`, `input → Cast → output` when input/output share the same shape): cost can be **literal 0** because the model has zero `value_info` intermediates and no `params`. These yield the 25-point ceiling via `max(1.0, cost)` guard.
- **Constant-output via initializer/Constant node**: params = element count of the output buffer the initializer provides. A `[1, 10, 30, 30] BOOL` constant-output costs 9,000 params, NOT ≤100. Only genuinely small constant outputs (small grids, scalar) can hit the sub-100 band this way.

**Realistic floor for any task with non-trivial per-cell output**: ~500-1,500 bytes.

### Cost 100-3,000 — shape-only + minimal cast
- task299 bool-Concat: cost 1,058. Algorithm uses Concat of bool intermediates + Pad. No Conv, no MatMul.
- Fixed rotation via Slice→Gather→Pad sandwich: 1,091 (was 36,030 with full-canvas Gather).

Worth chasing this band when the algorithm is purely "rearrange the input pixels".

### Cost 3,000-30,000 — narrow intermediates + small Conv
- task374 conv_run_combined: 36,315 → 24,587 after polish. One Conv path with F16 intermediates.
- task343 crop15 + onnxsim: 12,872 (was 19,108). Crop-to-active-grid + F16 placement combined.
- task071 ring-dyn at rank-2: 24,150 (was 26,859, params 2,140 → 133).
- task115 axisfactor: 14,087 (was 149,687, -135k via factorization).

### Cost 30,000-300,000 — full-canvas F32 intermediates
- Single-Conv on the full input + scoring overhead lives here; "haven't been F16/cropped yet" territory. Most remaining mid-band targets sit here — count the planes (taxonomy above) before chasing a number.

### Cost > 300,000 — many F32 [1,10,30,30] intermediates OR large LUTs
- Usually the first 30 minutes of work brings these into the 30k-300k band.

## Decision tree

When designing an ONNX for a new task, ask in order:

1. **Can the output be a constant?** (Is the answer purely a function of the task ID, not the input?) → ~0 cost. Rare but worth checking.

2. **Can the algorithm be pure shape-only?** Slice + Gather + Pad + Concat + Reshape + Transpose. → ~1,500 floor. Applies when the task is "rearrange input pixels" with no per-cell decision.

3. **Can it be done in BOOL throughout?** All masks, no F32 arithmetic. → ~9,000 floor (one `[1, 10, 30, 30]` BOOL intermediate). See techniques.md rule 8 (input→BOOL whole-graph).

4. **Does it need one Conv on a narrowed (cropped) input?** Crop to active grid first → narrow intermediates. → 10,000-30,000 band.

5. **Does it need multi-stage decision (period detection, CC analysis, etc.)?** → 50,000-200,000 band. task110 is the exemplar — 7 candidates × 2 directions × scoring tensors.

6. **Does it need MatMul/Gemm/big lookup table?** → likely > 200,000. Consider if the table is compressible (separable_mask_collapse), packable (bit-pack), or replaceable by dynamic compute (rank reduction).

(Worked price-shopping examples — task110 breakdown, the F16 pre-cast anti-example: `_archive/cost_floor_evicted_2026-06-10.md`; the F16 placement law itself is traps §18 / techniques §4.)

## Constants and initializers — separate budget

The above table is `memory` (intermediates). `params` is a separate budget:

- Each non-zero-dim initializer element counts as 1 param (dtype-agnostic).
- Constant nodes' value attributes also count.
- A `[1, 10, 30, 30] F32` initializer = 9,000 params (NOT 36,000 bytes; element count).
- A `[1, 10, 30, 30] BOOL` initializer = 9,000 params. Same. Dtype doesn't help params.
- INT32 narrowing does not reduce `params` either; it only reduces intermediate `memory` when the tensor is
  produced/transported at runtime.
- All-initializer generator subgraphs are not automatically free under the scorer. If a `Tile`/`Range`/
  `Gather`/`ConstantOfShape` result survives as an intermediate, its output bytes are charged; the win is
  param-side only unless the generated tensor is the graph output or is absorbed inside a free-output op.

If your algorithm needs a 30×30 lookup table:
- Stored as F32 init: 900 params.
- Stored as BOOL init: 900 params. **No savings via dtype.**
- Stored as `[1, 30]` separable + broadcast at use: 30 params. **30× savings via shape.**
- Stored as a fully computed-on-the-fly value: 0 params, but adds compute nodes.

This is why `separable_mask_collapse` is high-leverage: shape compression saves params; dtype compression doesn't.

## Recolor / channel-permutation floor = C (proven 2026-05-30)

Any recolor whose rule is a color permutation π on the C=10 one-hot channels has a **hard floor of cost = C = 10**, realized by a single `Gather(input, idx[C], axis=1)` (memory 0, params C). This is provably minimal — no sub-10 exists:
- Every data-carrying intermediate is `[1,k,30,30]` ≥ 900 B, so a cost-≤9 graph can have **no** intermediate; the last op must read `input` directly.
- Only `Gather` emits a channel-permuted `[1,10,30,30]` cheaply; its index must have C entries (C params). Transpose can't permute within an axis without a ≥9000-B reshape intermediate; MatMul/Conv permutation matrices are ≥100 params; Slice/Concat reassembly is ≥900 B.
- The one sub-10 accounting path (sparse index, NNZ = C−1) is **categorically rejected** — Gather forbids sparse indices (3 scorer gates). See `core.md` sparse note + `task337.md` Floor proof.

**Implication: do NOT sub-floor-hunt GENERIC single-Gather recolor tasks.** Verified 2026-05-30: 016 `[0,5,6,4,3,1,2,7,9,8]`, 276 `[0,1,6,3,4,5,2,7,8,9]`, 309 `[0,1,2,3,4,7,6,5,8,9]`, 337 `[0,1,2,3,4,8,6,7,5,9]` are all generic permutations already at floor 10. The Reshape+Transpose "block trick" never beats a 10-param Gather (it carries a ≥9000-B intermediate).

**The ONE clean sub-10 exception — channel REVERSAL.** If a recolor's index is the exact reversal `[9,8,7,6,5,4,3,2,1,0]`, use `Slice(input, starts=[9], ends=[-11], axes=[1], steps=[-1])` instead of Gather: 4 one-element index inits, memory 0 ⇒ **cost 4 (23.61 pts)**, byte-identical, no accounting gimmick (verified). Transpose can't reverse within an axis; opset-1 attribute-Slice has no `steps`; so 4 is the reversal floor. **Quick test before shipping any recolor: `idx == list(range(9,-1,-1))` ⇒ reversal ⇒ Slice-reverse (cost 4); else ⇒ generic ⇒ Gather floor 10; `idx == list(range(10))` ⇒ identity ⇒ cost 0.** Full proof: `task337.md`.

## Single-channel color-index canvas = 900 B is the floor for a [1,10,30,30] output (proven 2026-05-30)

When a task's OUTPUT is the full `[1,10,30,30]` one-hot canvas, the cheapest *counted* way to produce it is a single-channel `[1,1,30,30]` U8 color-index canvas fed to a final `Equal(canvas, channel_ids[1,10,1,1])` — the BOOL `[1,10,30,30]` Equal result is the graph OUTPUT (outputs aren't counted), so the only counted cost is the **900-B U8 canvas** (+ a tiny `[10]` channel-ids init). This is the canvas+broadcast pattern (task387 / 009 / 054 / 388 family). Extreme tiny answer cells invert this: task056's `[1,10,1,1]` Equal-tail costs only 10 B, and folding an affine scalar bias into the `ids` vector can avoid a separate Add/Sub.

**Non-obvious corollary — do NOT try to eliminate the canvas itself.** task388 v6 tried to skip the `[1,1,30,30]` canvas and emit a BOOL `Equal` directly off a smaller `[1,10,12,12]` intermediate: it scored **2,546 — WORSE** than v5's 1,998, because the `[1,10,12,12]` BOOL counted intermediate (1,440 B) is *larger* than the `[1,1,30,30]` U8 canvas (900 B). The 900-B single-channel canvas is the architectural floor whenever the active region ≳ 10×10; only when the active region is genuinely tiny (task384-style ≤10×10 crop, output via the fold-mask-into-gather trick) does a cropped BOOL output beat it. **Price both before choosing:** full canvas = `30*30*1 = 900 B` (U8) vs cropped-BOOL-output = `10 * h * w` B (crosses 900 at h·w = 90, i.e. ~9×10). If the producer plane is already U8, the 10×10 tie flips toward the 900-B canvas (task035). See also techniques #81 (pad-floor crossover) and #116 (direct OneHot output crossover).

**Small-region rotate/reflect floor (task140).** For a fixed K×K one-hot region re-embedded in the
30×30 output, collapse channels first, rotate/reflect the U8 color-index crop, then Equal-expand the
cropped output and Pad. Measured K=3 rot180 floor: F32 crop-collapse Conv 36 B → Cast U8 9 B →
U8 Slice-rotate 9 B → cropped BOOL Equal `[1,10,3,3]` 90 B + small params = **210**. Rotating the
F32 canvas was 229, and padding to a full `[1,1,30,30]` canvas before Equal was 3,739. QLinearConv as a
whole-input F32→U8 dodge priced 9,180 because it first materialized the full U8 input.

**Empty-cell sentinel gotcha (task275).** When you collapse a 10-channel one-hot to a single-channel color-index canvas, all-zero (background / empty) cells decode to color 0 unless you inject an explicit out-of-range sentinel: `canvas = color_index + (1 - ReduceSum(onehot, axis=ch)) * 10` makes empty cells map to value 10 (≥ depth) so `Equal(canvas, [0..9])` yields all-zero one-hot there, matching the canonical. Missing this is a silent OOD bug — task275 v1 caught it on 1/2000 random pairs only.

**Canvas+Equal BEATS ScatterND channel-patch for partial-recolor outputs (proven 2026-05-31, task070).** When the rule recolors only K of the 10 output channels and leaves the rest = input, `output = ScatterND(input, [[0,c1],…], updates)` looks free (input and output both uncounted) — **it loses to the canvas**: `ScatterND.updates` must match the **F32** input dtype, so K channels cost `K*3600` B F32 + `K*900` B BOOL precursors; the K=2 anchor (~9,000 B) alone exceeds the whole canvas core (~6,300 B). Measured: task070 channel-patch **15,956** vs canvas **8,958**. The Equal-expand gets ALL 10 channels free; ScatterND pays F32 per patched channel. **Rule: for "recolor/patch a few channels of a one-hot output", use the single-channel canvas + `Equal(canvas, ids)`, not ScatterND.** Marginal win: apply the recolor `Where` on the active-region crop and `Pad` the recolored canvas once — but only when the grid size is provably fixed (task070: ARC-GEN hardcodes 17×17).

**F32 input-read tax (tasks026/040/299/341, 2026-06-11; task271 CARVE-OUT 2026-06-19).** A direct
`Slice`/`Gather` of the graph input MATERIALIZES it in F32 — THAT result is the irreducible `channels*h*w*4`
read until a narrowing op. **The tax hits a materialized Slice/Gather result, NOT reading the input:** a
`Conv`/#98 dilated crop-collapse reads the free `input` and pays only OUTPUT bytes (task271 was mis-listed
as a victim — its #98 Conv reprices **1942→1136**). Two non-adjacent narrow slices can beat one wide slice,
but no dtype trick makes a materialized source read U8. Sparse points → #145 flat `GatherND`; fixed
sub-rectangle channel-reduce → #98 when kernel params beat the F32 crop. **Few-column extraction (task010):**
per-column `Slice` of the free input (4×144 B + Cast + Concat ≈ 360 B) beats slicing a materialized `[9,9]`
(486 B); F16 basis optimal. **Bitset cross-line floor (task047): 1003** (736 B / 267 p, opset-18 `BitwiseAnd`). 2026-06-24 confirmations: task255 one-channel read = 3,600B; task370 F32 slice before U8 QLinearConv, cast-full-input loses; task198 Slice+Cast double-charges.

**Per-cell data-dependent recolor floor (task040, 2026-06-11).** A KxK
data-dependent two-way recolor is not just "one mask plus output": source crop,
condition, replacement field, and result are all real planes unless the final result
is the graph output and the replacement is a fixed one-hot vector. If the
replacement is built from data-dependent row/col vectors, the broadcast
`Where(col_axis, vec_w, vec_h)` is a charged KxK field. Price the four-plane floor
before reporting a 3-plane shortcut.

**Full-canvas Where-overlay tax (task341/task102/task129).** A free-output
`Where(cond, paint, input)` still needs `cond` at the broadcasted output extent.
If the unchanged branch is the full `[1,10,30,30]` input, the condition is usually
a `[1,1,30,30]` BOOL = 900 B floor even when the logical active region is 10x10.
Add any entry/source read tax on top (for example a `[1,1,10,10]` F32 slice = 400 B).
Use ScatterND tiny-patch (#139) only when the edit is a few localized cells.
**Where-condition broadcast pin (task064 W4, 2026-06-24):** `crop_to_size_advisory` is blind to broadcast constraints on Where conditions. A `[1,1,24,24]` condition cannot broadcast to a `[1,10,30,30]` Where output (spatial dims must be 1 or 30). Doing `Where` in small space then Padding charges `[1,10,24,24] F32 = 23,040 B` (25× worse than the 900 B BOOL condition). When the charged plane is the *condition* of the output `Where`, its extent is forced by the OUTPUT tensor shape, not by the active grid.

**Multi-scale detector floor (task158, 2026-06-22 — generalizes #134):** per-scale planes are irreducible when any part of the acceptance predicate is fixed in absolute cells while the accepted shape is scale-proportional. No pooling or Resize can carry both constraints through a shared canonical detector. Count channel firing over the corpus before declaring a channel dead.

**Free-output single-Conv params floor (task283, 2026-06-11).** If one Conv output
is the graph output, memory can be zero but params remain
`M * (C/group) * kH * kW + bias(M)`. For cross-channel 10-color 3x3 rules,
`group=1` forces roughly `10*10*3*3 + 10 = 910`; a grouped/depthwise Conv only
applies when every output channel reads its matching input group and no background/
deletion channel needs cross-channel evidence.

## Shift / translation floor = gathered-axis length (proven 2026-06-01, task053)

A "translate the grid by (dr,dc), zero-fill" rule has a hard floor of **cost = the length of the shifted axis** (e.g. **30** for a 1-row shift on the 30-row canvas), realized by a single `Gather(input, idx[30], axis=2)` whose index sources an *existing background row* into the vacated band (memory 0 — the Gather output IS `output`; params = index length). task053 exemplar: `idx=[2,0,1,29,…,29]`, **cost 30, proven optimal**.

Why nothing cheaper works: the 0-cost `Pad`-shift is **byte-WRONG** under the ch0-hot background encoding (traps §31 — diverged on 2060/2060); a 20-param depthwise-Conv shift also diverges (geometric zero-fill ≠ the `ch0=1` background the vacated band needs); `Slice` row orders are strided-arithmetic only; sparse index NNZ=29 is scorer-rejected (§8e). Only a primitive that **relocates an existing background row** is correct, and that needs the full-length index. Multi-op assemblies all exceed 30 (measured: Gather3+Pad 3,603; Slice+Concat 7,205). **Do NOT sub-floor-hunt single-`Gather` shift tasks** (~5–15 frame-shift tasks in the set). Full proof + probe harness: `task053.md` + `Task_by_Task/task053/prototypes/build_and_probe.py`.

## Quick gut checks

- `[1, 10, 30, 30]` F32 intermediate = 36,000 bytes. **The largest common full-canvas F32 activation** — anything larger (e.g., INT64 = 72,000, or any `[…,K]` expansion) is avoidable and almost always the cost dominator if present.
- Crop to `[1, 10, 10, 10]` = 4,000 bytes. **9× smaller.**
- BOOL `[1, 10, 30, 30]` = 9,000 bytes. **4× smaller.**
- Drop the channel dim entirely (`[1, 30, 30]`) = 900 bytes F32. **40× smaller.**
- A single scalar = 4 bytes. **9000× smaller** than full canvas F32.

For a task at cost 200,000, asking "where are my biggest intermediates and can each one shrink 4-10×?" is the right question. Inventory them with `tools/agent_helpers/largest_intermediate.py --top 20`.

## The plane-conservation-law floor method (task328, 2026-06-04 — promote-worthy proof style)

Stronger than per-route rejection: define the task's per-cell DECISION CORE (e.g. "2 ternary decisions x leg-compare x parity x 3-state output"), then price EVERY legal op family against it — task328 found every 2-input formulation lands within ~150 B of the same ~19-plane total (even the Sign/Clip-code + wrap-Gather-LUT route ties exactly at 3,240 before losing on LUT build cost; packed payloads are plane-neutral because each pack costs one Div/Mod unpack). Escapes must come from ops with >2-input fan-in per materialization (3-operand Einsum corner reads bought −900 there). Write the law with numbers in the task .md — it converts "couldn't go lower" into "no legal op can".

**Packing-law refinement (2026-06-04, Research H2):** the "packed payloads are plane-neutral" findings are the α≈0 special case of the general law (`techniques #45`): when no consumer executes on packed words, packing is strictly worse — those floors STAND. But graphs with ≥4 packed-executable consumer sites (U8, n=8) ESCAPE: re-audit conservation-law floors only when the decision core is rich in pure AND/OR/XOR stages.

## Floor verdicts — compact ledger (one row per shipped/proven floor)

Full dated ship-blocks: `_archive/cost_floor_full_pre_phase1.md` + `_archive/cost_floor_evicted_2026-06-10.md`.

| task | floor / note |
|---|---|
| task193 | 55,657 → **5,281**. F32 `mark` `[1,1,20,20]` Conv plane is ORT-forced |
| task287 | 50,758 → **3,016**; **memory floor 2,948 proven exhaustively** |
| task367 | **19,334** floor; connector false-walls defeat global/separable rectangle fill, shipped 25-channel formulation is current floor |
| task308 | 49,062 → **2,990** (byte-equal seqrect) |
| task328 | 47,384 → **8,367** (byte-equal; einsum-corners + mod-parity) |
| task364 | 46,250 → **14,162** (negbias-cliff §43; THEOREM-GRADE floor) |
| task049 | 44,787 → **748** (CAMPAIGN-RECORD 59.9×) |
| task187 | 49,813 → **21,282 interim** (scanline) |
| task363 | 43,636 → **3,796** (byte-equal full 200,265; tie-break decode held) |
| task338 | 45,649 → **8,469 CLOSED** (negpad v8, +1.685 exact) |
| task252 | 39,858 → **280 (CAMPAIGN RECORD +4.958, 142×)** — zero-memory |
| task293 | 37,940 → **838** (+3.81 exact; batch-free free-input Einsum) |
| task374 | 34,392 → **2,831** (+2.50 exact; QLinearConv neighbor sums + Slice crop) |
| task094 | 39,015 → **3,806 static** (+2.33 exact; the 313/451 dynamic rungs repriced ≈24k/19k — §GRADER-GAP) |
| task064 | 39,510 → **14,706** → **9,743** byte-equal v20. The 9,707/9,715 count-span routes are Path-B dirty on padding-perturbed LIVE (341/163 A/B divergences); robust in-box span needs the 48-param coordinate index. **Where-condition broadcast pin** still makes 900 B BOOL the floor. |
| task025 | 36,433 → **23,495 k5-exact** (k4 19,355 @99.88% parked) |
| task359 | 40,808 → **4,510 zero-params** (+2.20) |
| task270 | 38,430 → **2,978** (+2.56 exact; center-noop probe). gate slow candidates with `--n 20000` |
| task398 | 39,875 → **≈13,300 GRADER-CREDITED** (local 364–519 = 37× §GRADER-GAP; +1.10 banked). Honest floor est. 1.5–2.5k OPEN |
| task054 | 47,180 → **26,043** (ScatterElements axis=3; <10k/<1000 PROVEN impossible — min one plane ≥4,520) |
| task080 | 49,405 → **5,664** (FIRST F16-Conv ship grader-validated; info-flow floor ≈5,050) |
| task092 | **3,980** term-axis-last free-output span terms; layout deleted Reshape/shape-init scaffolding |
| task158 | 24,148 v111 convolutional-family floor. Per-scale detector planes are a #134 FLOOR when rejection geometry is fixed-size and accepted shape is scale-proportional. `AveragePool` rejects U8 (ort_reality). |
| task174 | **2,225** full-palette TopK color ids + small-pad crossover; exact-K non-bg colors let TopK return ARC ids directly |
| task209 | 4,922 v14 (beats lane 4,995). ConvTranspose/DepthToSpace dissolves WRITE side only; READ plane+matcher irreducible. Sub-2,000 not reachable. |
| task204 | **7,682** two-color size-parity fill floor; 8 length channels + 10x10 paint kernel forced, Constant index nodes are worse than inits. |
| task366 | 37,577 WALL. Per-cell consumed-state flats don't collapse to per-component scalars (consumption is per-cell). |
| task392 | 1,009 FLOOR (shape-table LUT; ~40 distinct patterns, rank up to 10 on 7.5% of corpus; runtime-decoded selectors non-initializable). |
| task034 | **2,488** (-986 vs LIVE 3,474). Locate via free-input Einsum marginals; read K cells via dynamic Slice folding channel+hw into ONE node. |

## Salvage from empirical-cost-probes.md (Phase-2)
- \* Memory column **as reported by onnx-tool**. For single-Conv graphs the only intermediate is the output tensor, which the validator excludes; effective memory contribution = 0. For multi-node graphs the memory term is real and additive.
- The COVERED table's `cost = macs + params` shortcut is **misleading** outside the single-Conv regime (memory is real and additive for multi-node graphs).
