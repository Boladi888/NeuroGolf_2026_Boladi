# task162 — WALL at 3496 (Sho `scaled_cast` is the proven correct floor)

- **Live bar:** 3496, SHA `b8951b5f95d030b98762cdab822b350a38b440247ccb68987af5551aa4317c92`
  (manifest, owner Sho, 2026-07-09; confirmed via `ng_current_bar.py --task 162 --json`).
  Verified **visible 266/266 PASS** — the bar is a genuinely correct model, not an artifact.
- **Verdict:** WALL. No correct candidate exists below 3496. Do NOT re-explore sub-3496.
- Session: `bo_ng_cld_worker3`, 2026-07-11.

## The bar model (decoded)

`score_network_v2` = (macs 0, memory 3320, params 176) = **3496**. opset 18, 6 nodes:

```
Conv(input, w_seed4_scaled[1,10,4,4], pads=[1,1,-10,-10]) -> seed_score F32[1,1,18,18]   (1296)
Cast(int8)                                                -> seed_i8   INT8[1,1,18,18]   ( 324)
MaxPool(k3, pads=[2,2,2,2], s1)                           -> cover_i8  INT8[1,1,20,20]   ( 400)
Cast(bool)                                                -> cover     BOOL[1,1,20,20]   ( 400)
Pad(cover, [0,0,10,10], axes=[2,3])                       -> mark      BOOL[1,1,30,30]   ( 900)
Where(mark, oh1[1,10,1,1], input)                         -> output
```
params: conv 160 + oh1 10 + pad 4 + axes 2 = 176.

**Mechanism.** The scaled 4×4 kernel (core ×0.10989 ≈ 1/9.1, top-row/left-col dedupe penalty
−0.02198) makes an all-black 3×3 anchor score ≈1.099 and a deduped/blocked anchor score <1.0.
`Cast(F32→int8)` truncates toward zero, so the cast IS the anchor+dedupe threshold at exactly 1.0
([1,2)→1, [0,1)→0), collapsing what earlier paths did in 3–5 separate planes into one Conv + one Cast.
Negative pads `[1,1,-10,-10]` position the 4×4 window over the active 20×20 (top-left of the 30×30).
This is why the old v4_int8 "min = 4672" proof was wrong: it assumed `Slice ch0 → ch0_i8 → QLinearConv`
(paying 1600+400 for the slice + separate sum3/anchor/nbr planes). Sho convs directly on the F32 input.

## Forced-component floor argument (why sub-3496 is impossible)

The rule (fill 3×3 all-black holes, top-left dedupe) forces the topology
`linear-conv → nonlinear threshold → nonlinear dilate → canvas mask → Where`:
a single conv/Einsum output for ch0 is **LP-infeasible** (`grouped_finalconv_linear.md`), so the
threshold and dilation cannot fold into the conv. Each intermediate is at minimal legal size:

1. **Head `seed_score` F32 1296 — forced.** Any spatial reduction (Conv/AveragePool/MatMul) on the
   F32 input yields F32; there is no ONNX op that reduces F32→int8 in one step. 18×18 is the exact
   count of 3×3-anchor top-left positions in a 20×20 grid. To get int8 seeds you must Cast *after* a
   reduction, so the 1296 F32 plane must materialize first. Alternatives are worse:
   - Slice/Quantize ch0 first → the F32 ch0 slice alone is 1600 (20×20) > 1296.
   - F16 conv needs an F16 input → Cast the F32 input = 18000 (`Conv F16-out on F32-in impossible`).
2. **`seed_i8` INT8 324 — forced.** Cast target = the threshold; also makes dilation cheap. Pooling
   the raw F32 instead (probe **P2**) is correct but costs the 1600-B F32 cover → **+878**.
3. **`cover_i8` INT8 400 — forced.** MaxPool 3×3 dilation. ORT rule `pad[dim] < kernel[dim]` caps a
   3×3 MaxPool output at 18→20; you **cannot** dilate straight to 30×30 (probe **P1** = ORT error
   `Pad should be smaller than kernel`). So a separate Pad to the canvas is mandatory.
4. **`cover` BOOL 400 — forced.** Where needs a BOOL condition. Casting at 20×20 (before Pad) beats
   padding int8 to 30×30 then casting (400+900 vs 900+900).
5. **`mark` BOOL 900 — forced.** Where output is 30×30 ⇒ condition must be 30×30. The changed cells
   are a data-dependent 2-D union of detected regions — NOT a separable row/col mask, so free-output
   Einsum / techniques #67/#81/#89 do not dissolve it (worker1 rework, still valid).
6. **params 176 — forced.** `[1,10,4,4]`=160 consumes all 10 channels so no ch0 slice is needed
   (folding channels 1–9 into a bias keeps the tensor `[1,10,4,4]`, scorer still counts 160; a
   `[1,1,4,4]` weight would need the 1600-B slice). oh1 must be the [1,10,1,1] blue one-hot for the
   final Where. Axes-form Pad (4+2) beats full pads[8].

Sum = 1296+324+400+400+900 + 176 = **3496**. No component can shrink without breaking correctness or
an ORT constraint. ∎

## Probes run this session (`prototypes/build_task162_bo_ng_cld_worker3_floor.py`)

| probe | change vs bar | result |
|---|---|---|
| P1 merge30 | one MaxPool 18→30 (pads [2,2,12,12]) to drop the Pad + `cover` plane | **ORT-illegal**: `pads[dim] < kernel_shape[dim]` — proves the separate Pad is forced |
| P2 f32pool | drop the early int8 Cast; dilate in F32, Greater(>0.999), Pad(bool) | **visible 266/266 correct**, cost **4374 (+878)** — proves the int8 cast is load-bearing |

Note (P2 threshold trap): a naïve `Greater(>0.5)` gives **0/266** — the int8-truncation boundary is
exactly 1.0, so an F32 re-expression must threshold at ≥1.0 (used 0.999), else deduped anchors
(score ≈0.945) wrongly fire.

## Formulations already priced (all ≥ bar) — do not re-run

- QLinearConv dedupe path (v4_int8) = 4727; directconv variants 4081–4089 (worker1) — all > 3496.
- Grouped final-Conv as graph output (memory 0) — LP-**infeasible** for ch0 (`grouped_finalconv_linear.md`).
- Smaller 2×3 anchor conv — **visible 229/266** (`smaller_anchor_directconv.md`); hidden-zero.
- UINT8 Slice-shift dedupe — tensor blow-up ≈9341 (task162.md don't-do).
- Sparse-write (NonZero→ScatterND) to avoid `mark`: NonZero needs the 900-B mask as input, so it
  adds tensors + a dynamic INT64 index (local-cost fiction) — strictly worse. Dead.

## Second pass (2026-07-11, forced re-derivation before review)

Profiled the bar: `profile_real_cost --arcgen 266` → **static=profile=3496** (fully static, no gap).
Mined `by_symptom.md` / `techniques_catalog.md` / library_db / BoBrain for a cheaper head. The three
highest-leverage candidate levers, all **disqualified**:

- **input→BOOL whole-graph (task376, "biggest leverage") — DEAD here.** Track A casts the input to BOOL
  first = `[1,10,30,30]` BOOL = **9000 B**, dwarfing the 1296 F32 head (Sho never materializes a full
  canvas). Track B (output→BOOL) saves only a *trailing* `Cast→F32`, which Sho's graph does not have
  (`Where` outputs F32 directly). Built **P3** = the bool/MaxPool anchor head (Slice ch0 1600 + Cast 400
  + Sub 400 + MaxPool 324 + Equal 324): head-alone = **3048** ≈ Sho's *entire* graph (3320), before any
  dedupe/dilate/mask. Root cause: isolating ch0 needs a 1600-B F32 Slice (> the 1296 conv head), and the
  top-left dedupe needs a per-cell-weighted kernel with a **zeroed self-corner — only Conv can do that**
  (MaxPool's rectangular window can't exclude self; shift-slab dedupe adds ~1600 B).
- **task090 MaxPool-forest replacement — N/A.** That lever replaces a *forest* of Conv/ConvTranspose shape
  detectors with UINT8 MaxPools. task162 has ONE anchor type and already uses ONE conv; MaxPool cannot do
  "all-black" (it is "any") without the non-black complement + isolated ch0 (the P3 cost).
- **DepthToSpace #80 block-stamp (task304) — N/A.** That expands a static motif into a fractal/tiled canvas
  in one op. task162's fill is a **data-dependent** union of detected regions, not a static tiling;
  DepthToSpace does non-overlapping sub-pixel expansion, which is not the required overlapping dilation.

**Global-minimum argument (strengthens the head=1296 claim).** Since the input is F32, some F32 tensor is
the first intermediate. Enumerate the cheapest first op that yields anchor-relevant evidence: Conv
`[1,10,4,4]`→18×18 F32 = **1296** (fuses channel-reduce + 3×3 anchor + 4×4 dedupe); Slice ch0 = 1600;
1×1 channel-reduce = 3600; any Pool needs an isolated channel first. So 1296 is the **global** minimum
first-F32-tensor, reachable only by a 1-channel-out conv at the 18×18 anchor grid — and only a ≥4×4 conv
also delivers the dedupe for free. The head cannot be cheaper by any assembly.

## One remaining (vanishing) hypothesis

The only unbroken lever would be a single ONNX node that reduces the F32 input to a ≤1-byte/element seed
plane (deleting the 1296 F32 head). None exists at opset ≤21: QLinearConv/ConvInteger require already-int8
input (F32 rejected — `ort_reality.md`); ConvInteger outputs INT32=4B; AveragePool is float-only;
ReduceMax/Min take int8 but reduce whole axes, not sliding 3×3 windows. **WALL at 3496.**
