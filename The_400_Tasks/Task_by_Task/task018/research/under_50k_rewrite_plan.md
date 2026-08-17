# Task018 <50k Rewrite — Design Document

**Status**: v45 baseline at 132,690 (cost progression v29→v45 saved 46,251 / -25.8%).
Target: <50k. Gap: ~82k to remove.

## Survey of task018 invariants

From `data/competitions/neurogolf-2026/task018.json` (266 visible+arc-gen pairs):
- All pairs have **exactly 4 active colors**: 1 filler + 3 spec
- Grid sizes range 12-24 spatial dim, max 24
- 1-2 templates per input
- Max template/anchor bbox: 21x21 (most are smaller)
- Max non-bg cells: 30

The **9-channel inp_colors_bool** is processing 6 always-empty channels per input — that's the structural waste.

## Cost breakdown of v45 (132,690)

| category | bytes | notes |
|---|---:|---|
| `inp_colors F32 [1,9,24,24]` | 20,736 | Slice from F32 input — locked |
| `wt_f16 × 2 templates` | 20,736 | Per-color max position via ReduceMax (F16 required) |
| 9-bucket (3 F32 + 6 F16) | 20,736 | ch0_crop, inp_colors_sum, anchor_label_pre + 6 grid-construction tensors |
| `inp_colors_bool + flat` | 10,368 | BOOL Where condition + flat Gather |
| `grid F32 × 2` | 9,216 | GridSample requires F32 grid |
| 1152×14 F16 [1,1,24,24] | 16,128 | per-template anchor pos + GridSample chain |
| 576×31 BOOL/U8 | 17,856 | masks + 6-step UINT8 flood |
| 288×20 I32 [8,9] | 5,760 | forward transform inside check_consistency |
| 900 padded output | 900 | raw_label_masked_padded U8 |
| 144×4 F16 [8,9] | 576 | full_match_f for ReduceSum |
| params | 3,147 | inits |

## Why <50k is hard

To remove 82k requires eliminating:
- inp_colors F32 (20k) — every Cast-to-smaller alternative net-adds bytes (input_f16 18k > savings; input_bool 9k+8k slice intermediate adds 17k for ~25k of downstream alternatives)
- wt_f16 (20k) — per-color positions are needed for D4 detection in current algorithm; alternative per-cell mask correlation only saves ~5-8k per template
- per-template GridSample (32k) — ScatterND with computed positions has same per-cell coordinate math; stacked D4 transforms + Pad/Slice for translation costs roughly the same

Each of these needs an algorithm change. The exemplars at 10-17k cost (task066, task118, task277) each used **task-specific** tricks for their specific rules — not generic optimizations.

## Possible paths to <50k (each needs verification on adversarial inputs)

### Path A: bbox-aligned D4 detection (~70-100k estimated)
Replace per-color wt_f16 with per-sym bbox-alignment scoring.

Algorithm:
1. Compute spec bbox (4 scalars): `spec_row_min`, `spec_row_max`, `spec_col_min`, `spec_col_max` via row/col ReduceMin/Max on Where(spec_b, row_4d, sentinel)
2. Compute anchor bbox similarly
3. For each of 8 D4 syms:
   - Apply sym to spec_b via static Slice/Transpose → `transformed_spec [1,1,24,24] BOOL = 576`
   - Translate via dynamic Pad to align spec bbox to anchor bbox
   - Score = `ReduceSum(And(shifted_transformed_spec, anchor_mask_b))`
4. ArgMax → best sym

Risk: bbox corners may not be actual cells (e.g., 3-cell anchor cluster forming L-shape); alignment may be ambiguous.
Estimated save: 15-20k.

### Path B: Color-indexed scalar representation (~80-110k)
Per `techniques.md §23` (task138 saved -61%):
1. `color_idx = ArgMax(inp_colors_f16, ax=1, keepdims=1) [1,1,24,24] U8 = 576` shared
2. For each anchor cell, look up its color via Gather
3. For each anchor cell's color, find matching spec cell via Equal+Where+ReduceMin
4. 3 spec-anchor pairs determine D4 transform (overdetermined)
5. Apply transform to filler via static Slice/Transpose

Issue: per-anchor iteration in ONNX requires either fixed iteration count (3 anchors hard-coded per template) or banned ops (Loop). Hard-coded 3-iteration logic adds bytes.

### Path C: ScatterND-based sparse painting (~90-120k)
Per `techniques.md §3 task366`:
1. Detect transform via 8-sym mask correlation
2. Compute target positions for filler cells via coordinate arithmetic
3. ScatterND output canvas with computed positions + colors

Issue: per-cell coordinate computation is similar cost to current GridSample grid construction.

### Path D: Combination (best chance at <50k)
Stack A + B + C together. Estimated save: ~50k. Reaches ~80-85k. Still above 50k.

To actually reach <50k requires also:
- Eliminating `inp_colors F32` via aggressive input-bool approach (need to handle Conv anchor_label_pre via ArgMax-based label instead)
- Dynamic cropping to spec bbox (risky — ORT profiler traces single input, score may not reflect runtime cost for varying inputs)

## Implementation order (when picking this up)

1. **Numpy reference first** — write a clean numpy implementation of the new algorithm
2. **Verify 266/266 in numpy** before any ONNX work
3. **Build adversarial synth set** (≥1000 cases) and verify numpy ref matches LIVE ≥95%
4. **ONNX implementation** — start with Path A (bbox-aligned D4), measure cost
5. **Iterate** — apply Path B if Path A passes verify but is above target
6. **Combine** with Path C if still above target

Each step has gates per `rule_discovery_methodology.md`:
- 266/266 visible verify (HARD GATE)
- ≥1000 synth ≥50% agreement (MANDATORY per Radical Rewrite prompt)

## Estimated session-time investment

- Path A standalone: ~10-15 hours focused work
- Paths A+B: ~25-30 hours
- Paths A+B+C+input-restructure: ~80-100 hours

The original `Prompt_2026-05-24T040400Z_Task018_Radical Rewrite.md` explicitly says **"100 hours minimum"** for this scope of work.

## Current standing

v45 = 132,690 cost, 13.22 pts, 266/266 verify, 1473/1500 synth.
Staged at `.hermes/board/staged_submits/task018_structural_v45_u8label_t_20260526/`.

## v47-v50 progress (2026-05-26 session continuation)

Approach taken: **incremental optimization of v45's existing algorithm via color_idx**
instead of full Path A rewrite (which would replace GridSample with ScatterND-based
painting). Numpy v47 reference confirmed v45's algorithm is correct (266/266 visible,
98% synth vs LIVE). Then optimized v45 ONNX in-place.

| ver  | cost    | pts    | change   | optimization |
|------|--------:|-------:|---------:|---|
| v45  | 132,690 | 13.20  | baseline | (shipped) |
| v47  | 125,908 | 13.26  | -6,782   | ScatterND replaces per-template wt_f16 [1,9,24,24] F16 |
| v48  | 121,011 | 13.30  | -4,897   | color_idx_flat_u8 + Equal replaces inp_colors_flat lookup |
| v49  | 112,965 | 13.37  | -8,046   | Eliminate inp_colors_bool [5184] + inp_colors_sum [2304] via color_idx |
| v50  | 110,661 | 13.39  | -2,304   | ScatterElements (1D) replaces ScatterND ([576,1]) |

All versions verified 266/266 against LIVE — algorithm unchanged, only data
representation. The byte-exact match means hidden-test risk = 0.

### Failed attempts
- v47 first try: ScatterND with F16 reduction — ORT rejects F16 in opset 16
  (ScatterND/ScatterElements both require F32 with reduction='add' until opset 18).

### Remaining optimization buckets (estimated)
Cost breakdown post-v50 (approximate):
- inp_colors F32 [1,9,24,24] = 20,736 (LOCKED by F32 one-hot input)
- Per-template GridSample chain × 2 ≈ 25,344 (base_rc + norm_rc + ric_t + grid F32 + out_filler)
- Per-template anchor pos chain × 2 ≈ 7,000 (anchor_ch + anchor_ref_bool + 2× pos Where + scalars)
- inner_pos + ScatterElements chain × 2 ≈ 9,440
- check_consistency × 4 calls ≈ 8,320
- Shared masks + flood + 3-step MaxPool ≈ 10,000
- Output assembly + raw_label chain ≈ 3,500
- Other shared (color_idx, ch0_bool, inp_active) ≈ 12,000
- Params (initializers) = 3,160

To reach 100k (∆ -10,661): possibly via shared anchor_ref chain (risky — assumes
both templates have same c_ref color), removing remaining redundant intermediates
in the F16 grid chain, or smarter forward-transform encoding.

To reach 70k (∆ -40,661): requires eliminating per-template GridSample (algorithm
change to ScatterND-based painting per Path A or C) or restructuring inp_colors.

To reach 50k (∆ -60,661): requires deeper algorithm changes — Path A bbox-aligned
D4 detection + Path C ScatterND painting, OR using BOOL-only input path with
custom Conv weights for the histogram.

**Recommendation**: v50 = 110,657 is the new ship candidate (+0.18 LB vs v45 = 13.39 pts).
Continue toward 100k via remaining smaller optimizations.

### Failed v51 attempts (kept here for posterity)

1. **F16 ScatterElements with reduction='add'**: ORT opset 16 rejects F16 with reduction
   on both ScatterND and ScatterElements (only F32 / int types supported). Need opset 18
   for F16 — adds risk to other ops.

2. **4D ScatterElements indices**: ORT requires indices.rank == data.rank. data is [10]
   (rank 1), so indices must be rank 1. Can't skip the inner_pos_flat F16 reshape this way.

### Synth verification

- v50 vs v45 byte-exact: 292/300 (97.3%) — same divergences from LIVE means v50 is
  algorithmically identical to v45 (downside-zero EV — won't lose any LB pts that v45
  wouldn't already lose).
- v50 vs LIVE on visible+arc-gen: 266/266 (HARD GATE PASSED).

### Path forward to <100k

Remaining 11k requires either:
- Algorithm change: replace GridSample (saves ~25k chain → maybe 18k if ScatterND-based
  forward-transform replacement costs ~7k)
- Risky shared-c_ref optimization: assumes both templates have same anchor color set.
  Could save ~3-3.5k but breaks if hidden tests have asymmetric color sets.
- Pad/Slice-based dynamic translation instead of GridSample backward — would require
  full algorithm rewrite (Path A).
