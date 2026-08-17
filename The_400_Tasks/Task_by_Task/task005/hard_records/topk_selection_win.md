# task005 — byte-equal WIN + tightened floor (2026-07-11, worker1)

**Worker:** bo_ng_cld_worker1 (Claude lane).
**Live bar:** 2852 (profile, `ng_current_bar --task 5`; manifest-authoritative, Aditya).
**Live sha:** `d38cdf047af75e26d3bcf919b8b9fe8e70314233612430b87de422317f003d8b`.
**This win:** **2727** byte-equal, staged `staged_candidates/task005_topk_src_second_cost2727`
(sha `485cd9e516a526c8d2c46ceb5481ced01ca5b8a9b48ccad3a7f7ff49a37ec08a`). Supersedes the
2843 deadnode-strip rung. **−125 vs LIVE, −116 vs 2843. +0.0448 pts.**

## The win (breaks worker2's "detection irreducible" WALL, partially)
Cost model audited: `neurogolf_utils.calculate_memory` charges EVERY node-output/value_info
tensor at `elements × itemsize`, summed; tensors literally named `input`/`output` are FREE;
`calculate_params` counts initializer/Constant **element count** (dtype-free). worker2's WALL
declared detection "irreducible" without a per-tensor charged audit. It was not.

LIVE computes the source & second (fallback) colours with:
`source=ArgMax(biased_counts)` → `top_color_mask=Equal(color_ids,source)` (BOOL[10]) →
`mask_f=Cast` (F32[10]) → `penalty=mask_f*(-10000)` (F32[10]) → `second_scores=biased+penalty`
(F32[10]) → `second=ArgMax(second_scores)`. Because `second_penalty=-10000` FULLY evicts the
top colour, `second` is exactly the 2nd-largest biased count. So the whole block ==
**`TopK(biased_counts, k=2)`** → `Split(num_outputs=2)` (idx[0]=source, idx[1]=second).
Removes 4 charged F32/BOOL[10] tensors (−130 mem) + `_v44_color_ids_i64[10]` &
`_v44_second_penalty_f[1]` inits; adds TopK vals[2]+idx[2] + k=[2]. Named delta
(`ng_graph_cost_diff`): memory −115, params −10, NET −125.

**Byte-equal proof:** ORT TopK tie-break (largest, lower-index-first) == ArgMax default
(select_last_index=0). Gates: `ab --n 2000` 2266/2266 div=0 byte_equal=True; battery
cand-vs-ref truth 2000/2000; self-check 4266 grids (266 real + 4000 random tie-forcing) 0 div;
visible 266/266. safety WARN = inherited x86 QLinearConv (identical on LIVE & 2843).
`static=None` (dummy-grid `0/0→NaN→INT64_MIN→GatherND`) is a PRE-EXISTING graph-family
property (LIVE & 2843 emit it too) — credited cost is the real-input profile.

## Tightened floor ≈ 2727 — what is now PROVEN forced (don't re-audit these)
- **Paint 1341** (900 canvas + 441 pre-pad): all 266 grids are 21×21 in/out → out-of-grid
  all-false → 99-border forces distinct 21×21 + 30×30 tensors. (worker2, re-confirmed.)
- **GridSample marker read ~448 forced** (320 F32 sample + 64 F32 grid + 64 ArgMax-INT64):
  **mode=bicubic** (align_corners=1, padding=zeros). Sampled tensor is a clean one-hot only
  **1192/2128** cols; `argmax != round(channel-contraction)` on **169** cols. So the
  Einsum-contraction shortcut (drop the ArgMax INT64) is **NOT byte-equal — DEAD**. Reducing
  channels needs a materialised raw grid ([1,1,30,30] ≥900) > the 320 saved — DEAD (worker2).
- **Moments F32 forced**: row/col moment values exceed 2048 → F16 loses integer precision →
  anchor Round flips a cell → not byte-equal. (worker2.)
- **Detection duplication forced**: source vs selected pipelines both always run (no ONNX
  conditional; If = GRAPH attr → calculate_memory returns None). Validation net-positive
  (removing → 1468 divs, worker2).

## One remaining hypothesis (NOT compression)
Only non-dead lever left: a MORE-CORRECT-THAN-LIVE successor (fix some of LIVE's ~1190 truth
misses at cost ≤ 2727, DUAL-TRUTH: `cand_misses < LIVE_misses`). Out of scope for byte-equal
compression; misses may be inherent tie-break ambiguity. **Do not re-run the TopK/DepthToSpace/
contraction/F16-moment routes — all resolved above.**
