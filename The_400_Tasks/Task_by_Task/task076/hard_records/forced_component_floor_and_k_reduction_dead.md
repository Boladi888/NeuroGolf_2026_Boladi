# task076 — forced-component floor (WALL below 12,030) + K-reduction dead-end

- **Live bar**: 12,539 (profile), owner Sho, SHA `ec95ba95e46eeb94984d1a20b525eaf4f94851b8fd63b04f53a9618ddabdd773`, from `ng_current_bar.py --task 76 --json`. **Ambiguous class** (LIVE truth 1961/2000 = 98.05%) → byte-equal-to-LIVE is the only gold gate.
- **Current best (staged, mine)**: 12,030 (`task076_int8coords_i32idx_cost12030`, SHA `aa5292e6…`), byte-equal to LIVE (AB 2266/2266). This record explains why **< 12,030 is a WALL for the byte-equal-locked formulation**.

## Dead route: TopK K-value reduction (shrink the [3,7,6] candidate block)
Hypothesis: `topk_targets=3 / topk_hidden=5 / topk_visible=6` might be over-provisioned; reducing them shrinks the whole `[3,7,K]`/`[7,K]` candidate-match block (~400 B). **KILLED by direct measurement** — instrumented LIVE (expose the three TopK-input bool tensors as outputs, correct 1× encoding `x[0,color,r,c]=1`, all 266 train+test+arc-gen cases):
- `target_red` (→topk_targets): dist `{2:129, 3:137}`, **max 3 = K** → TIGHT.
- `source_hidden` (→topk_hidden): dist `{2:49,3:99,4:83,5:35}`, **max 5 = K** → TIGHT.
- `source_visible` (→topk_visible): dist `{5:2, 6:263, 7:1}`, **max 7 > K=6** → K is already UNDER max (the lone 7-cell case is truncated — that IS part of LIVE's 1.95% miss). Cannot reduce.

Any K reduction drops cells on the max-count cases → diverges from LIVE → fails AB byte-equal. **Do not retry K reduction.**

## Dead route: transform-count reduction (7 → fewer)
LIVE uses 7 dihedral transforms (the 8-group minus the never-occurring main-diagonal transpose; Sho already dropped the 8th). Measured `selected_pos` distribution across all targets/266 cases: `{0:285, 1:103, 2:82, 3:87, 4:76, 5:86, 6:79}` — **all 7 are selected with material frequency**. Dropping any breaks byte-equality on its cases. **7 is forced.**

## Forced-component / dtype floor (why 12,030 is the byte-equal floor)
Output `[1,10,30,30]=9,000` is grader-EXEMPT. Every remaining charged tensor is forced:
| tensor | bytes | why forced |
|---|---|---|
| `board_code_f` F32 [1,1,15,15] | 900 | Conv on an F32 one-hot must output F32 (dilation-15 crop-collapse trick = color+1); quantizing the [1,10,30,30] input to feed a QLinearConv encoder = 9,000. |
| 3× TopK feeds F16 [225] | 1,350 | ORT TopK has no INT8/U8 kernel; INT32=900 ea (2× worse). F16 is the floor. |
| `candidate_safe_indices` INT32 [3,7,6] | 504 | Gather index must be INT32/INT64; no INT16 kernel. Shape [3,7,6] forced (see above). |
| `cm2ch` U8 480 + `cm16_sq` 240 | 720 | parabola decode `−(code−c)²+1` needs the [code, code²] quadratic channel. |
| ~21 U8/BOOL 225-masks | ~4,725 | channel Equal masks + 8-tensor geodesic MaxPool flood; minimal dtype (1 B); count = algorithm working set, byte-equal-locked (3-step geodesic ≠ single big MaxPool). |
| 14 U8/BOOL [3,7,6] candidate | 1,764 | O(targets×transforms×body-cells) validation; op-granularity forced (no ONNX clip-shift-mul fusion). |

All F16 coordinate tensors were already narrowed to INT8/INT32 (the −509 win). **DUAL-TRUTH gives no relief**: a simpler graph can't strictly dominate LIVE's 98.05% (it would miss cases LIVE gets right → `ab_divs ≠ LIVE_misses − cand_misses`), and a more-correct graph (toward the 8-transform reference) is strictly MORE expensive. So byte-equal narrowing is the only cost lever, and it is exhausted.

## One remaining hypothesis (low confidence, next batch)
The 8-tensor geodesic flood (1,800 B) and the 6-tensor channel-mask block (1,350 B) are the largest byte-equal-locked buckets. If a future agent finds an ONNX assembly that produces the IDENTICAL `source_component` + masks with fewer materialized intermediates (e.g., a single fused op that ORT charges once), that is the only remaining sub-12,030 path — but every attempt so far to fuse/reduce flood steps changes geodesic behavior and diverges. Treat as WALL unless a byte-equal fused-flood is proven on ≥2000 AB.
