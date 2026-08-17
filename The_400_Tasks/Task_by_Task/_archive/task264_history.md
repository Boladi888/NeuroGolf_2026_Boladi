# task264 — archived pre-siege history (moved 2026-06-08)

All superseded by the SIEGE rewrite in `task264.md` (LIVE 34,606 → siege_v8 4,842).
Kept for provenance only.

## Algorithm sketch (OLD live graph, opset-10 template-matcher)

_See live ONNX. 114 nodes / 38 inits / opset 10. Op mix: Mul:74,
Slice:24, ReduceSum:10, Gather:2, Sum:2, Sub:1, Pad:1._

Key initializers:
- v2 (1, 1, 9, 9) F32 all-ones — used in `Sub(v2, v224)` = `1 - reduce_sum`
- v3 (1, 10, 1, 1) F32 — color basis
- v4 (4,) INT64 [0,0,0,0] — Slice starts (origin slice)
- v7 (4,) INT64 [1,1,1,1] — Slice steps (default; redundant in opset 10+)
- v44, v57, v70, v79, v88, v97, v102, v107 (1,1,9,9) F32 — 8 sparse 9×9 templates (3-4 nonzeros each)

## Attempts (pre-siege, opset-10/12/13 incremental)

| Date | Candidate | Result | Cost | Notes |
|------|-----------|--------|------|-------|
| 2026-05-18 | worker-7 `t_dc8111c3` | onnx_opt_safe / with_fold on live 242020 | 0 byte change | 242020 | Graph at fixed point. |
| 2026-05-18 | worker-7 `t_dc8111c3` | v2(1,1,9,9 ones)→scalar[1.0] in Sub + drop v7 Slice steps | -84 cost | 241936 | 50/50 eq, 265/265 verify, adversarial 100%. Banked-stack-only (Kaggle-negative history). |
| 2026-05-19 | worker-7 GROUP-O | crop16: Slice input→[1,10,16,16] + outer Pad | STAGED -2628 | 221299 | `task264_crop16_t_a32bb099_on_5770_74` |
| 2026-05-19 | worker-7 GROUP-O | conv_mask: 24 Slice+72 Mul → Conv[8,2,3,3]+Greater+Cast (sum==9) | STAGED -38835 | 185092 | `task264_conv_mask_t_0f8e7d52_on_5770_74` |

## Compression chain (2026-05-20 / 05-29) — the path to the LIVE 34,606

- 2026-05-20 [OB_HUGE]: f16_whole_graph_with_f16_output −73,296; 265/265 → CANONICAL 111,787.
- 2026-05-29 [worker-7]: compress_v1 opset-12 (Cast-after-Slice; Einsum eliminates 8×masked_mk* −31,360; Einsum template-fusion eliminates 8×scaled_mk* −12,800). 111,787→60,210.
- compress_v2 (padded-mask Einsum eliminates 6×icrop_cXX). →39,239.
- compress_v3 opset-13 (BOOL-Pad masks + Einsum-vstacked). →36,634.
- compress_v4 (Where-fill background). →34,933.
- compress_v5 BEST (A-G+H+I). →**34,606** (mem 33,627 + params 979). Builder `prototypes/build_task264_compress_v5.py`. Worker note (now disproven by the siege rewrite): "This IS the structural compression floor for this algorithm. 30k target requires different algorithm." — It was the floor of the *template-matching* algorithm, not the task; decoding the generator showed the task is just an 8-colour read.

## Pre-siege cost analysis (stale)

Live cost 242020 (memory 241168 + params 852); memory dominated (99.6%) by a (1,8,30,30)
Gather and ~20 (1,10,14,14) F32 crops. The 8 sparse (1,1,9,9) templates were the 2nd-largest init bucket.
