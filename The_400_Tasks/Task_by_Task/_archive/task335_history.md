# task335 — archived cost-analysis ladder (pre-floor history)

Moved from `Task_by_Task/task335.md` 2026-06-20 (worker-335-1) to keep the live
file under the task-md-size budget. The Attempts table in task335.md still has
one row per version; this file holds the verbose per-version cost prose. The
CURRENT best (v19 @ 1346) analysis and the FLOOR PROOF remain in task335.md.

Live cost 60245 (memory=60136, params=109). Intermediates dominated by input_inner [1,10,20,20]=16000 bytes and output_inner [1,10,20,20]=16000 bytes, plus ~28 [1,1,20,20] tensors at 1600 bytes each. The Sub×Sub+Mul+Less pattern (nodes 12-15, 22-25) is NOT a Sub→Mul(self)→Less Equal-fold; it checks (a-x)*(b-x)<0.5 (between-two-points range), not distance².

2026-06-01 v1 rewrite cost 5206 (memory=5140, params=66). Key savings: avoid the full `[1,10,20,20]` F32 crop, avoid F32 `stacked`/Conv/output_inner, derive marker coordinates from `[1,10,30,1]` and `[1,10,1,30]` row/column summaries, and overlay the padded BOOL path onto original input via final `Where`.

2026-06-01 v2 rewrite cost 2536 (memory=2428, params=108). It replaces v1's row/column summary tensors with two `[1,2]` `Einsum` coordinate projections. Remaining dominant memory is the required padded path condition `path30` (900 bytes) plus `vert`, `horz`, and `path` BOOL tensors (400 bytes each); remaining params are mostly `idx30` (30), `row_idx` (20), `col_idx` (20), marker selector (20), `color4` (10), and `pad_path` (8).

2026-06-01 v8 rewrite cost 2428 (memory=2324, params=104). Remaining dominant memory is still `path30` (900 bytes) plus `vert`, `horz`, and `path` (400 bytes each). The sub-100B ladder after v2 was: direct scalar coords v3 saves 16 memory; rank-2 masks v5 save 4 params; UINT8 coordinate comparisons v6 save 8 memory; `Clip` range tests v7 save 40 memory; `Xor(range, endpoint_eq)` legs v8 save 40 memory.

2026-06-01 v9 rewrite cost 2353 (memory=2248, params=105). It leans on the observed generator margin: in official pairs and 20,000 ARC-GEN pairs, marker coordinates are always in `[1,17]` and markers are never same-row/same-col. That makes U8 `+1`/`-1` endpoint-adjusted ranges safe for the task distribution. Remaining dominant memory: `path30` 900 bytes, `vert`/`horz`/`path` 400 bytes each, then only six 20-byte row/column masks.

2026-06-01 v14 rewrite cost 1351 (memory=1228, params=123). It removes the 20x20 path plus 30x30 Pad pattern entirely: row/column masks are length 30, and the broadcast comparison directly produces the final `[30,30]` path condition (900 bytes). The score-union ladder after v9 was: v10 replaces `vert`/`horz`/`path` with `row_score + col_score >= 3` (2074); v12 rewrites that as `row_score >= 3-col_score`, dropping the 400-byte `path_score` (1655); v13 computes the path at full 30x30 and removes Pad (1381); v14 creates `col_need` directly (1351).
