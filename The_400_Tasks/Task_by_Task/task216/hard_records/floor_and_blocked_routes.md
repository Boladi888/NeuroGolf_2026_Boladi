# task216 — floor proof + blocked routes (bo_ng_cld_worker2, 2026-07-12)

**Live bar:** 4895 (profile), Team_Best `182a6d2f…`, owner Sho, manifest 2026-07-10.
**Current frontier (2026-07-12 second pass):** `staged_candidates/task216_pack20_u8range_cost4334/` — profile **4334** (−561 B, −11.5%), byte-equal (ab 2266/2266 vs LIVE), truth 2000/2000, visible 266/266, LOCAL_GOLD. sha `5146e1a2…`.
  - **SUPERSEDES the debruijn 4744** below. The 4744 "practical minimum" verdict at the bottom of this file was WRONG — two more byte-equal routes (NOT in the blocked list) took it to 4334: **(A) [30]→[20] narrowing** of the packing/finding/residual vectors (grids are always 20×20, so cols/rows 20..29 are always background; count masks rif/cif MUST stay [30] because the 2-count Einsum reads the full [1,10,30,30] input) = −360-ish B; **(B) U8 unsigned-range count masks** — `ge+le+and` (3 BOOL[30]) collapses to `(iota_u8 − lo8) ≤ span8` (1 U8 Sub + 1 BOOL LessOrEqual) = the rest.
  - **Lesson: this task has now had THREE beaten floor-claims (13,995 → 4744 → and each "wall" gave way to a dtype/shape narrowing or a new sub-assembly). Do NOT treat any number here as a proven absolute floor; 4334 is the current frontier.**

## Rule (re-derived from data, prompt is MISLABELED)
Prompt calls this `ShapeOnlyRearrange` — **wrong**. Actual rule (visual-first over train pairs, confirmed by 266/266 + the shipped model): **output the bbox-crop of the rectangular object (solid block of 1s/2s) containing the most color-2 cells.** The slice/gather/pad "corpus hits" that triggered the ShapeOnly card are just the ops the bbox-crop uses, not the rule. There is NO pure shape-rearrange shortcut. The old task216.md "13,995 proven floor" is SUPERSEDED — Sho's bit-packed 4895 beat it long ago.

## Corpus facts (266 pairs, connected-components analysis)
- Components per grid: **always 3 or 4** (137×3, 129×4). Max 4, never more, never fewer.
- All components are **rectangular** (0 non-rect). No 2-count ties.
- Winner rank in scan-order(top,left): {0:86,1:74,2:78,3:28} → winner is the **4th-found in scan order 10.5%** of the time ⇒ **4 candidates are forced** (matches the old 15,173-pair analysis).
- Winner rank in size-order(area desc): {0:214,1:41,2:11,3:0} → winner never the smallest *in this sample*, BUT this is a statistical regularity (a small 2-dense rect can beat a large sparse one); relying on it to drop to 3 candidates = hidden-bad risk (score 0). NOT pursued.

## The shipped architecture (Sho's 4895, and my 4744 golf of it)
Bit-packed, plane-free (largest intermediate = 120 B; NO [30,30]/F32 plane is ever materialised — a single [1,1,30,30] F32 slice = 3600 B would already blow half the budget). Two Einsums pack rows/cols into 30-bit ints (`msk`,`mskT`); 4 greedy candidates k1–k3 (explicit find+count+residual) + k4 (bbox only, count = `Tred − (k1+k2+k3)` complement trick). ArgMax of 4 counts → gather winner bbox → output via one radix-2 shift-Einsum (E0–E4).

## Priced floor — why ~1500 is FANTASY and 4744 is near the ORT floor
Forced charged components (cannot be removed without a bigger plane or a wrong answer):
| component | bytes | why forced |
|---|---|---|
| setup bitpack msk+mskT + 2 F32 Einsum transients | ~480 | F32 input ⇒ Einsum outputs F32 ⇒ Cast to U32; both row+col packings needed |
| 3× residual (rbu Cast + ctr Mul + rem Sub, U32[30]) | ~1080 | greedy needs per-cell removal; rectangles share rows (side-by-side) so can't blank whole rows |
| 3× separable box 2-count masks (rif+cif F32[30]) | ~720 | 2D box count = rank-1 rif⊗cif; F32 to match F32 input; 4th count via complement |
| output crop masks mr/mc + E0–E4 radix params | ~665 | static plane-free shift needs radix permutation matrices |
| range BOOL masks + bit-decode scalars | ~1300 | ge/le/and per axis; log2/de-Bruijn bit-position extraction |
| **sum ≈** | **~4245+** | + profiling ⇒ **true floor ≈ 4400–4600** |
1500 is below even {residual 1080 + counts 720 + params 458 = 2258}. Definitively unreachable.

## BLOCKED byte-equal golf routes (do not re-attempt)
1. **Eliminate the 3× `rbu` Cast (−240 B at [20])** via `ctr = Where(rbo20, bm, z32)`: **BLOCKED — RE-CONFIRMED 2026-07-12** by `golf_task216_v5_where20.py`: ORT has NO `Where` kernel for UINT32 (`NOT_IMPLEMENTED: Where(16)` at profile/session-init). No integer-select exists for U32; every path back to "bm where rbo else 0" needs a Cast(rbo→U32) == rbu. INT32 residual (to use `Where(INT32)`) is ALSO blocked: de Bruijn needs unsigned `>>27` (arithmetic shift on a sign-bit-set product returns the wrong index), and a U32↔I32 boundary Cast on `bm` eats the saving. (dead prototypes: `golf_task216_v1.py` at [30], `golf_task216_v5_where20.py` at [20].)
2. **de Bruijn shrink further:** already applied (the shipped win). Gather rejects UINT32 indices, so each site needs an extra `Cast(U32→I32)` — static net is only ~−17 B; the real −151 comes from ALSO killing the parent's Log/Round runtime-inflation surface. That inflation is now gone, so it can't be harvested twice.
3. **Narrow count masks to F16:** BLOCKED — Einsum requires all inputs share the F32 input's dtype; going F16 forces a materialised F16 2-plane (1800 B) ⇒ net loss.
4. **Drop to 3 candidates:** hidden-bad (winner is 4th-in-scan 10.5%; size-order shortcut is statistical).
5. **Coarser output radix / Gather-shift:** any Gather/Slice crop materialises a [1,10,30,30] intermediate (36 kB) — catastrophic; radix-2 Einsum is already optimal.

## Verdict (updated 2026-07-12 second pass)
Current frontier = **4334** (byte-equal, LOCAL_GOLD), a STRICT WIN over the 4895 bar (−561, −11.5%). The below-4334 wall is now two-sided and empirically probed:
- **F32-Einsum marginals (1200 B: packing 240 + count 720 + output 240):** forced F32 because they read/feed the free F32 `input`/`output` via Einsum (homogeneous dtype). #109 caveat: free-output dissolves output *canvases*, NOT independent F32 bbox/extent *marginals*. F16 forces a materialised F16 plane (net loss).
- **U32[20] residual rbu (240 B):** forced by the ORT `Where(UINT32)` kernel gap (re-confirmed via golf_v5); INT32 breaks de Bruijn.
- **Remaining ~708 B scalars** (de Bruijn 12 sites + coordinate arith + 4× 5-shift parallel-prefix smear): the smear is forced (5 shifts cover a 20-bit run); intricate and high-risk to touch for sub-100 B, NOT pursued.

Do NOT target 1500 (below residual 720 + counts 720 + params 462 = 1902 alone). BUT — given three beaten floor-claims on this task — the honest verdict is "4334 is the current frontier at the ORT-constrained floor," not "proven absolute minimum." Next-batch angles only worth trying if a NEW primitive appears: a ORT UINT32/INT-select kernel (unblocks rbu), or an assembly that avoids one of the 3 explicit count Einsums (unlikely — corpus proves 4 objects, sharing both rows and cols, so 2D counting is genuinely required).
