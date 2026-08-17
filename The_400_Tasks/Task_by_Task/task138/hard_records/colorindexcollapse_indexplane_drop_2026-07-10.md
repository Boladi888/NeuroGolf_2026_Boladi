# task138 — DROP: ColorIndexCollapse / "spatial logic on one index channel" route

**Recorded:** 2026-07-10 by `bo_ng_cld_worker1` (Claude lane), task138 only.
**Verdict:** DROP (false-floor). The charged component the Direction targets is already deleted in the
live team-best; the alternative it proposes is structurally *more* expensive than the live family.

## Live bar at time of record (REFRESHED — this is the key change vs prior records)
`python tools/ng_current_bar.py --task 138 --json`:
- Authoritative manifest live **4311**, owner **Sho**, sha256
  `72ffa3463b99f3a41b11a2d1ee431771769274fa99ec94a416ddfc022b3520a9`, updated 2026-07-09T01:09:17Z.
- Fresh `profile_real_cost --arcgen 266` of `Team_Best_Onnx/task138.onnx`: **static==profile==4311
  = memory 2830 + params 1481**, clean. (`team_best` source row shows 4292 for the same sha — stale
  cost cache; manifest 4311 is authoritative.)
- **This supersedes the entire Vamshi-5403 / staged-5283/5278/5276/5275 lineage** and its hard records
  (`below_5283_freeoutput_narrowing_wall.md`, `rank4_shared_basis_probe.md`,
  `marker_counts_direct_unpriced.md`) — all of those are > 4311.

## What Sho's 4311 family is (why the Direction is obsolete)
Same free-output separable-Einsum family as Vamshi 5403, further collapsed. Verified by
`inspect_graph.py` + a full `shape_inference` intermediate dump (scratchpad `dump_shapes.py`):
- Per-color collapse is **already done**: interior color counts `cntin` → `Mul K10` → `ArgMax am` →
  `OneHot sOH` = the single marker color (nodes 18-23). The "replace the per-color path with ArgMax"
  fast-probe lever is already pulled.
- Frame bounds: count-Einsums `rcnt/ccnt` → `ArgMax` r1/r2/c1/c2 (nodes 0-5).
- Ray direction: `Gather` marker profile at the 4 borders + `Greater` (nodes 26-33).
- Render: low-rank positional basis `rxLO30[30,6]`, `rxHI30[30,5]` + shape tensors `SHA[6,6,6,2]=432`,
  `SHB[5,5,5,2]=250`, `CLC[2,2,6,6]=144`, `CHC[2,2,5,5]=100` → **one final free-output Einsum** (node
  65) straight to `output[1,10,30,30]`.
- **ZERO 2D-spatial intermediates.** Largest intermediate = `Ac[6,6,2]` = 288 B. Every spatial tensor is
  a 1D profile (`scr/scc/rcnt/ccnt/mR/mC` = `[30]`, `inRf/inCf` = `[1,30]`). The `[1,10,30,30]` output
  (36000 B) is the graph output = free. There is **no per-color carrier, no canvas, no index plane**.

## Route attempted (fast probe) and the priced wall
Direction: "perform spatial logic on **one index channel**" ⇒ materialize a single `[1,30,30]` index
plane and do the ray-fill on it, one-hot only at the boundary.

Priced the forced first tensor of that family — `prototypes/task138_indexplane_floorprobe.onnx`
(builder `build_task138_indexplane_floorprobe.py`): `Einsum('bcyx,c->byx', input, [0..9])` → `idx[1,30,30]`.
```
largest_intermediate task138_indexplane_floorprobe.onnx:
  36000 F32 [1,10,30,30]  OneHot  oh    <- graph output (free)
   3600 F32 [1,30,30]     Einsum  idx   <- THE forced index plane
   3600 I32 [1,30,30]     Cast    idx_i <- int copy required to one-hot-expand
```
The single forced index plane is **3600 B (F32)** — **> Sho's entire 2830 B memory budget** and **4× the
900 target** — before adding frame-detect, crop, ray-fill, or one-hot expansion. A U8 plane (900 B)
cannot be arithmetic-filled or used as one-hot indices without casting back to F16/F32 (re-inflating to
1800/3600) plus a fill-matrix (~900 params). **Fast-probe kill rule fires: prices ≫ 900.**

## Why this route is dead
The Direction was written against an older, more expensive family. Its premise — that a charged
per-color carrier/canvas/template still exists to delete — is false for the live 4311 build, which
already collapses color to one index and renders with **zero 2D intermediates**. "Spatial logic on one
index channel" is a *step backward*: it re-introduces the 2D plane the winning family eliminated. The
`~900` floor_est is the class-SWEEP heuristic (4311 ÷ 4.79), not a proven construction; there is no known
index-plane build that even beats 4311, let alone reaches 900. **DROP.**

## One remaining hypothesis (out of scope for this recapture prompt; not pursued)
The only sub-4311 lever is *within* Sho's free-output family: rank-reduce the shape basis
(`SHA` rank-6 = 432, `SHB` rank-5 = 250, the two largest params). But `rank4_shared_basis_probe.md`
already showed the row/col bases are rank-5 on 4702/5000 fresh inputs, and `SHB` is already rank-5;
dropping `SHA` below rank-6 or `SHB` below rank-5 changes the decoded ray shape (falsifies). That is an
algorithmic rewrite, not a color-index collapse, and the 2026-07-10 workflow update de-scopes
rung-ε shaves for this false-floor prompt. Any real attempt must re-derive the shape function and pass
full 200k truth + a raw nan/inf scan.
