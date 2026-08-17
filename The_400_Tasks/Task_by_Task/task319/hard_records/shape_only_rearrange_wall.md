# task319 — ShapeOnlyRearrange card: PRICED WALL (2026-07-12, bo_ng_cld_worker2, Claude lane)

## Bar (live, reconciled)
- **team-best = 5833** (cost = memory 5674 + params 159). Source: manifest, owner **Vamshi**, updated 2026-07-12T15:15:54Z.
- sha256 `5df0de2e9e52ba143be115613efaa645621ffc6664611985e40f136c0a5fba41` (== local `Team_Best_Onnx/task319.onnx`, == submission.zip).
- Verified: current best **profile cost = static = 5833** (`profile_real_cost.py --arcgen 266`), and **visible 267/267 PASS** — it is a legit 100% *structural* solve, not a probe/memorized rung.
- No `Staged_Boladi/task319.json` sidecar. Prompt/context_pack showed STALE local state (16716 / v36 12714, the old Conv color-index lineage); the live bar is the newer **bit-pack** lineage (Vamshi/kmmmvs), not in this repo's prototypes.

## Prompt premise vs reality
Prompt: "shape-only rearrange/crop rewrite, floor_est ~1500, EV +22 pts". **The floor_est is fantasy.** The card was triggered by corpus op-hits (slice/gather/pad/reshape) but the rule is NOT a rearrangement — it requires an irreducible **2-D bitmap witness** (per-cell decision), which the prompt's own kill-rule says to stop on.

## Rule (confirmed 267/267 via `prototypes/task319_numpy_rule_v1.py`)
Input `[1,10,30,30]` one-hot, always exactly 4 colors: bg (uniform) + 3 shapes.
S = the shape whose 2×-upscale (clipped at grid edge, any edge) contains another shape B as a sub-rectangle; tie → smallest bbox area then h+w. Output = S's bbox bitmap, S.color where set, bg where hole, placed **top-left** in `[1,10,30,30]` BOOL. Output tensor is FREE (excluded from cost by `calculate_memory`).

## Why the witness is IRREDUCIBLE (all cheap proxies falsified on all 267)
Two shapes can share size/cells while only one is the true upscale source, so no scalar proxy separates them:
| proxy for "which shape is S" | hit rate |
|---|---|
| smallest bbox area | 119/267 |
| smallest cell count | 137/267 |
| dim-match (∃j: hⱼ∈{2hᵢ-1,2hᵢ}, wⱼ∈{2wᵢ-1,2wᵢ}) | 7/267 (B is clipped by an arbitrary amount, not just ±1) |
| cell-ratio (∃j: cellsⱼ∈[2·cellsᵢ,4·cellsᵢ]) | 174/267 |
| area-ratio [2×,4×] | 164/267 |
| dims≤2× & area≥2× (best proxy) | 196/267 |
| positional (topmost/leftmost/corner) | 66–82/267 |
| B = argmax cells | 250/267 (not deterministic) |
| downscale2(B) matches top-left of S | 127/267 (clip phase ambiguous — B may be clipped at top/left) |

⇒ the selection needs the full `contains_subrect(upscale2(Sᵢ), Bⱼ)` 2-D template match. This is exactly the prompt's kill condition ("correctness needs per-cell decisions rather than rearrangement").

## Cost model + current-best breakdown (cost = params[init elem count] + Σ intermediate bytes; input/output FREE)
- carrier1 `Einsum(input, [2^0..2^18,0..], "bchw,w->ch")` → `[10,30]` F32 = **1200 B** (per-(color,row) column bitmask; exact in F32 since active grid ≤19×19 ⇒ max mask 2^19-1 < 2^24, verified max occ row/col = 18).
- carrier2 `Einsum(input, "bchw->cw")` → `[10,30]` F32 = **1200 B** (per-(color,col) row-count). Double-duty: bg via sum→argmax AND column-bbox extent via argmax-of-presence.
- rest ≈ **3274 B**: 6 gathers `[30]`F32 (720), bit-unpack `[5,5]`U32 ×4 (400), role-finding ArgMax chains, bboxes, bit-space witness (BitShift/BitwiseAnd), render `[10,5,5]`BOOL (250). Tight, no identified fat.

## Floor proof (realistic floor ≈ 5500–5833, NOT 1500)
1. **carrier1 (1200) forced** — need per-row column patterns to reconstruct S's bitmap and to run the witness. Channel-reduction to the 4 present colors does NOT help: gathering input channels materializes `[1,4,30,30]`F32 (14 400 B); gathering after the Einsum still pays the `[10,30]` (1200) first.
2. **column-offset (c0) forced, and carrier2 is its cheapest source.** The render must shift S's bitmap to column 0 (needs c0=leftmost occupied col) and the bit-space witness needs alignment. carrier2 gives c0 in 1200 B while KEEPING the witness in cheap bit-space (BitShift on packed `[·]` scalars).
3. **Single-carrier LOSES.** Dropping carrier2 forces recovering c0 either (a) via an equal-cost `[10,30]` col-presence reduction (no saving), or (b) via unpacked bitmaps — which then forces the *witness* into unpacked space: `upscale2(bmA)` = `[3,20,20]`BOOL = 1200 B alone, before match scores. That exceeds the 1200 saved. Net ≤ 0.
4. The witness itself is irreducible (§ above). The current best realizes carriers+witness+render at 5833 with no identified slack.

⇒ No formulation I can construct prices below 5833. The two-carrier bit-pack structure is at/near the structural floor.

## Verdict: **WALL / DROP** (status REJECT_LOCAL — no-gain)
No candidate built. Building would either reproduce the current best's bit-space witness (multi-hour, ~break-even by the carrier accounting) or use a simpler-but-costlier scheme (the v34 Conv color-index lineage is 12 714–16 716, 2–3× worse). Neither can beat 5833; a build is grind, not a win. Marginal LB upside even if a ~5% trim existed (points = 25−ln(cost): 5833→16.33, a hypothetical 5000→16.48, +0.15 pt).

## PASS 2 (2026-07-12, forced second pass) — EMPIRICAL build evidence

Built and validated two from-scratch front-ends against the numpy reference (ORT, all 267 pairs), then measured with `_score_v2`/`calculate_memory`:

1. **Single-carrier front-end** (drop carrier2, derive column-offset c0 by unpacking the row-bitmask to `[3,10,19]` bits): **267/267 correct, cost 16 639.** Refuted empirically — the killers are exactly what carrier2 avoids: the `[3,10,19]` unpack materialized in 3 dtypes (6 840 B) + INT64 `col_idx` gather tensors (4 800 B). Deriving c0 without carrier2 is FAR more expensive than carrier2 itself. Single-carrier is dead.

2. **Lean two-carrier front-end** (carrier1 rowmask + carrier2 colcount; bboxes via Greater→ArgMax; row-align via GatherElements; **column-align in bit-space via BitShift** — no big unpack; INT32/UINT32 small dtypes): **267/267 correct on the extracted column-aligned bitmasks + bg + h/w, cost 5 268** (carriers 2 400 + gathers 720 + bbox occupancy/indices ~1 788 + align ~360; params 48).

**The decisive number:** the cheapest correct front-end I can build (carriers + bboxes + aligned bitmaps, nothing else) is **5 268** — already 90 % of the live best's *entire* 5 833. That leaves **565 B** for the 2-D witness + one-hot render, which the live best spends **~2 000 B** on. Even a theoretically-optimal from-scratch front-end (~3 880 B after trimming the duplicate `[3,30]` occupancy/index tensors my build incurs) + a minimal witness+render (~2 000 B, and the live best's is tighter than anything I assembled) ≈ **~5 880 B ≈ break-even at best, realistically above 5 833**. Every component I built ran 2–3× the live best's bit-space equivalent.

⇒ **Empirically confirmed WALL.** A from-scratch rebuild cannot beat the live 5 833. `ng_graph_cost_diff` N/A (no winning candidate to diff). No candidate staged.

## One remaining hypothesis (low confidence, for a future batch)
The ONLY untested lever is trimming the current best's *role-finding* (triple ArgMax+Where to peel the 3 shapes, 6 gathers) inside the SAME two-carrier structure. If a from-scratch two-carrier build could realize role-finding+bbox in <3274 B for the non-carrier part while keeping the exact bit-space witness, a small (<~10%) win might exist — but the current best shows no obvious redundancy and this repo has no builder for that lineage to diff against, so it is high-risk/low-EV. Do NOT re-run the falsified cheap-proxy families above.
