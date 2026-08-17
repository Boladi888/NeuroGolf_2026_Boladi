# marker Einsum-crop — forced-floor WALL (batch 50, bo_ng_cld_worker1)

- **Live bar:** 327 (manifest, sha `b3a7297c064902e13b5bd86fc46aeb686ab4f1d476015a977193772c19a4a0be`, owner Vamshi).
  `ng_current_bar --task 121` authoritative_source=manifest. Effective local best =
  **326** (prior worker's staged `task121_markerrecolorcrop_cost326`, byte-equal to LIVE, −1 via GatherElements).
- **Prompt target:** ~150 (floor_est). **Verdict: REFUTED — true floor ≈ 310–326.**

## Rule (verified over all 266 train+test+arc-gen pairs)
Input = 13×13 active grid (F32 one-hot [1,10,30,30]) with several 3×3 objects.
Exactly one object holds a single marker cell of color **8 at its geometric center**
(count-1, center — 0 violations in 266). Output = that object's 3×3 patch with
**only the center (marker) cell recolored to the object color k**; every other cell is
copied verbatim. `output == patch-with-8→k` held for all 266. Padded to [1,10,30,30]
(top-left). Object color k is unique (non-0, non-8) in every case.

## Why the ~150 estimate is wrong
The prompt floor counts only "marginal Einsum locate + 3×3 crop". It omits the two
forced reads the **recolor** rule requires, plus the output:

| forced tensor | why forced | bytes |
|---|---|---:|
| `small` U8[1,10,3,3] | output one-hot, padless is DEAD (see other hard_record) | 90 |
| `color_pixel` F32[1,10,1,1] | must read object color; input is F32 ⇒ 10-ch pixel = 40 | 40 |
| `bg_patch` F32[1,1,3,3] | must read the silhouette (which cells are bg) to reconstruct all 9 cells; input F32 ⇒ 36 | 36 |
| U8/bool casts of the above (`color_selector` 10 + `patch_bg` 9) | dtype narrowing for the Where | 19 |
| locator scalars (row_f/col_f/row_z/col_z …) | marginal Einsum outputs | ~16 |
| **params**: coord[30]+chan8[10]+e0[10]+pads[8]+slice consts | memory-optimal locator + Where + Pad | ~66 |

Forced memory ≈ 90+40+36+19+16 = **201**; + dynamic non-bg-finder machinery + slice
index arrays (~40–55) ⇒ mem ≈ 245–256, params ≈ 66 ⇒ **total ≈ 310–326**. LIVE 327 /
prior 326 already sit on this floor.

Key sub-proofs:
- **No fixed color-sample position is safe.** bg-frequency per patch cell over 266
  pairs: every non-center cell is background in 115–134 of 266 cases; the only always-non-bg
  cell is the center = the marker itself. So the object color must be found by a *dynamic*
  non-bg search (the LIVE top-row-first-non-bg offset). Fixed sampling ⇒ hidden 0.
- **Reading the full patch is strictly worse.** Any "avoid the 40B color pixel by reading
  [1,10,3,3] once" route pays 90B (F32 slice) + 90B (U8 copy) before `small` — see
  formulation 2 below (769). The LIVE bg(36)+color(40)=76 < 90 patch read.
- **The marginal Einsum locator is memory-optimal.** coord[30]+chan8[10]=40 params buys
  scalar (~8B) row/col. The ArgMax-marginal alternative costs 240B of [1,30] marginals;
  slicing the channel-8 plane costs 3600B. 30 params for coord is forced (Einsum contracts
  the length-30 axis) and is the correct trade.

## Formulations tried from scratch (built + priced)
| # | formulation | cost | vs 326 | verdict |
|---|---|---:|---:|---|
| 1 | marginal locator + bg-slice + **ArgMin** top-row offset + color-pixel + Where | **350** | +24 | worse — int32 top-row (12B) + int64 ArgMin outputs (16B) beat by LIVE's bool-gather chain |
| 2 | full-patch read + Einsum channel-index + ReduceMax + OneHot + center Where | **769** | +443 | dead — double-90 patch + reduce/onehot machinery. Correct 4/4. |
| — | LIVE family (marginal locator + bg + single color pixel + Where) | 327/326 | — | near-optimal; prior worker already squeezed to 326 byte-equal |

Both from-scratch graphs were functionally correct (train0 / train+test) but priced far
above the floor. Builders: `prototypes/build_lean_argmin.py`, `prototypes/build_channelfold.py`.

## Remaining hypothesis (thin)
The only sub-326 lever would be shaving the ~40B of slice-index arrays / offset machinery
by a byte or two (prior worker already took top_bg −3B → 326). No structural rewrite reaches
150; the recolor's forced color(40)+silhouette(36)+output(90) reads dominate. Do **not**
re-explore "drop the [1,1,13,13] F32 marker plane to ~150" — the LIVE graph already has NO
[1,1,13,13] plane (that description is stale; it uses a scalar marginal Einsum locator).

## Gate note
No truth/ab/battery run: both candidates priced above the effective best, so heavy gates
would be pure burn (WORKER_PROTOCOL gate hygiene). Functional correctness confirmed by
direct ORT replay on the visible/train set.
