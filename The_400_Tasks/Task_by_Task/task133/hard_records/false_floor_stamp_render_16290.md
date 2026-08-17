# task133 dead-route — "delete hash-MLP → ~900 floor" is ALREADY CRACKED; ~900 is infeasible

**Recorded:** 2026-07-10 by `bo_ng_cld_worker3` (Claude lane), batch for Prompt_2026-07-09T150809Z_5801_Task133_HashMlpRuleFloor.

## Live bar (refreshed this session)
- **live_cost = 16,290** | owner **Sho** | sha256 `535fec113fb4b6a599022b5f2a64f7db0898c85931688b1bde2085e41b41e7d6`
- source: `ng_current_bar.py --task 133` → authoritative = OnnxlyFans manifest, updated 2026-07-09T01:09:17Z, points 15.3017.
- graph file: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task133.onnx` (local copy: `prototypes/live_16290_sho.onnx`).
- `profile_real_cost --arcgen 266`: **static=16290 profile=16290** (memory 16,188 + params **102**; 267 real inputs). No GRADER-GAP.

## Why the prompt's Direction is stale
Direction = "hash-MLP / dense classifier decode: extract generator rule and replace learned/template graph with tiny determinist. Delete charged carrier/canvas/template." Target ~900 (floor_est assumed deleting the 47,234-param memorized hash table).

**Sho already did the entire rewrite.** The live 16,290 graph has NO hash-MLP, NO dense classifier, NO param bank — **params = 102 bytes total**. It is a deterministic bbox + template-stamp solver:
1. `Conv(input, W10[1,10,1,1])` → `gridf` F32 [1,1,30,30] color-code grid; Cast→`grid` U8.
2. `ReduceMax` row/col of F32 input → `rowany`/`colany`; `ArgMax` → per-color bbox corners.
3. Identify template shape (largest-area present color) + target sprites (other present colors); extract the creature pattern window.
4. `TopK(4)` selects ≤4 sprites; for each, compute Resize ROI+scale from its magnifier → **4× `Resize`** stamps `s0..s3` U8 [1,1,30,30]; `Max(grid,s0..s3)`→`outgrid`; `Equal(outgrid, arange)`→ BOOL [1,10,30,30] output.

The charged component named in the Direction (hash-MLP / 47k-param table) **is already gone**. Per workflow rule (2026-07-10): component gone ⇒ DROP.

## The task (generator `task_57aa92db.py`, confirmed by running it)
Sprite-completion / "Filling Regions": input has ONE full template creature (sprite 0, magnifier 1, pivot pixel = shared `pcolor`) + 2–3 PARTIAL sprites (only pivot + one body pixel shown, magnifier 1–4, own color). Output completes every partial sprite by stamping the template shape **scaled by its magnifier and recolored**. `num_sprites ∈ [2,4]`.

## Why ~900 is structurally impossible (the real floor note)
This is a **data-dependent, variable-scale, 2D template-stamp** task. All three free-output killers are present:
- **Arbitrary 2D shape** (creature, not a separable line/rect) → per lib #109 a 2-D region can't be rendered by separable `Equal`; forces a canvas + ≥1 render/Gather plane.
- **Runtime-decoded selectors** (which color is template, which are targets, each position/scale/color computed from input) → per lib #110 free-output fails regardless of rank.
- **Variable placement count + per-sprite scale** → per lib #132 this is a Conv/render floor, not a separable free-output floor.

Forced components of the live (bbox+stamp) formulation, all re-derived this session:
| tensor | dtype/shape | bytes | why forced |
|---|---|---:|---|
| `gridf` | F32 [1,1,30,30] | 3,600 | color collapse of F32 input; INPUT-DTYPE-LOCK. ArgMax-over-ch alt=7,200; cast-full-input=9,000. (same 3,600 the w05 21,394 proof forced) |
| `rowany`+`colany` | F32 [1,10,30,1]+[1,10,1,30] | 2,400 | per-color bbox occupancy from F32 input; ReduceMax preserves F32; narrowing input first ≥9,000 |
| `grid`+`gflat` | U8 [1,1,30,30]+[900] | 1,800 | U8 color canvas + flat for Gather |
| `s0..s3` | U8 [1,1,30,30]×4 | 3,600 | ≤4 sprites, each distinct (pos,scale∈1..4,color); one Resize = one box/one scale ⇒ 4 planes forced |
| `outgrid` | U8 [1,1,30,30] | 900 | Max-merge for final Equal expansion |
| subtotal forced | | **12,300** | |
| sprite-corner / template-match / TopK / ROI chains | INT64/F16/U8 small | ~3,850 | inherent detection stages, 80–160 B each |
| **≈ total** | | **≈16,150 ≈ live 16,290** | |

The ~900 floor_est conflated "tiny rule **params**" (achieved: 47,234 → 102) with "tiny **total** cost". Cost is now MEMORY-dominated by forced rendering intermediates. **Absolute floor ≥ ~7k** even in the most optimistic accounting (must read F32 input ≥3,600 + localize sprites ≥2,400 + ≥1 U8 canvas 900), realistically ~16k with the actual stamp mechanism.

## Second pass — MEASURED alternative assemblies (2026-07-10, forced review)
Built + `profile_real_cost --arcgen 266` (static=profile, no grader-gap) on genuinely-different assemblies for the two dominant stages. Both are strictly WORSE — Sho uses the optimal lever at every stage.

| stage / assembly | profile cost | verdict |
|---|---:|---|
| color-collapse — **Conv 1×1 F32** (Sho) `probe_colorgrid_conv` | **4,520** (gridf F32 3,600 + grid U8 900 + 20 p) | floor |
| color-collapse — **ArgMax index-plane** `probe_colorgrid_argmax` | **8,110** (idx INT64 7,200!) | +3,590 WORSE (ArgMax has no dtype attr → INT64; task064 trap, measured) |
| stamp — **index-remap Gather**, 1 sprite, fixed pivot, no detection `probe_stamp_indexremap` | **16,260** | DEAD: INT32 flat index plane = 3,600/sprite (values 0–899 > U8 255, traps §8c) = 4× a Resize plane (900); one INCORRECT sprite already ≈ full live |

Library confirms the optimum: **#117 "Resize with a DATA-DEPENDENT ROI to scale/place a constant template" (Aditya task021)** is exactly Sho's stamp; **#132** flags data-dependent variable placement as a Conv/render floor (not free-output). DepthToSpace (#80) has a fixed blocksize → cannot express per-sprite magnifiers 1–4; GridSample needs a `[1,30,30,2]` F32 grid = 7,200 (worse than 4×Resize 3,600). No different assembly beats the live.

## Independent formulations considered (all fail to reach a sub-16,290 structural win)
1. **Free-output separable Einsum (~900 dream)** — DEAD: arbitrary 2D shape + runtime selectors + variable scale (lib #109/#110/#132). No separable (r,c) formula exists.
2. **ScatterND sparse-paint (old family)** — DEAD & priced-worse: variable pixel count per sprite (bmag² per template pixel) ⇒ dynamic shape (scorer rejects symbolic dims) or static max-index bank; the prior sparse_paint/qroot builds of exactly this priced **21,394+**, worse than Sho's Resize stamp.
3. **Resize/render stamp** — this IS the live 16,290; forced floor ≈16k; only sub-16,290 moves are epsilon shaves (e.g. drop the 900-B `gflat` by restructuring one Gather), explicitly out of scope ("false-floor attack, not a rung−epsilon shave") and not verified visible-safe.

## One remaining hypothesis (thin)
Only conceivable non-epsilon lever: fuse the 4 stamp planes (`s0..s3`, 3,600 B) into fewer via a single strided/batched render — but Resize needs one uniform scale per call and the sprites have independent magnifiers (1–4), so batching is blocked; a unified static ScatterND already priced worse. If a future ORT feature allowed per-batch Resize scales, retest; until then this is a priced wall.

## Verdict
**DROP / WALL.** Named charged component already deleted by Sho (params 102). ~900 target infeasible (Conv/render floor). No candidate strictly below 16,290 by any non-epsilon route.
