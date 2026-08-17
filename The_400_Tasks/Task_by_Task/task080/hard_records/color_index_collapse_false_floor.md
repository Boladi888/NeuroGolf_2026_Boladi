> **2026-07-12 UPDATE (bo_ng_cld_worker3):** The *front-end-collapse* route below is still DEAD (pixel-plane/Gather-LUT all lose). BUT the "WALL at 5605" conclusion was PARTIALLY overturned by executing THIS record's own Pass-1 "remaining hypothesis" (shift-bank / detect-block node reduction, predicted ~300-800 B). Result: **5605 → 4965 (−640)**, byte-equal (ab 2231/2231, truth 2000/2000). The template read (ArgMax→8-nbr-GatherND→shift-bank) was NOT forced — a flat `Gather(cell_flat, fc+offsets)` + center-zero is byte-equal and far cheaper; the cross-correlation kernel = rotate180(3×3). The `cost_floor_reference.md:265` "info-flow floor ≈5,050" is thus NOT a true floor (it billed the reducible template-read as forced). See task080.md attempt 2026-07-12 + `staged_candidates/task080_ColorIndexCanvas_cost4965/`.

# task080 — DEAD ROUTE: "color-index collapse" false-floor attack (2026-07-10)

**Worker:** bo_ng_cld_worker1 (Claude lane) · **Verdict: WALL / DROP**

## Live bar (re-measured, authoritative)
- **live_cost = 5605** (`tools/ng_current_bar.py --task 80` → manifest, owner Sho, 2026-07-09).
- live sha256 = `9cd05340d174845476a93bda5d9eb3283f3d4c41e853e00d4a5318331683499f` (`Team_Best_Onnx/task080.onnx`, 11025 B).
- `score_v2` on live = **5605** (memory 4306 + params 1299). Confirmed.
- **profile == static**: profiled over 231 fittable real inputs (train+test+arc-gen, skipping the 35 arc-gen 31×31 that don't fit the fixed `[1,10,30,30]` input) → profile_cost 5605, gap **0**. Fully static graph, no dynamic inflation, no credit-probe risk.
- NOTE: local staged `task080_pruned_deadcode_cost5649` (sha `ecfe506a`) is **worse than live** (5649 > 5605) — a superseded old best, not the bar.

## Prompt premise (the "false floor")
Prompt Direction: *"color-index collapse sweep: perform spatial logic on one index channel, use Gather/Equal only at the boundary. Delete charged carrier/canvas/template."* Target floor_est ~900. Fast probe: *"replace the per-color path with ArgMax/index-plane prototype and price it."*

## Why it's DEAD — the premise is already satisfied, and the literal fast-probe prices STRICTLY WORSE

### 1. The rule (decoded from officials)
Periodic lattice: equal cells (content 3/4/5 px) separated by 1-px walls (color 3 or 8). A **template motif** appears once — a center-color cell surrounded by a cardinal cross of an arm-color (TRAIN0: `6` center ringed by `3`s at cell (1,1)). Isolated **seed** cells (lone center-color, e.g. `6` at cell (4,3)) mark where to **stamp** the same cross motif. Output = stamp the plus at every seed, upsampled to pixel res. → motif-detect-and-stamp on a periodic lattice ("Filling Regions").

### 2. The live 5605 graph ALREADY works on one index channel, at CELL resolution
Per-tensor dump of live (66 nodes): **zero `[1,10,*,*]` per-color intermediates exist.** Everything is single-channel cell-space `[1,1,10,10]` (100 B) or scalars, plus the forced output tail. The one-hot input is collapsed to cell-space color values *inside* a fused Einsum. The "per-color spatial manipulation" the prompt targets **does not exist here** — it was already eliminated. There is no `[1,10,30,30]` carrier/canvas to delete.

### 3. The charged "template" (safe_name_38, 900 params) is a TRUE floor
`safe_name_68 = Einsum('bchw,pxyrh,pxysw,pmno,oc->bors', input, A, A, W, C)` where `A=safe_name_38 [3,1,1,10,30]` is a **double spatial-pooling matrix** (3 period-classes × 10 cells × 30 pixels) that fuses channel-collapse + periodic-cell pooling into one op, emitting `[1,1,10,10]` F32 (400 B) with **no large intermediate**. Its 900 dense params are the cheapest realization:
- **Sparsify A (nnz≈90)?** DEAD. `calculate_params` counts sparse-init params as nnz, BUT: sanitize_model doesn't rename sparse names, strict shape-infer fails pre-scoring, and value_info billing makes it dense anyway. traps §8e + task204 + task387. "Dense init is the floor."

### 4. The two Direction-literal alternatives price STRICTLY WORSE (measured, `score_v2`)
| Formulation | front-end cost | vs cell-space Einsum |
|---|---|---|
| **AltA** collapse to pixel-res index plane `[1,1,30,30]` (`Einsum(input,C)`), Direction's literal "one index channel" | **3620** (3600 B for the single plane + 20 p) | **9× worse per plane**, before any detect/stamp/upsample |
| **AltB** Gather-LUT cell sampler (`Gather(input,rows)→[1,10,10,30]`) | **73100** | catastrophic |
| **REF** current cell-space Einsum `[1,1,10,10]` | 400 mem + 900 p | the floor realization |
Working at *pixel* resolution (the literal ArgMax/index-plane probe) costs 3600 B for ONE tensor — 65% of the whole 5605 budget — whereas the current graph's entire 14-plane cell-space representation is ~1400 B. A third formulation (direct pixel-space conv-stamp, no cell grid) is also strictly worse: every plane is `[1,1,30,30]` (900 B, 9× cell-space) AND the dynamic period (3/4/5) breaks static-attribute Conv/Pool stride. All three lose.

### 5. The ~900 floor_est is a naive lower bound
900 = only the forced output-resolution index tail (`safe_name_123 [1,1,30,30]` U8). It ignores the forced: cell extraction (Einsum 400 + 900 p), motif-detect + stamp (~14 × `[1,1,10,10]` + 2 QLinearConv + shift-bank ≈ 1550), and separable upsample (`[1,1,30,12]` 360 + pads 265 + tables ~210 ≈ 835). **True algorithmic floor ≈ 4000+**; live 5605 is within ~1.4× of it, and the remaining slack is node-level micro-tuning the owner (Sho) has already pursued (5664→5649→5605) — a rung-ε shave, explicitly out of scope for this false-floor task.

## One remaining hypothesis (low confidence, for a future batch)
The only untested slack is a modest **node-count reduction in the cell-space detect/stamp block** (14 `[1,1,10,10]` planes @ 100 B) — e.g. fusing the two ReduceMax+Equal center/seed detectors or dropping the 8-kernel shift-bank to 4 cardinal kernels (72→36 p). Upside is bounded (~300–800 B, does NOT reach the ~900 prize), risks correctness on the finely-tuned owner graph, and is a rung-ε shave — not a floor route. **Recommend DROP** unless the team re-scopes task080 to "shave below 5605" rather than "collapse to ~900".

## Gates run
- `ng_current_bar.py --task 80` → live 5605, sha 9cd05340.
- `score_v2(Team_Best_Onnx/task080.onnx)` = 5605 (4306+1299).
- Static profile over 231 fittable real inputs = 5605 (gap 0).
- `ng_gate visible --task 80 --cand Team_Best_Onnx/task080.onnx` → **231/231 (100.00%) PASS** (incumbent = cheapest correct formulation found).
- `score_v2` on AltA/AltB/REF fragments (see table). No truth/ab/battery run on a candidate — no candidate beats live, so heavier gates would be pure burn (gate ladder); full truth/battery on the shipped incumbent adds no decision value.

---

## PASS 2 (2026-07-10, forced second pass) — WALL confirmed, decisive floor number found

### Rule corrected & verified across ALL 4 officials (train0-2 + test0)
Decoded each pair into cell-space. The context-pack "spread to whole row AND column" is **WRONG**. Actual rule: a single **fully-formed 3×3-cell template motif** exists (train0 = plus of `3` around `6`; train2 = 3×3 ring of `4` around `2`; test0 = corners-`8`/edges-`3`/center-`6`); **isolated seed cells** (lone cells of the template's center color) are found; the exact 3×3 pattern is **stamped at each seed** (clipped at grid edges). This is a *data-dependent template-stamp* — inherently the multi-phase pipeline the incumbent implements.

### DECISIVE: team's own documented info-flow floor ≈ 5,050
`Onnx_Advanced/cost_floor_reference.md:265` — `task080 | 49,405 → 5,664 (... info-flow floor ≈5,050)`. Live 5,605 is **~11% above the documented floor**, and the owner (Sho) already squeezed 5,664→5,605. The prompt's ~900 is the naive output-tail-only estimate. Class precedents confirm hard few-kB floors: task209 "sub-2,000 not reachable; READ plane+matcher irreducible", task064 "900 B BOOL is the floor", task054 "min one plane ≥4,520".

### Per-phase forced-component floor (why sub-1,000 is impossible)
| phase | cheapest realization | ~cost | why it's forced |
|---|---|---|---|
| output tail | `[1,1,30,30]` U8 → Equal→one-hot (free output) | 900 | 30×30 resolution; task064 precedent |
| cell sampler | fused Einsum + `[3,10,30]` dense template | 400 + 900 p | sparse-init dead (§8e); pixel-plane=3620, gather=73100 (priced) |
| template extract | QLinearConv neighbor-count + ArgMax + GatherND | ~600 | data-dependent 3×3 read |
| data-dep stamp | QLinearConv (extracted 3×3 kernel) | ~400 | variable placements = Conv/render floor (lib #132) |
| wall-aware upsample | pad-based separable 2× Gather | ~1100 | avoids a 2nd 900 B wall-mask plane |
| period detect | ReduceSum→Sqrt→Where ladder (H→period) + Einsum fallback | ~200 | 3 periods × wall {3,8} |
Σ ≈ 4,500 forced; team info-flow floor ≈ 5,050; live 5,605. **The ~900 target requires the output tail to be the ONLY charged tensor — impossible given forced cell-extraction, 3×3-template read, data-dependent stamp, and wall-aware upsample.**

### Pass-2 formulations tried & priced (all lose)
- **Upsample-index LUT swap** (replace the dynamic `Div/Mod/Where` `[1,1,30,1]` f16 index chain, ~360 B, with a static Gather-LUT). Traced: `idx[r] = 11 if r≥H else 10 if (r+1)%p==0 else r//p` — depends on **both period AND H**. A period-only `[3,30]` LUT is wrong across arc-gen H∈{24,26,27,29,31}; a full (period,H) LUT needs `[32,30]`=**960 p > 360 B saved**. Net-negative.
- **DepthToSpace / strided-Conv upsample**: blocksize/stride are STATIC attributes; period is dynamic (3/4/5) → cannot encode. Dead (same class as task080's Conv-stride limit).
- **Wall-mask upsample** (render walls via a `[1,1,30,30]` BOOL overlay instead of pads): adds a 2nd 900 B plane — strictly worse than the incumbent's pad trick.
- (Pass-1) pixel-res index plane 3620, Gather-LUT sampler 73100, dynamic-Slice sampler 4000-B intermediate — all worse.

### Verdict: **WALL**. Cheapest correct formulation found = the incumbent (5,605, profile==static, visible 231/231). No genuinely-different assembly reaches below it; the ≤~555 B gap to the info-flow floor is an unverified estimate representing rung-ε micro-optimization the owner has already mostly captured — not the false-floor ~900 route. **DROP.**
