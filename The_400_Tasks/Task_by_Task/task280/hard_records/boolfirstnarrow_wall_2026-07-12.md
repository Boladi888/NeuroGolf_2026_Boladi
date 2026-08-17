# task280 — BoolFirstNarrow WALL (priced) — 2026-07-12

**Worker:** bo_ng_cld_worker3 (Claude lane)

## Live bar
- **live_cost = 2618** (memory 1874 + params 744), authoritative source = team manifest.csv, owner **Aditya**, updated 2026-07-04.
- Team_Best ONNX sha256 = `ba4701c0455ac7715daa31b1117682876c1e41238176a3395569e980adde36c3` (== manifest sha).
- `profile_real_cost --arcgen 266` on the live team-best: `static=2618 profile=2618` (no dynamic fiction).

## Verdict: WALL. The card premise is obsolete.
The card targets `2618 -> floor_est ~900, EV +18.29`. That floor was priced against a **stale ~6716 baseline** (the old Slice+Pad live). Aditya has since rewritten task280 to a **moment-based** formulation at 2618 that is already within ~40–80 B of the forced-component floor. Points at 2618 = `25-ln(2618) = 17.13` (matches manifest 17.1298). A drop to 900 is worth only `25-ln(900)-17.13 ≈ +1.07 pt`, not +18.29 — and 900 is unreachable (see floor proof). The log-scale scoring makes any realistic sub-2618 delta worth <0.1 pt.

## The live formulation (decoded, fully verified)
1. **Moment extraction** (5 Einsums, F32): compute Σrow, Σcol, Σrow², Σcol², Σrow·col over the two color-2 marker cells (`channel_select`+`term_select` isolate channel 2). The quadratic formula (`Sqrt` of discriminant) recovers the two individual marker rows/cols from their moments; `Σrow·col` disambiguates the pairing.
2. **Direction + thickness**: 6 `GatherND` (batch_dims=1) sample right/up neighbors (direction) and 4 inward cells (contiguous depth ≤5). INT8 index arithmetic throughout.
3. **Render** (1 free-output Einsum): outer-products 20-wide row/col profile terms, scattered to the 30-grid via `spatial_select=[I₂₀|0]`, channel-remapped by `channel_tf3`; input passes through the same Einsum (no fill canvas ever materialized). Border (h or w ≥20) = all-zeros, matching the target encoding.

## Charged-cost decomposition of 2618
- Params 744: **spatial_select F32[20,30] = 600** (81% of params); channel_select 30, channel_tf3 27, ones 20, coords20 20, coord20m 20, term_select 15, scalars ~12.
- Memory 1874: **row_terms+col_terms F32[5,20] = 800** (43%); 6× GatherND I64[1,2,3] idx = 288; row/col_terms20 BOOL = 200; 16× [1,20] profile vectors = 320; misc ~266.

## Forced-component floor argument (~2540)
- **Render scatter (spatial_select, 600 params) is forced.** Placing 20-wide active profiles onto the 30-wide output grid via a linear (Einsum/MatMul) op needs a ≥20×30=600-element matrix. params = element count regardless of sparsity/dtype, so the [I₂₀|0] identity costs 600. The one initializer is shared by both the row-`rh` and col-`cw` scatters (optimal). Any alternative widens the *intermediates* instead (see below), which is dearer because they are not shared.
- **F32 render terms (800 B) are forced.** The final render Einsum consumes the F32 input alongside the terms; Einsum requires uniform dtype, so the terms must be F32. Narrowing to F16/INT8 requires casting the input → an [1,10,30,30] F16 intermediate = **+18,000 B** (the exact cost the pre-Aditya F16 live paid; see task280.md "input_f16 … 18000B unavoidable"). Aditya's F32 render *avoids* that cast; going narrow is a proven +18k regression.
- Channel remap (channel_select 30 + channel_tf3 27 + term_select 15) + moment machinery (~100) + depth/direction gathers (~450, GatherND indices are I64-only) + profile-build vectors (~520) are each near-minimal.
- Sum of forced components ≈ 2540. Live 2618 is ~80 B above it. **floor_est 900 is fantasy** — it is ~1600 B below the render+machinery floor.

## Second pass (2026-07-12, forced re-attempt): 2 genuinely-different formulations BUILT + GATED, both lose

| # | Route | Result | Why dead |
|---|---|---|---|
| 1 | **Fold spatial_select out of the 5 moment Einsums** into cw30/cw30sq coord-weight vectors | **BUILT + gated**: visible 267/267 PASS, `profile=2658` (mem 1874 + params 784) = **+40** | spatial_select must stay for the final render scatter, so the fold only *adds* cw30(30)+cw30sq(30)−coord20m(20)=+40. Artifact: `prototypes/build_task280_foldmoments_probe.py`. |
| 2 | **ALL-30 render, NO scatter matrix** (genuinely different dominant intermediate shape: 30-wide profiles fed directly to the render Einsum `…,th,tw→…`; spatial_select deleted; moments folded) | **BUILT + gated**: visible **267/267 PASS**, `profile=2738` (mem 2534 + params **204**) = **+120** | params collapse to 204 (−540) but 16 profile tensors + row/col_terms widen 20→30 = **+660 memory**. `ng_graph_cost_diff` names it exactly: the scatter's job is to *avoid the materialized 30-wide crops*. Shared 600-param map < 660 B of unshared widened intermediates. Artifact: `prototypes/build_task280_all30_noscatter.py`. |
| 3 | **F16/INT8 render** | proven **+18,000 / +9,000** | input F32 → narrow cast materializes [1,10,30,30]; `ort_reality`: free-output Einsum reading the F32 one-hot is **F32-locked** (mixed F32+F16 operands fail shape-inference, task351). |
| 4 | **Pad [5,20] terms → [5,30]** in lieu of spatial_select | priced **+640 net** | Pad output is a per-term intermediate (+600 B each, unshared) plus the [5,20] source still exists; worse than route 2's clean widen. |
| 5 | **Broadcast-Add / Greater render** (#67, e.g. task295 22748→2038) — no scatter, no F32 terms | information-theoretically **≥2,700–4,500 B** for the render alone | task280 output = **rank-2** (two separable stripes) **+ input passthrough**. Broadcast-Add is rank-1 per channel; two stripes need materialized 2-D planes (≥3×900 B). Passthrough-of-input forces a detection plane + 2-D Where (**#141: ≥4,500 B**). Aditya's fused Einsum does rank-2 + passthrough in **1,400** — it *beats* the #141 floor. |
| 6 | **Fewer / I32 GatherND** for the depth chain | no win | GatherND indices are **I64-only** (traps: "shape-inference hard-rejects INT32"); 2-D point sampling can't use I32 Gather without flattening the input (36 kB); batching conserves numel. |
| 7 | **Factor spatial_select** `[20,30]=D[20,X]@P[X,30]` (#119 in-kernel factoring) | no win | spatial_select is the **rank-20** identity `[I₂₀|0]`; a rank-20 map needs X≥20, so 50·X ≥ 1000 > 600. Full-rank identity is unfactorable-cheaper. |
| 8 | 20×20 output+Pad (16 kB pre-pad, visible 0/267); Slice/TopK local (6058/6716) | dead | don't-do log / historical. |

## Information-theoretic floor (why <2618 is unreachable by this route)
The render must (a) place **rank-2** separable stripe profiles onto the 30×30 grid and (b) pass the input through, all as a free output. Two irreducible charges:
1. **The 20→30 placement map.** Feeding a 20-wide active profile to a 30-wide output axis needs a linear map; its cost floor is `min(600 param, +660 memory)` — the shared full-rank identity **spatial_select = 600 params (zero memory, #119 SLOT-width optimal)**, empirically confirmed by route 2 (+120 when moved to memory) and route 7 (unfactorable). There is no third, cheaper primitive.
2. **The render terms are F32-locked at 800 B** (`ort_reality` Einsum dtype rule) and **irreducibly 5** (2 stripes × {band `any`, centerline `red`} + 1 passthrough `ones`; each color needs a distinct linear profile — a level-encoded merge is non-linear, impossible in Einsum).

Render floor = 600 + 800 = **1,400** (53% of cost). Add moments (~85), I64 depth-sampling (~450, forced), profile-build vectors (~320), term-Concat/misc (~360) → **~2,615 ≈ the live 2,618**. `floor_est 900` is ~1,700 B below this and is a category error (it ignores the forced render). **WALL.**

## Remaining hypothesis
None with positive EV. Only untested micro-idea: fold the constant "ones" passthrough term ([5,20]→[4,20], ≈−160 B ≈ +0.06 pt) — but it is entangled with the channel_tf3 inclusion-exclusion algebra and risks the documented hidden-bad −12.29 LB regression. Not worth it on a hidden-bad task for <0.1 pt.

## Hidden-bad note
task280 is a documented hidden-bad task (`beicicc-6645-39` swap = −12.29 LB despite visible+arc-gen pass). Any future candidate MUST be `ab --n 2000` byte-equal to the live 2618 output, not merely visible/truth-correct.
