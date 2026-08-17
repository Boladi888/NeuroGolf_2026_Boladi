# Hard record — render/scatter consolidation below 5722 is DEAD (bo_ng_cld_worker3, 2026-07-12, second pass)

## Live bar / state
- Authoritative bar: **LIVE 6008**, sha `085a8d5f5ac24623985183ae6f21e96bd9b689a5138e4769ceef95117835fd83` (`ng_current_bar.py`, manifest, Boladi).
- **Standing win this session:** `staged_candidates/task157_flat16_tail_cost5722/` (cost **5722**, −286 vs LIVE, PATH_A_BYTE_EQUAL, ab 2265/2265 byte_equal=True). Effective floor to beat = strict-below 5722.

## Routes attempted this pass (genuinely-different render/scatter assemblies) — all worse/blocked

The 5722 graph's placed-plane render is a 4× sequential `ScatterElements` (axis=0, [90] bool canvas) with obj4 uniquely doing `Gather(prev)+Or(crop)` to OR-merge its overlapping window. Two consolidation ideas:

### 1. Single `ScatterElements`/`ScatterND` with all 64 placements — WORSE (+18B..+60B, paper)
Collapsing the 4 scatters into 1 needs the indices as one `[64]` (or `[64,1]`) int32 tensor + updates `[64]` bool. Concatenating the four per-object `[16]` index vectors materialises a **256B** `[64]` int32 (and 256B `[64,1]` for ScatterND) + **64B** `[64]` updates. That **added ≥320B > the 302B** saved by dropping 3×90B canvases + obj4's 32B Gather/Or. The sequential scatter is *cheaper precisely because it scatters `[16]` at a time and never materialises the `[64]` concatenation.* Partial (obj1+2+3 → one `[48]` scatter) is also net +60B for the same reason.

### 2. `ScatterElements reduction='max'` (OR-accumulate; drop obj4 Gather+Or, −32B) — BLOCKED by ORT
`max` of bool == OR, so `Scatter(data, idx, crop, reduction='max')` reproduces obj4's `Gather+Or` merged value by construction (and obj1 is trivially equal since its base is all-False). This would delete `obj4_prev_place_values` + `obj4_merged_place_updates` (−32B → ~5690). **ORT rejects it at runtime:** `bool data type is not supported with ScatterElements opset 18 when reduction is 'max'`. Builder `prototypes/build_task157_flat16_scattermax_v2.py` (onnx deleted; RUNTIME_EXCEPTION → visible-fail/unscoreable). A uint8 workaround (cast crops→u8, scatter into u8 [90] canvases with max, cast back) keeps the same 90B/canvas but adds 4 crop Casts + 1 final Cast (~+80B) → **net worse** than the −32B it buys.

## Other priced byte-equal micro-levers NOT pursued (poor byte-per-risk on an ambiguous-class gold gate)
- `[4,1]→[4]` `*_case_code_matrix`/`obj*_code_matrix` reshapes (4×8B=32B): the `[4,1]` is `Gather(bottom_code_by_anchor[4,3], anchor[1], axis=1)`; getting `[4]` directly needs the anchor index squeezed `[1]→[]`, but the anchor `[1]` tensors also feed row-base computations → load-bearing shape cascade.
- `bottom5_b [4,15]`/`top0_visible_b [2,15]` Squeeze double-count (~90B): the bool is genuinely needed in **two ranks** — 2D for the crop `Concat→Gather` path and 4D `[1,1,·,15]` for the matcher's u8 Cast — so one reshape is forced unless the whole matcher is refactored 4D↔2D (high byte-equality risk).
- 13× float16 `[]`→Unsqueeze→`[1]` `*_norm_summed` (~26B): `ReduceSum` reduces `[3,4]` over axes `[0,1]`; `keepdims=1` yields `[1,1]≠[1]` → breaks the downstream Concat. NOT byte-safe.

## Forced-component floor at 5722 (truth-safe / byte-equal family)
Dominant costs, each ORT-forced or inherited from the LIVE structural matcher:
- **matcher ~4600** (float16 norm/code 658B + int32 place/score + bool candidate masks): the inherited LIVE footprint-matcher; the cheapest truth-safe matcher found across 5+ workers (assign-only probe alone = 4619; enumeration = 37,310; lookup = truth-unsafe 1.3%).
- **carrier 300B** `acg2ch` uint8 `[1,2,10,15]` — a linear QLinearConv decoding 3 colours {0,1,2} to a 10-ch one-hot needs 2 channels ({c,c²} or {is1,is2}); uint8 is the 1-byte floor; must span all 10 active rows (carrier6row is visible-fail).
- **placed plane 150B** `ch1_active_pad10` bool `[1,1,10,15]` — rows 6-9 must render channel-0=1 explicitly (conv-pad renders all-zero → visible 0/265).
- **F32 reads 360B** (240B+120B) — F32-input Slice stays F32 (cast-full-input loses).
- **scatter canvas 360B** (4×90B) — sequential last-wins/OR; consolidation is worse (this record), reduction is blocked (this record).

## Verdict: STRICT WIN at 5722; WALL below it for the truth-safe family.
Every genuinely-different render/scatter assembly prices worse or is ORT-blocked; the remaining byte-equal micro-levers are load-bearing shape cascades with poor byte-per-risk on the ambiguous-class byte-equal gold gate. The only sub-5722 *numbers* remain the truth-unsafe lookup memorizer (u64 port truth 261/20000=1.3%). Nothing new staged beyond the standing 5722 win.
