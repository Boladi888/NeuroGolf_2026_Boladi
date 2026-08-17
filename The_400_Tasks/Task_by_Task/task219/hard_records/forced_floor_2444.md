# task219 hard-record — 2444 F16-CumSum graph is within ~150 B of its byte-equal forced floor

- **Live bar:** cost **2444** = memory 2354 + params 90. Team_Best sha256 `97fee100f21a77b5f3edc4e748ea5a2fdb15d0c5ad7e8c39aaf920dddd661cff` (owner Sho, manifest 2026-07-09). Points = `25 − ln(cost)` → 2444 = **17.20** (already banked).
- **Gold gate:** byte-equal-to-LIVE. Harness `run_raw = (out>0)` binarizes → only per-channel SIGN pattern is compared. Grid is ALWAYS 15×10 (all 200k corpus pairs); output colors {0,1,8} → channels {0,1,8}. LIVE is 71.1% vs ARC-GEN truth (ambiguous / "local-fail-LB-pass"); the ARC-GEN-CORRECT rule costs **39,370** (Sho `K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Sho/task219_analysis/`), so correctness is the *expensive* direction, not the cheap one.

## The graph (38 nodes) and its per-tensor cost
Slice ch8 15×10 → `Cast u8` mask → col-weight `Mul`+`ReduceMax` frontier & row-has → shifted-`Concat`+`Greater` **band-start markers** → `Cast f16`+**`CumSum`** = band-id → `ScatterElements`(min)=band-start-row, (max)=band-frontier + `Gather` per row → `ArgMax`=first-band row → template = 3 `Gather`ed rows → offset = r−band_start → fill-rows = band≥2 ∧ offset<template_height → `Gather` template-row-by-offset → frontier `Where`/`Greater` mask → merge `Where` → **`+3`** → **`BitwiseAnd[5,6]`** → **`ConvInteger[10,2,1,1]` pads=[0,0,15,20]** = expand+pad to [1,10,30,30].

Charged memory (sum of node outputs; input & output are FREE):
| block | tensors | bytes |
|---|---|---:|
| READ | `v_t0` F32[1,1,15,10] 600 + `v_t1` u8 mask 150 | **750** |
| DECODE | `v_t34` u8 tri-state 150 + `v_t35` (+3) 150 + `v_t36` 2-ch conv-in 300 | **600** |
| FILL-MERGE | `v_t30` template-per-row 150 + `v_t33` frontier-mask 150 | **300** |
| FRONTIER | `v_t2` col-weight plane 150 | **150** |
| small row/scalar vectors (~30 of them) | | **~554** |
| params (initializer elems) | | **90** |

## Why each big block is FORCED (proofs, not vibes)
1. **READ 750.** `Slice` preserves the F32 input dtype → the ch8 15×10 plane is F32 (600). Cheaper reads all lose: cast-whole-input = 9000 B; a `Conv`/`ConvInteger` that packs while reading emits ≥[1,1,30,21]=2520 B or needs an int input (=cast 9000). The `Cast→u8` (`v_t1`, 150) is mandatory because every downstream plane in F32 would be 4× (600 not 150). Both tensors needed.
2. **DECODE 600 (`+3` is unremovable).** Expanding a per-cell 3-state plane to a 10-channel 30×30 output cheaply requires ConvInteger (its 10-ch padded result is the FREE graph output; input is only 2×15×10=300). A 1-channel linear/`Conv`+bias CANNOT put channel-8 (the MIDDLE state=1) >0 while ≤0 at states 0 and 2 (a line has no interior max) → ≥2 input channels. With 2 channels from `BitwiseAnd`, background=state0 maps to `0&mask=0` = the origin; a **scalar** `x_zero_point z` then makes bg-output `−z·(w0+w1)` and the sign constraints force `z<1`, contradiction with integer `z≥1` — so bg can't be separated without moving it off-origin, i.e. the `+3` plane. **Per-channel `x_zero_point=[0,1]` is feasible** (clean weights exist: ch0=(-1,-1), ch8=(1,0), ch1=(0,1) on `&[1,2]`) **but the pinned ORT rejects a 1-D input zero-point for ConvInteger** (`IsScalarOr1ElementVector(X_Zero_Point) was false`). Repro: `prototypes/c1_drop_plus3.onnx` → RUNTIME_EXCEPTION. So `+3` stays.
3. **FILL-MERGE / FRONTIER.** The Where merge is per-cell (needs template-per-row + frontier-mask as 15×10). Frontier = rightmost ch8 col/row: the col-weight `Mul`+`ReduceMax` (150+15) beats the only alternative, `ArgMax(select_last)×row_has`, which adds an int64[1,1,15]=120 B tensor (net +105). LIVE is optimal here.
4. **Prompt hints that DON'T apply.** "TopK stream compaction" ⇒ `Compress`/`NonZero` are BANNED (`_EXCLUDED_OP_TYPES`). U8/BOOL `CumSum` is ORT-rejected (kill-rule; task322). CumSum-ordinal/run-start/rank-lane are ALREADY in the graph (Sho built exactly the suggested approach).

## Second pass (2026-07-11, forced re-attempt) — library-mined + empirically gated alternatives, all WORSE
Mined `INDEX→techniques.md/op_cost_patterns.md` + BoBrain provider. Every genuinely-different assembly costs MORE:
- **Insight found (still can't exploit):** output = **input + fill-overlay** — the input one-hot ALREADY has ch0-hot@bg and ch8-hot@color8; the ONLY change is at fill cells (ch0 1→0, ch1 0→1). But any Add/overlay to place a delta needs a full `[1,10,30,30]` (36 KB) intermediate unless it is the *last* op, and the last-op free-output slot is exactly what ConvInteger already uses. So overlay can't beat reconstruction.
- **task147 ScatterElements-onto-free-`output`:** `updates` dtype is LOCKED to `data`; with `data`=F32 input the fill `updates[1,2,15,10]` = **1200 B F32** (vs ConvInteger u8 300); casting input first = fatal 36 KB. WORSE.
- **task280 Pad-BOOL one-hot decode — BUILT + GATED:** `prototypes/c2_pad_onehot_decode.onnx` (Gather color {0,8,1} → `Equal` one-hot `[1,10,15,10]` → `Pad`). **byte-equal to LIVE (mism=0/300)** but profile **4,691** (memory 4604+params 87) = **+2,247 (≈2× worse)** — the 10-channel `[1,10,15,10]`=1500 plane + int32 planes dominate. This is the empirical proof that ConvInteger's 2-ch→10-ch-free expansion is near-optimal, not just convenient.
- **task201/task255 (same "Filling Regions" family, "F32 one-hot read floor"):** the applicable levers (slice-BEFORE-cast; narrow 0/1 planes to UINT8) are ALREADY in LIVE. No Gather-index fold available (LIVE's 4 Gathers have distinct data+index).
- **Read floor is information-theoretic:** ONNX has no fused slice-narrow; `Slice`/`Gather` preserve F32, `Cast`/`Greater` on the whole input = 9000 B. So the first materialization of the 15×10 ch8 region is F32 = **600 B, forced**; a strided-`Conv` pack emits `[1,1,30,3]`=360 but then needs an expensive uint16 unpack (300+) AND still needs the u8 mask for output-ch8 → nets worse.

## Verdict
No byte-equal candidate below 2444 exists at the tensor-count/dtype level; the read, decode, and merge blocks are individually proven forced. The `floor_est ~900` = `2444/2.7` is a heuristic that inspection disproves. **WALL.** One remaining (low-probability, NOT demonstrated cheaper) hypothesis: a wholesale bit-packed row-descriptor rewrite might shave only the single frontier plane (`v_t2`, ~150) while the 750 read + 600 decode + 300 merge stay fixed → best-case ~2294 with high byte-equal risk on the idiosyncratic 71% rule; the ArgMax result shows packing tends to net worse via int64 index tensors. Not worth the regression risk for a ~0.06-point (`ln(2444/2294)`) log-scale gain.
