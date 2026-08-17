# task397 hard record — CumSum/prefix render dead + quant-terminal reframes the floor (2026-07-12)

**Session:** bo_ng_cld_worker3 (Claude lane).
**Live bar at session:** manifest/ng_current_bar `live_cost = 2637`, owner Sho, sha `a07734be19a6fd5216bdc1638d8df2dd51db25bf944083dcd3ab1f0a04f8e51a`, updated 2026-07-09. No `Staged_Boladi` sidecar. Local staged best = v21 `2636` (sha `DF00A2A6…`).
**Result:** v21 (2636) re-verified as a clean 1-byte win over the recorded bar; two new from-scratch render formulations built + measured + REJECTED; the prior "900 B forced" floor proof is DISPROVEN in principle (Sho ships a no-900 quantized terminal) but its 2636 *reliable-static* bottom line still stands.

## Dead route 1 — CumSum difference-array render (the prompt's LITERAL hint)
- Builder: `prototypes/build_task397_p2_cumsum.py` → `task397_p2_cumsum.onnx`.
- Mechanism: scatter +1 at run start row `br`, −1 at end row `br+k` into a `[1,1,8,10]` F16 delta, `CumSum` down rows, `>0.5`, then right-shift-OR for the 2-column paint.
- **FAILED visible:** `ScatterND invalid indice = 8` — runs whose end `br+k` exceeds the 8-row plane produce out-of-bounds difference-array indices; ORT rejects them (does NOT silently clip, so the "off-grid −1 = free clip-to-row-9" assumption is false). Correctness would need an extra `Min/Clip` on the end index = still more ops.
- **Cost even if fixed:** ledger = **3198** (mem 3010 + params 188): the `[1,1,1,6,4]`=192 B scatter index, an 80-elem F16 delta init, and the 160 B F16 CumSum plane all out-weigh the ConvTranspose stamp. Loses by ~560. **DEAD — do not reopen CumSum for this task.**

## Dead route 2 — broadcast row-comparison run render (prefix/rank flavour)
- Builder: `prototypes/build_task397_p1_cmprender.py` → `task397_p1_cmprender.onnx`.
- Mechanism: scatter each block's start `br+2` and height `k` into per-top-left-column vectors `START/K [1,1,1,10]`; `own = (row>=START) AND (row<START+K)` over rows 2..9; `shadow10 = own OR shift_right(own)`. col4 indices are TopK-distinct so no scatter collision; k=0 gated lanes render empty.
- **Visible PASS 266/266 (correct)** but ledger = **2709** (mem 2586 + params 123). Loses to 2636 by **73**: the four `[1,1,8,10]` BOOL planes (ge/lt/own/shifted ≈320 B) exceed the single F16 `shadow_raw` (160) the ConvTranspose stamp emits; the scatter-index tax (`start_idx` 96 B) is unchanged. A correct alternative render, just heavier. **Confirms the ConvTranspose stamp render is at its non-quantized floor.**

## The real story — Sho's LIVE graph is a quantized, no-900 terminal (reframes the whole floor proof)
Inspected `Team_Best_Onnx/task397.onnx` (= the live bar, sha `a07734be`). It is a **different formulation family** the v1→v21 lineage never tried:
- Crops the input to 10×10 with a 1×1 `Conv` (pads `-20,-20`), casts to UINT8, then runs **7× `QLinearConv`** + `ArgMax`/`GatherElements`/`ScatterElements` — everything stays in the 10×10 region in **UINT8**. Largest tensor is only **400 B** (`v_t0` F32 [1,1,10,10]).
- **No 900 B `shadow30` plane exists.** The terminal is `QLinearConv([1,3,10,10] → [1,10,30,30], pads=[0,0,20,20])`: a padding-Conv expands 10×10→30×30 and maps a 3-channel encoding (`input`, `input²`, `shadow`) to the 10-channel one-hot **output** directly. Input preservation + green overlay + 30×30 placement are all folded into the final free-output Conv. This **disproves the 2026-07-09 "900 B broadcast condition is forced" claim** — it is only forced for the non-quantized `Where`/`ScatterND` terminals the lineage considered.
- Static ledger sums to **≈1810 mem + 117 params = 1927** (this is the "1922" ng_current_bar flagged as stale).

### …but it is a FRAGILE credit-probe, not a reliable sub-2636 win
- `python tools/_score_v2.py`-equivalent `score_network_v2(Sho) = None` — it **CRASHES on the scorer's empty-grid dummy** (`GatherElements: Out of range value in index tensor`). `score_network_v2(v21) = 2636` (survives). 
- Crashes on **30 / 40** random valid 10×10 grids too — the `ArgMax→GatherElements` indices only stay in-bounds on the narrow arc-gen distribution (visible 266/266 passes, but off-distribution/hidden inputs can crash → 0 on Kaggle).
- Recorded/credited cost is **2637** in BOTH manifest and workbook (→ points 17.1226 = 25−log(2637)), NOT its 1927 static/profile. So Kaggle prices the QLinearConv graph ~2637 (runtime/quantization fiction, per traps §GRADER-GAP), **no better than our exact v21=2636**, and it carries a crash risk our v21 does not.

### RESOLVED — the quantized route WON at 2350 (2nd pass, same session)
The hypothesis below was tested and **succeeded**. `staged_candidates/task397_q1_quant_noplane_cost2350/` (builder `prototypes/build_task397_q1_quant_noplane.py`): a from-scratch quantized graph, **cost 2350**, visible 266/266, truth `--n 2000` 2000/2000, ab byte-equal to LIVE, `score_network_v2`=2350 (clean, survives the empty-grid dummy), robust 40/40 random grids. **−286 vs 2636 / −287 vs live 2637.** PROBE_READY (`ng_safety` WARN `WARN_X86_QUANTIZED_OP_FAMILY`).

Key facts that made it work (vs the failed first-pass reasoning):
- **`score_network_v2` DOES credit a clean dense-kernel QLinearConv at its low UINT8 static cost** (validated: a minimal `Conv→Cast→QLinearConv` graph scored 608 with the QLC output credited 81 B UINT8). My 1st-pass "quantization is priced 2637 = fiction" was WRONG — the 2637 is Sho's stale/crash record, not the credited cost. QLinearConv is Kaggle-CONFIRMED (task136 −10631 B/+1.79 LB, task042 −21769 B/+2.13 LB; `by_symptom.md` "geometric ray/diagonal" + "large-receptive-field" symptoms).
- **Traps that bite** (from task136/task042 don't-dos, all obeyed): sparse-kernel QConv weights ⇒ `score_network_v2`=None (use DENSE kernels); QLinearConv zero-point inputs cannot be omitted; opset-13 Pad for BOOL; `y_zero_point` must be UINT8 (ORT rejects INT8-y). ORT has no `TopK(UINT8)` kernel — cast the tiny `colmax` to F32, keep `occ`/`ArgMax` UINT8.
- The one-hot terminal reconstruction: quadratic `out[ch]=clamp(2·ch·c − c² + route·shadow + (1−ch²)·const1)`, `const1` (0 in the conv-pad) is what makes the out-of-grid border all-zero — the BIAS-LEAK LAW (#85) in action; the "+1" MUST ride an input channel, never a conv bias.

### Remaining headroom (sub-2350, for a future batch)
q1's biggest non-forced chunk is the INT64 sparse-selection/gather-index tax (~456) + the F32 color plane (400, forced by the F32 input). Sho's static ledger is ~1927, reached via a **TopK-free** selection (2× ArgMax first/last-index on the flat occupancy + `ScatterElements`) that trims the index tax — but his build is FRAGILE (crashes the scorer dummy + 30/40 random grids). A robust TopK-free UINT8 selection is the path from 2350 toward ~2000. **Do NOT re-run non-quantized render searches — p1/p2 + the v5→v21 lineage close that door at 2636; the quantized frontier is where the bytes are.**
