# task119 — MomentEinsumPack route: WALL

- **Live bar**: cost **1131** (memory 1005 + params 126), creator Aditya, sha256 `38381961b34e7bd6868048abbea87fbae2a20afe34414248bf7bc11f3aaddae0`.
  Verified live in workbook (`NG_OnnxlyFans_TaskStatus.xlsx`, Tasks sheet), `manifest.csv`, and by re-scoring the Team_Best_Onnx (`score_v2`=(0,1005,126)=1131, `profile_real_cost`=1131, no runtime inflation).
- **Route attempted (both named levers)**: (1) merge duplicated seed/wall moment Einsums; (2) pack ScatterND indices.
- **Verdict: WALL.** Both named levers are dead; the dominant cost is a proven floor.

## Live architecture (Aditya, 56 nodes)
Fully analytic edge-seed-bounce-ray → coordinate ScatterND. 6 free-input Einsums compute seed(ch8)+wall(ch2) count & row/col moments (`nchw,c->`, `nchw,c,h->`, `nchw,c,w->`); scalar direction/bounce logic derives 10 ray-cell (row,col) coords via reflected-linear Gemms (`|t·apex + base|·sign + boundary`); a single terminal `ScatterND(input, indices[20,4] int64, updates[20])` writes the ray. Output = ScatterND output (graph output, free). Input free.

## Per-tensor memory (static == profile == 1005)
- **`scatter_indices` [20,4] INT64 = 640 B (64% of memory)** ← the target
- scatter_indices_u8 [20,4] = 80 (pre-cast); geometry 7×[10,1] f16 = 140; vec_r20/vec_c20 [20,1] u8 = 40; vec_r/vec_c u8 [10,1] = 20; ~30 scalar logic tensors ≈ 85.
- params 126: batch_channel_cols[20,2]=40, idx30[30]=30, scatter_updates[20]=20, sel8+sel2=20, tvec9=9, scalars≈7.

## Falsification — lever 1 (merge Einsums): MEASURED WORSE
Stacked sel8+sel2 → [2,10], ran 3 merged Einsums `nchw,kc->k` / `nchw,kc,h->k` / `nchw,kc,w->k` producing `[2]` vectors, then Sliced back to the 6 scalars downstream.
- Result `score_v2` = (0, **1029**, **130**) = **1159 > 1131 (+28 B)**.
- Why: the 6 Einsum outputs are scalars (4 B each) — merging them is byte-neutral, but splitting each `[2]` back into two scalars **adds Slice intermediates** (mem 1005→1029) and the stacked selector + Slice start/end constants **add params** (126→130). Merge is strictly counterproductive.

## Falsification — lever 2 (pack ScatterND indices): PROVEN FLOOR at 640 B
The [20,4] int64 index cannot shrink:
1. **int64 forced.** int32 indices rejected by `onnx.checker.check_model(full_check=True)` (which `calculate_memory` runs) AND by ORT (`INVALID_GRAPH: Type Error`). Tested directly. No smaller dtype exists for ScatterND `indices`.
2. **20 updates forced.** Decode (`ng_harness/encode.py::decode_to_grid`) requires **exactly one channel >0 per cell** (0→invalid −1, ≥2→invalid 99). Ray cells are color-0 background (ch0=1). Converting to color-3 needs BOTH a ch0-clear (→0) and a ch3-set (→1): two element writes per cell. A single element write always leaves the cell 0-hot or 2-hot (invalid). The "set ch3=2, skip ch0-clear, let argmax win" trick FAILS this strict decode.
3. **10 ray positions forced.** Distinct color-3 cell count over 20,000 arc-gen ground-truths is {9: 9997, 10: 10003} — max = 10, hit in half the cases. 9 positions would miss ~50% ⇒ certain visible-zero. So updates = 2 channels × 10 cells = 20.
4. **4 columns forced.** Writing individual elements of the 4-D `[1,10,30,30]` needs full-rank (4) index rows. Reducing to width-3 addresses whole rows (30 cols), not cells. Dropping the batch column would require reshaping/squeezing input to `[10,30,30]`, whose intermediate is [10,30,30] F32 = **36,000 B** — catastrophic.
5. **Single ScatterND forced.** Splitting into two chained ScatterND (clear, then set) makes the first scatter's output a non-terminal `[1,10,30,30]` F32 intermediate = **360,000 B**. Dead.
6. **Assembly already minimal.** Keep coords in u8 and do ONE full-width u8→int64 cast at the end (80 B pre-cast + 640 final). Casting per-column to int64, or building the [20,1]/[20,2] blocks in int64, all inflate intermediates (int64 [20,2]=320). Computing geometry at [20,1] (pre-duplicated) doubles the f16 chain (140→~280). Current graph is Pareto-tight.

## Alternative render family ruled out
Where-overlay `Where(ray_mask, color3_onehot, input)` needs a `[1,1,30,30]` bool mask = **900 B > 640** (must be 30×30 at the Where to broadcast against input; a 12×12 mask must Pad to 30×30 → 900 B intermediate). Confirmed by home ladder v3 = 2958 and library verdicts #109/#132 (data-dependent diagonal/ray spreads are Conv/render floors, not free-output separable). Free-output Einsum was previously priced worse (don't-do log). ScatterND@640 is the cheapest known render.

## One remaining hypothesis (not pursued — sub-1% EV, high byte-equal risk)
Shave 1–2 scalar-logic intermediates from the ~85 B tail via boolean fusion (e.g. collapse a Where/Or chain). Would need full 200k byte-equal re-gate for a <20 B (<1.8%) gain and cannot reach the 950–1050 target (640 index + ~365 machinery + 126 params is structural). The 950–1050 prompt target appears infeasible — it predates recognition of the 640 B int64 floor. Per prompt ("not a floor-proving task; if the named lever fails, report the exact falsification and stop"), stopped here rather than sieging the machinery.
