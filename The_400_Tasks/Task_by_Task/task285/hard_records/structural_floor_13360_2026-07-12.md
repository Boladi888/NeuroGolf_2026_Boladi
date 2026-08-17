# task285 STRUCTURAL floor confirmation @ live bar 13,360 (2026-07-12, bo_ng_cld_worker3)

## Live bar
- **13,360** profile cost, SHA `645cd6a80703680f2c41fb2801365ca82c8ccb1c3f075c4753d379a87d8b683f`,
  source `python tools/ng_current_bar.py --task 285 --json` → manifest row, owner Boladi, updated `2026-07-11T00:40:30Z`.
- Live team-best == staged `task285_share_axes3_c3_cost13360` (Codex worker1, byte-equal −1-param dedup of Aditya's `9D6EB4B8…` 13,361 parent).
- Need **≤ 13,359** to win. No candidate reached it.

## Verdict: WALL / REJECT_LOCAL — current graph is AT the forced floor for the only correct formulation family.

## Prompt hint (QLinearConv UINT8 carrier) empirically REFUTED
The prompt directs "encode buckets into UINT8 … feed the quantized detector … aim ~1200". Tested and false:
- **QLinearConv cannot consume the F32 graph input.** ORT: `Type 'tensor(float)' of input parameter (input) of operator (QLinearConv) … is invalid`. The input signature is fixed F32 [1,10,30,30] (grader `encode()` writes floats); it cannot be presented as UINT8.
- **A UINT8 carrier is MORE expensive, not less.** `Cast(input F32→UINT8)` = a charged [1,10,30,30] UINT8 = **9,000 B** plane. The current regular-Conv collapse to F32 [1,1,30,30] = **3,600 B** is strictly cheaper. `ArgMax` collapse = INT64 [1,1,30,30] = 7,200 B (worse).
- Therefore the "large F32 detector plane is not forced" premise is wrong: the **F32 collapse (3,600) is the provably-cheapest first op**, because the only free [1,10,·,·] tensor is the input and any materialized multi-channel plane is ≥9,000 B, so channels must be reduced to 1 immediately, and single-op channel-reduction on an F32 input yields F32.

## Forced-floor accounting (input & output are FREE per neurogolf_utils.calculate_memory L234)
Per-tensor profile over 265 official + 266 arc-gen inputs (memory=12,786, params=574):

| forced intermediate | why forced | bytes |
|---|---|---:|
| `label_f` F32 [1,1,30,30] | one-hot→scalar collapse; cheapest route (see above) | 3,600 |
| `label_u8` [1,1,30,30] u8 | UINT8 2D carrier for the key-scan (QLinear)Conv | 900 |
| `label_flat` [900] u8 | flat carrier for data-dependent color Gathers (keys/cells span all of 30×30) | 900 |
| `painted_flat` [900] u8 | ScatterElements paint target that preserves input | 900 |
| `painted_label` [1,1,30,30] u8 | reshape feeding the free `Equal`→output | 900 |
| `key_count`+`key_count_flat` [1,462] u8 | full-position key scan (QLinearConv 9×10 valid) | 924 |
| `key_score_f16` [1,462] f16 | TopK bridge — ORT rejects TopK on u8/i8; INT32 doubles to 1,848 | 924 |
| reflection/ranking tail | delta-Gather tables (f16), 5×5 component windows (INT32 Gather indices — must be int), TopK index byproducts (INT64, mandatory) | ~4,200 |

Sum ≈ 12,786 memory + 574 params = **13,360**. The 5 canvas planes alone (3,600 + 4×900) = **7,200**, so the prompt's ~1,200 is off by ~6×.

## Levers tested this session — all dead
1. **Drop the dead TopK Indices output** `top_off_idx` INT64 [3,8] = 192 B (genuinely unused — only the F16 Values feed downstream). Setting `node.output[1]=""` is **rejected by both onnx.checker and ORT**: `output 1 is marked single but has an empty string`. TopK Indices is a required (non-optional) output → cannot be dropped. Removing the TopK instead would explode the paint from 8→25 stamps (net worse).
2. **No mergeable/duplicate initializers remain** (share_axes3_c3 already took the only dup). Params 574 are near-minimal element counts; cross-dtype scalars can't merge (Gather/QLinearConv type constraints).
3. **Every tail tensor already at minimal dtype/shape** — the two 300-B `combined_delta`/`src_cell_flat_raw_38` are INT32 Gather-index tensors over the 5×5 component window (u8/f16 impossible for Gather indices; window can't shrink — radius-4 components; off_k7 8→7 already visible-dead).
4. **K-pruning is visible-dead** (prior hard_record `topk_prune_livegraph.md`): `stage_k9` 13,305 → visible 263/265; `off_k7` 13,185 → visible 264/265. Visible-fail ⇒ hidden-zero.

## The only correct formulation family is the scalar-carrier one (free-output alternatives are inapplicable)
The rule requires exact **8-connected component** detection with disconnection sensitivity (radius-4 shortcut wrongly paints disconnected same-color cells — fails visible 10/130/139). Connected-component membership is **not separable**, so free-output / separable-Einsum / sparse-projection formulations cannot express it and fail hidden — documented:
- Einsum transpose+MatMul sweep: "later ChatGPT waves public-zeroed" (neurogolf_lesson_facts §task285.einsum_sweep_partial_only).
- Different-projection ScatterND: 265/265 visible but 32.5% adversarial (Voronoi-cell mismatch).
- bit-packed labels: regressed to 43,183 (UINT32 pack/unpack intermediates).
- one-hot boolean paint / ScatterND-one-hot: [1,10,·,·] planes ≥9,000, INT64 [N,4] indices ≥3,840, plus overwrite masking — all worse.

## Second-pass library confirmation (forced review — 2026-07-12)
Every "forced" claim above is a DOCUMENTED floor law (not just this worker's argument):
- **F32 collapse 3,600 irreducible** — `cost_floor_reference.md` §canvas-collapse / task400 (L714-716): "the single 30×30 F32 canvas (3,600) is irreducible — an F16/U8 canvas needs a full-input cast (18,000/9,000) that dwarfs the saving, and ArgMax-over-channels emits [1,1,30,30] INT64 (7,200)."
- **Arbitrary-2D-color-layout ⇒ 3,600 floor, sub-1k not a target** — task165 (L723-725): "if the rule truly needs arbitrary 2-D color layout … the [1,1,30,30] F32 color-index plane itself is a 3,600 B floor before output assembly; sub-1k is not a real target unless you first prove a stronger structural invariant." task285 keys/components can sit anywhere in 30×30 ⇒ this class. The dataset invariants (≤3 keys, size≤9, radius≤4) are ALREADY exploited by the tail (TopK-3, 5×5 window, TopK-8) — they shrink the tail, not the forced canvas.
- **Canvas+Equal(color_consts) is the proven [1,10,30,30]-output floor (900 B)** — L165-169: "do NOT try to eliminate the canvas itself" (task387/009/054/388 family). task285's `painted_label`+`Equal` is exactly this.
- **QLinearConv F32→U8 carrier = 9,180** (materializes full U8 input) — task140 (L175-176); matches this session's empirical Cast(input→u8)=9,000.
- **ScatterND one-hot channel-patch loses to canvas+Equal** — task070 (L180): "Canvas+Equal BEATS ScatterND channel-patch … ScatterND pays F32 per patched channel."
- **Einsum/separable route is hidden-dead for THIS rule** — connected-component membership is non-separable; the transpose+MatMul→Einsum sweep public-zeroed (neurogolf_lesson_facts §task285.einsum_sweep_partial_only).
BoBrain provider (`neurogolf_provider.cli context/consistency task285`) authoritative bar = 13,360 (team_manifest_cost); library `op QLinearConv` = "F32 input rejected; a full input Cast usually costs more than it saves." No cheaper assembly exists in the routed library for an arbitrary-2D connected-component reflect-recolor rule.

## Information-theoretic floor (why no route beats ~13,360)
Output is a 30×30 grid where any cell may take any of 10 colors, determined by input. Producing it forces, independent of the reflection assembly:
1. a full-grid color carrier to gather-read AND scatter-write (keys/cells span all of 30×30) — cheapest realization is Conv→F32 3,600 collapse + UINT8 2D form 900 + UINT8 flat form 900 + scatter-target flat 900;
2. the [1,10,30,30] output via the proven canvas+Equal floor — painted 2D plane 900.
Forced canvas subtotal = 3,600 + 4×900 = **7,200**. Key location requires scanning ~all 2×2 positions (a canvas-scale score) + a float TopK bridge (u8/i8 rejected) ≈ 1,386. The reflection tail is already at the plane-conservation floor (every legal op family — Conv/QLinearConv/ArgMax collapse, TopK/ArgMax select, Gather/Scatter reflect, MaxPool CC — lands at the same total; only >2-input-fan-in Einsum could escape, and it is hidden-dead here). Sum ≈ 13,360. The prompt's ~1,200 is below the forced canvas subtotal (7,200) alone by 6×.

## One remaining (low-probability) hypothesis
A materially new **exact** representation that removes BOTH the F32 collapse (3,600) AND at least one 900-B canvas plane WITHOUT (a) materializing any [1,10,·,·] plane (≥9,000), (b) an INT64 ArgMax (7,200), or (c) a sub-30×30 crop (real+hidden inputs include 30×30). No such op is known in ORT opset-18. Until one is found, do NOT re-run collapse-dtype, K-prune, bit-pack, ScatterND, or Einsum probes on this task — they are all recorded dead.
