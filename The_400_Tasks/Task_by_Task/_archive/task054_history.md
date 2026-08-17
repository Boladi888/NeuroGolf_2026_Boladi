# task054 — archived history (moved 2026-07-11 by bo_ng_cld_worker1)

Verbatim history moved out of `Task_by_Task/task054.md` to keep it under the
repo-lint size cap. The live file keeps a compact Attempts table + current
floor/Don't-do; full per-rung prose and superseded analyses live here.

---

## Algorithm sketch (superseded partial graph decode — the true rule is in task054.md ## Rule)

Partial sketch from graph structure:

```python
color = argmax_onehot(input)              # scalar color index, 30x30
bg = color[0, 0]
nonbg = color != bg

# v48b: center of a non-bg solid 3x3 same-color block.
same_neighbors = eight_neighbors_have_zero_variance(color)
center_matches = 8 * color == sum_8_neighbors(color)
marker = same_neighbors & nonbg & center_matches

# v58b: marker cells whose color is the chosen marker color and have cross-neighbor support.
marker_color = max(color where marker)
candidate = nonbg & (color == marker_color) & cross_neighbor_count(nonbg) == 4

# h/v/diag colors are sampled from shifted candidate masks, then masks are grown around marker.
# CumSum + directional MaxPool extends arms until bg boundaries.
```

## Pre-build novelty statement (2026-05-21)

Polish-pass status:
- F16-output trick: applied (graph output is BOOL per diag)
- onnxsim + safe_opt: checked through `spread_scan.py --tasks task054 --verify-inputs 10 --summary-only`; 25 known patterns including onnxsim and safe_opt returned 0 bytes saved on current canonical
- spread_scan: no residual automatic pattern savings; `patterns_applied=0 saved=0 bytes`

My hypothesis for where 178k goes:
- Current canonical is already scalar-color internally, not ten-channel: most wide tensors are F32/BOOL `[1, 1, 30, 30]`, not `[1, 10, 30, 30]`.
- ~45 F32 `[1, 1, 30, 30]` intermediates dominate memory (each 3,600 B), especially Conv/Mul/Sub/Cast/Where/CumSum/MaxPool chains.
- 5 Conv outputs are F32 `[1, 1, 30, 30]` = 18k total; the 5x5 template Conv is only one of them.
- 4 CumSum + 4 MaxPool + 4 Sub directional arm tensors are full F32 grids = ~43k before the bool arm masks.
- 13 Where outputs are F32 when selecting color/index values; they are cheap only where prior v3 converted eligible boolean masks to And/Or.
- Rule 12 is not directly present in the current canonical: no `Greater -> Cast -> ReduceMax` chain was found; the four ReduceMax nodes consume masked color grids (`v50/v60/v66/v73`), not comparison masks.

Architectural change I'm trying:
- Vector A: selective F16 scalar-color pipeline after the first color-index Conv, preserving BOOL output and checking ORT dtype constraints.
- Vector B: eliminate or fuse duplicate directional CumSum/MaxPool blocks if forward/reverse arm detection has a smaller equivalent.
- Vector C: algorithm-level simplification of the multi-rect/cross-template rule if graph diagnosis exposes a smaller direct construction.

## Attempts — full per-rung prose (compact table now lives in task054.md)

- 2026-05-21 | radical v4 F16/UINT8 scalar pipeline | staged `.hermes/board/staged_submits/task054_radical_f16_u8_t_C7AD4366_on_178511/` | cost 178,511 -> 97,423 (-81,088B); 266/266 visible/live; adversarial_against_live 96/100; changes: opset 14 F16 CumSum, cast after first color-index Conv instead of full-input cast, UINT8 color reductions/final assembly, 5x5 any-template Conv -> UINT8 MaxPool, diagonal shift chain -> corner Conv, factored v103mask | `prototypes/build_task054_radical_v4_f16_u8.py`; hours_spent ~3.0
- 2026-05-30 | worker-7 **v1 true-rule from-scratch rebuild** | `prototypes/build_task054_v1.py` + `task054_v1.onnx` | **cost 97,423 -> 55,448 (-43.1%); +0.564 LB** (14.077 vs 13.513 pts at raw log_e ×1.0). GATES: verify_against_live 266/266 (proto-vs-LIVE) + 266/266 vs expected; **byte-equal vs SHIPPED PARENT 8000/8000 on fresh ARC-GEN** (seed 1230000) + both 100% vs GT; numpy ref 266/266 + 5000/5000 ARC-GEN. Decoded the TRUE rule (flag-star), dropped the parent's variance-Conv legend detector + redundant color-read/side-mask machinery. New graph: 1 color Conv, centroid legend-read, box/marker masks, **4-direction CumSum+MaxPool run-fill lines**, 3×3 surround stamp, surround-before-line compose. hours_spent ~4.0
- 2026-06-02 | **v2 exact compression of v1/live** | `prototypes/build_task054_v2.py` + `task054_v2.onnx` | **cost 55,448 -> 47,180** (`score_network_v2=(0, 47,146, 34)`, -8,268). GATES: visible 266/266; sample AB 5,266/5,266; **full Path-A AB vs extracted live parent 200,266/200,266 byte_equal=True**. Ladder: 55,448 -> 49,006 -> 48,107 -> 47,208 -> 47,180. Changes: direct U8 special/marker/legend algebra, bbox legend center, direct scalar color gathers, box-only base compose, U8 line-band scalar gating before one `Greater`, surround scalar-gate, center painted through the line layer, scalar orientation/surround `Not` folds, INT32 legend index arithmetic, initializer/Slice cleanup. Staged `.hermes/board/staged_submits/task054_v2_exact_t_158bf990_on_55448/`.
- 2026-06-04 | worker-7 **v5 sparse strip-runfill + ScatterND compose** | `prototypes/build_task054_v5.py` + `task054_v5.onnx` + `prototypes/proto_v3_numpy.py` | **cost 47,180 -> 27,690** (`score_network_v2=(0, 27,505, 185)`, -19,490 = -41.3%). Ladder: 30,247 -> 28,449 -> 28,388 -> 27,690. GATES: `ng_gate visible` 266/266; **full Path-A AB vs live submission.zip parent 200,266/200,266 byte_equal=True** (`gate_ab_full_v5.json`); numpy proto 266/266 + 20,000/20,000 fresh ARC-GEN. ARCHITECTURE: output = INPUT CANVAS + SPARSE DELTAS. Kills v2's 4x full-grid CumSum runfill (~27,900): lines run on EXTRACTED `[4,30]` strips written back via ScatterND row-writes; stamps + legend-erase = ONE flat 45-point ScatterND with a per-offset Where value-chain; marker locate = dual-path TopK over row/col presence vectors + plane-gather ArgMax for the derived axis. Staged `.hermes/board/staged_submits/task054_v5_sparse_scatter_t_7b3a90e4_on_47180/`. hours_spent ~2.5
- 2026-06-04 | worker-7 **v7 V-line ScatterElements (kill both transposes)** | `prototypes/build_task054_v7.py` + `task054_v7.onnx` | **cost 27,690 -> 26,043** (`score_network_v2=(0, 25,849, 194)`, -1,647). GATES: visible 266/266; truth 2,000/2,000; full 200k AB byte-equal (`gate_ab_full_v7.json`). CHANGE: V-lines drawn by gathering COLUMN strips of the H-result canvas (`Gather axis=3 -> [1,1,30,4]`), runfill along axis 2, columns written back via `ScatterElements(axis=3)` with INT32 `Expand`-ed index — eliminating the two `Transpose` planes (cT, cV = 1,800) for +480 (idxV). This is the v7-lineage 100%-truth memory floor (25,849).
- 2026-06-20 | worker-054a **v8 params-only constant de-dup** | `prototypes/build_task054_v8_params_only.py`; staged `staged_submits/task054_v8_params_only_cost26003/` | **cost 26,043 -> 26,003** (`score_network_v2=(0, 25,849, 154)`, -40 params). GATES: visible 266/266; truth 200,000/200,000; full AB 200,266/200,266 byte_equal=true; SHA256 `17aa8435bc2c3420e53f03b5dee1ef1b530384981629ff02206d901ac8072a85`. CHANGE: memory DAG unchanged; reuse duplicate INT64 constants, route dummy fallback to flat index 0, `Expand(bg1,[13])`, `Unsqueeze` instead of reshape shape init. **v8 = the 100%-TRUTH cost floor (byte-equal to the 100% submission.zip parent).**

> **2026-06-22 — A DIFFERENT-WORKTREE WORKER SHIPPED v25 (cost 22,674).** Team-best at `../K-NeuroGolf-OnnxlyFans/.../Team_Best_Onnx/task054.onnx` (graph name `task054_v25`, owner Aditya) is a polynomial-moment rewrite NOT in this repo's prototypes: it replaces the v7-lineage's 8 detection planes with **Einsum row/col moments + Gemm/Pow Vieta-root marker detection (poly_basis) + TopK** to find up to 4 markers, keeping the strip CumSum runfill + a 41-point ScatterND stamp. `score_network_v2 = (0, 22274, 400)` = cost 22,674. This broke the file's ~25,849 "definitive floor": the einsum route DOES pay once the canvas+line machinery is also reworked.

- 2026-06-24 | worker w01 **v26 dead-`sp`-pipeline prune of team-best v25** | `prototypes/task054_v26_no_sp.onnx`; staged `staged_submits/task054_no_sp_deadprune_cost13454/` | **cost 22,674 -> 13,454** (`score_network_v2=(0, 13,102, 352)`, -9,220 = -40.7%). SHA256 `5bbf53b7c5993090ff3fa08bbca4d330e139328d6589a1bca16e7f4620184e75`. GATES: visible 266/266; **full AB vs team-best v25 = 200,266/200,266 byte_equal=True** (GOLD Path-A pure dead-code compression); truth 199,827/200,000 (99.913%); profile_real_cost `--arcgen 266` = 13,454 (static==profile). CHANGE: deleted the entire `sp_` small-grid fallback pipeline (85 nodes / 16 inits, incl. F32 `sp_valid_sum` 3,600 + 8 U8 planes) + selector `Where(sp_small, ...)`; routed `final2d -> Equal -> output`. **WHY DEAD:** `sp_small=(active<899.5)` fires only sub-30x30; generator hard-codes size=30. Inherits v25's exact 173-miss set (99.913% truth).

## Cost analysis — full historical breakdowns (current floor summary is in task054.md)

2026-05-21 v4 analysis:
- Current canonical 178,511 is dominated by scalar full-grid tensors: many F32 `[1,1,30,30]` at 3,600B each plus many BOOL masks at 900B.
- F16 scalar pipeline (opset 14) cuts most numeric full-grid tensors in half. Keeping the first color-index Conv in F32 and casting only `v29` avoids an 18,000B full-input F16 cast.
- Moving color-selection reductions/final assembly to UINT8 saves ~12.6k versus F16 grids (ORT-valid at opset 14).
- Remaining floor ~72 full-grid BOOL/UINT8 masks + 16 F16/F32 full-grid tensors; <60k needs algorithmic reduction of the bool shift/And/Or mask network.

**2026-05-30 v1 true-rule floor decomposition (worker-7).** v1 @ 55,448 breaks down:
- **Lines (4-direction run-fill): ~28k (≈half).** Each direction = CumSum(F16 1800) + cast-u8 + Where-gate + MaxPool-prefixmax + Sub-band (4×~5.4k) + 2 Max/Greater + ORs. The line spans a flag's contiguous row/col run; `flip`/`xpose` can stack flags along either axis, so a cheap "box∧row-has-marker" broadcast is WRONG (69% ARC-GEN). 4 CumSum F16 = 7,200 alone.
- Color canvas (F32 Conv 3,600 + u8 900) = 4,500 — F32 Conv irreducible.
- Masks (box/box_dil/special/markers/box_region/legend/markval) ~12k.
- Legend centroid-read ~3k. Surround stamp + compose (4 Where) ~6k.
- **Realistic floor for THIS rule ≈ 50-55k.** (Superseded by the poly-moment route; kept for the run-fill decomposition reasoning.)

2026-06-02 v2 update: exact graph surgery trimmed whole planes to **47,180**. Removed full-grid `box_region`, marker-color reduction, one line-gating mask, the `special` BOOL plane, old F16 centroid/color-flat path, a full surround-present mask, the center scalar gather, three scalar `Not`s, 24 bytes of INT64 plumbing. Final v2 breakdown @ 47,180 (`memory=47,146`, `params=34`): color 4,500; 9×900 detection; line primitive ~27,900; compose/stamp tail 6×900; scalar plumbing ~300.

**2026-06-04 v5 (v5d) tensor breakdown @ 27,690** (`memory=27,505`, `params=185`): color4f F32 3,600 + color4u 900; 8×900 detection planes (plane-conservation law); canvas/scatter chain scat_h/cT/scat_v/cV/flatV/pflat/final4d 7×900; strip line blocks ~3,050(H)+3,300(V); locate+bbox+plumbing ~3,300; params 185.

## Cost floor — DEFINITIVE proof (2026-06-04, worker-7) — SUPERSEDED 2026-06-24, RETAINED for the probe method only

> The "definitive" floor below was FORMULATION-RELATIVE, not absolute: it proved a floor of the v7-lineage 8-detection-plane architecture. The team-best v25 (Einsum/Gemm poly-moment detection) sidesteps the 8 planes entirely (memory 22,274), and v26 prune reaches memory 13,102 / cost 13,454 — about HALF the "definitive" 25,849. Lesson: a probe-backed "irreducible plane" count is only irreducible for the decomposition measured.

Empirical floor probes (scored):
- `output = Greater(input, 0.5)` (one-hot passthrough, ZERO intermediates) = **cost 1**.
- color canvas + Equal only (Conv->U8->Equal) = **cost 4,520** (color4f 3,600 + color4u 900). Any approach materializing ONE full plane pays >=4,500.
- detection only (8 planes, no lines/stamps/scatter) = **cost 11,856**.

Source-read of `calculate_memory`: memory = `max(static_infer, runtime_trace)` per intermediate, summed; `input`/`output` free; symbolic `dim_param` -> `return None`. For STATIC graphs local `_score_v2` == Kaggle exactly. No accounting exploit available (ConstantOfShape/sparse-init/uint4 help params only, already tiny).

## Don't-do log — older superseded entries (current/active entries stay in task054.md)

- worker-7 2026-05-19: task054 always 30x30 (all 4 examples). Earlier "no viable optimization / skip this task" verdict — WRONG (later sessions reached 13k). Retained as a caution against premature walls.
- 2026-05-21: blanket F16 at opset 11 fails ORT load (`CumSum` rejects F16). Opset 14 required for F16.
- 2026-05-21: Rule 12 compare-after-reduce not applicable; the four `ReduceMax` consume masked color grids, not `Greater->Cast` masks.
- 2026-05-21: full-input F16 cast costs 131,635 (`input_f16` = 18,000B). Cast after the first color Conv.
- **2026-05-30 (worker-7): the prompt's whole premise was wrong.** task054 is NOT a "same-position context-recolor / denoise"; there is NO "10-channel And/Where collapse to 1 channel." Don't chase the sub-20k/floor-3000 target from that framing.
- **2026-05-30 (worker-7): parent is HIDDEN-CORRECT, compression-only (NOT an unlock).** A rewrite must PRESERVE the parent's function (byte-equal), not "fix" it — for byte-equal candidates. (See the 2026-07-11 more-correct-than-live exception in the live file.)
- **2026-05-30 (worker-7): cheap horizontal-line via "box_region ∧ row-has-marker" is WRONG** (69% ARC-GEN). After `xpose`, flags share rows so the broadcast leaks. Both H and V lines require the contiguous CumSum run-fill.
- **2026-06-02: `ReduceSum(UINT8)` legend count-subtraction is SCORER-REJECTED** (`score_network_v2` returns None). Keep `legend_u8 -> ReduceMax`.
- **2026-06-02: `CumSum(UINT8)` is SCORER-REJECTED.** Keep F16 CumSum + U8 post-cast.
- **2026-06-02: ONNX `If`/`Loop`/`Scan` scorer-banned** for scalar-gated line dispatch.
- **2026-06-02: static legend edge-zone crops are not exact** (contaminated by marker centers). Keep `special - markers` legend isolation.
- **2026-06-02: orientation shortcuts need filled-flag logic + a visible 3-box fallback.** Raw `box` components are polluted by marker-center holes; use `box|markers`.
- **2026-06-02: sub-20k probes only work by deleting required line behavior.** `task054_no_line_probe.onnx` scores 19,308 but 0/266 visible; `task054_broadcast_probe.onnx` scores 23,927 but 96/266 visible.
- **2026-06-04 (worker-7): plane-count CONSERVATION.** Every algebraic reformulation of the detection planes lands exactly 8×900. The DAG {!=bg, !=box, special, box01, dil, markers, legend} + 1 source is a derivation floor.
- **2026-06-04 (worker-7): einsum channel-classify does NOT pay** on the v7 plane layout (net ~-220, formulation-relative — v25 later proved it DOES pay with a fully reworked canvas). 
- **2026-06-04 (worker-7): boundary-ArgMax band variant breaks multi-marker strips.** DERIVED-axis strips can hold 2 markers; both blocks need the multi-marker-safe CumSum form.
- **2026-06-04 (worker-7): per-strip onehot exclusion loses centers on duplicate scatter rows.** The onehot must be the MULTI-marker indicator (plane-gather `markers[row]`).
- **2026-06-04 (worker-7): TopK INDICES output is a tie-order trap.** Use the VALUES as coordinates (absent slots -> 0).
- **2026-06-04 (worker-7): stamp writes must source EXISTING content** for no-surround/inactive-dir offsets (Gather from post-line canvas), not boxcolor. Official train[1] has same-box markers 3 rows apart.
- **2026-06-04 (worker-7): v6 2-D/4-D scatter restructure REGRESSED to 30,299.** Flat `[N,1]` indexing is structurally cheaper than 2-D/4-D on this scorer.
- **2026-06-04 (worker-7): the "notbox fold" saves ZERO planes.** 8 detection planes is the proven minimum.
- **2026-06-04 (worker-7): one-hot-native (no color canvas) does NOT beat the index canvas.** Every full-plane detection step needs an F32 channel read (3,600, same as the Conv).
- **2026-06-04 (worker-7): no scoring-accounting exploit is available** (source-read of `calculate_memory`).
- **2026-06-04 (worker-7): techniques sec40 plane-breaker catalog — NO breaker applies** to the v5d graph.

### 2026-05-20 - CL_HUGE worker
- Graph output is BOOL (Equal node), 1 byte/elem, drop-cast trick already applied.
