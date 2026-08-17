# DONE 2026-05-22T204400Z — task383 directional Pad/Slice → MaxPool windows (+1.01 LB CONFIRMED)

## Result

- Builder: `The_400_Tasks/Task_by_Task/task383/prototypes/build_task383_bool_u8_rewrite.py`
- Staged: `.hermes/board/staged_submits/task383_bool_u8_maxpool_t_80ce49cb_on_current/`
- ONNX SHA256: `80CE49CBAC62994EE6AF199DF451409A27CF5C9518661E8C12FA9CECC9FDA96B`

Cost: 152,592 → 55,467 (memory 55,418 + params 49). Δ = **−97,125 bytes (−63.6%)**

Validation:
- `verify_against_live --tid 383`: 266/266 proto-vs-LIVE
- `adversarial_test --tid 383`: 5852/5852 (100% across all perturbation types)
- Synthetic official-style random one-hot raw LIVE comparison: 100/100

**Kaggle submitted 2026-05-22T20:43Z (single-task wildcard, lineage base 5874.78)**:
- Score: **5874.78 → 5875.79 = +1.01 LB CLEAN MATCH** (predicted +1.01 LB raw log_e)

## Technique — directional Pad/Slice → MaxPool window

The LIVE task383 graph used directional Pad+Slice probes to test "is there a foreground cell at distance K to the {north, south, east, west}?". Each direction × distance was a separate Pad+Slice pair, producing F16 [1,1,30,30] intermediates per probe.

The replacement: **a single MaxPool with asymmetric kernel + pad = the same predicate, in BOOL → UINT8 dtype**.

Example — "is there a foreground cell within distance K to the EAST?":
- Old: `Slice(input, axis=3, start=K, end=30) → Pad(left=K, right=0) → cast → mask` (4 ops, [1,1,30,30] F16 intermediate)
- New: `MaxPool(input_b_u8, kernel=[1, K+1], pad=[0,K, 0,0]) → bool` (1 op, [1,1,30,30] UINT8 intermediate)

The mathematical equivalence: MaxPool with right-pad K and kernel width K+1 sees, at position (r,c), the max over cells (r, c..c+K). True iff any of those K+1 cells (including the cell itself) is set. Equivalent to the Pad-Slice probe.

**Generalizes to** any task with directional "is there a marker at distance ≤K in direction D" probes:
- Replace each directional Pad+Slice chain with a single MaxPool with asymmetric kernel + pad
- Keep mask logic in BOOL/UINT8 (Rule 11 dtype: opset 12+ unlocks UINT8 MaxPool)
- Verify: 100% byte-exact against LIVE on visible+adversarial+synthetic-LIVE
- LIVE-equivalent (not algorithm replacement) — same predicate, different graph plumbing

## Why it shipped clean

task383 has documented Kaggle-negative history: `task383_v2_visible_derived_axis_cross_out` matched 266/266 visible but lost -12.31 LB on Kaggle public LB. That candidate was an **algorithm rederivation**.

The +1.01 LB win here is a structural rewrite: same predicate semantics, different plumbing. No new algorithm.

## Sorter note: techniques.md §1 — add as Rule "directional_padslice_to_maxpool_window"

Pattern name: `directional_padslice_to_maxpool_window`. Applies when graph has multiple directional Pad+Slice probes computing "is there a marker at distance ≤K in direction D". Replaces N×(Pad+Slice) chain with 1×MaxPool(asymmetric kernel + pad), keeping mask logic in UINT8. Empirical: task383 saved -97,125 bytes (+1.01 LB Kaggle confirmed 2026-05-22).

## Sorter note: by_symptom.md — add entry "directional Pad+Slice chain for marker proximity"

Symptom: graph has 4-8 Pad+Slice pairs per direction × distance. Suggests directional proximity probes. Link to the new technique + this task383 exemplar.

## Sorter note: exemplars/01_whole_graph_rewrites/task383_bool_u8_maxpool.md — create new exemplar

Full writeup with before/after node count, ONNX surgery walkthrough, eligibility checklist (any task with directional Pad+Slice chains computing same-direction proximity). Cross-reference to traps.md §13 (visible-correct vs hidden-correct).
