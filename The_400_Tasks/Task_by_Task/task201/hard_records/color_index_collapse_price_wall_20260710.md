# task201 — color-index-collapse refreshed DROP (2026-07-10)

**Manifest/live bar:** 1773, SHA
`b099df389ec35e1725311ea3174ecf8f1d57fbf0c27d8e0211548fe0c3f920ae`
(`python tools/ng_current_bar.py --task 201 --json`; manifest authoritative and workbook agrees).

**Lower documented team bar used for reviewer readiness:** 1765, SHA
`03b0d82f44000be9ddd2403934a90b19e650f800a63b152cff8353e51407c7c5`
(`../K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/task201.json` plus paired ONNX).

## Attempted route

Refreshed the prompt's ArgMax/index-plane fast probe using the already-built, stronger
task-local fused implementation:

`prototypes/task201_v2_cropout_boolpad_opset20_edge_upperclamp_tvals23_int32_dilatedconv.onnx`

This graph uses a dilated Conv to collapse and crop the free one-hot input directly to
`color13_f32[1,1,13,13]`, casts it to a U8 color-index plane, performs scalar-index spatial
logic, and expands only at the boundary. A literal ArgMax is strictly worse because ONNX
ArgMax emits INT64 (7200 B at 30x30; 1352 B even at 13x13, before its source crop).

## Exact failed gate

Visible retry: `[visible] 266/266 (100.00%)  PASS`.

Price/profile gate:

`[score_v2] ...: macs=0 memory=2879 params=136 cost=3015`

`task201 ...: static=3015 profile=3015 (memory 2879 + params 136; 266 real inputs)`

Therefore the route is **DEAD on price**: +1250 versus the effective 1765 team bar,
+1242 versus manifest/live 1773, and +2295 versus the prompt target 720. No truth/AB
gate was run in this pass. The exact SHA already has historical task-local proof of
truth-full 200000/200000 and AB-full 200266/200266 byte equality, but repeating those
full gates on a losing graph would be pure burn.

## Why the formulation is dead

The charged component named by the refreshed direction is already gone from the current
row-code graph. The attempted family reintroduces `color13_f32` (676 B), `color13` (169 B),
and multiple 13x13 rule masks. The first two tensors alone total 845 B, already above the
entire 720 target before bbox, orientation, rendering, output decode, or parameters.
`ng_graph_cost_diff` names the largest additions as `color13_f32` +676 B,
`onehot_o` +560 B, and four 13x13 U8/BOOL planes at +169 B each.

## One remaining hypothesis

A genuinely different operator-level rewrite could try to consume packed row codes and emit
the final one-hot output directly, fusing away several 56-byte 7x8 render tensors. That is
not a color-index-collapse variant and needs a new rank/operator proof; it is unlikely to
bridge the current 1765-to-720 gap. Do not rebuild ArgMax/Conv color-index variants.

Machine-readable packet:
`../prototypes/task201_color_index_collapse_probe_20260710.manifest.json`.
