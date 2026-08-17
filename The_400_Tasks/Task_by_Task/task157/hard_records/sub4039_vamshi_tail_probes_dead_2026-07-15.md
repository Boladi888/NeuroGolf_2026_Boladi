# Hard record — sub-4039 Vamshi tail probes are dead (bo_ng_worker1, 2026-07-15)

## Live bar / state
- Authoritative bar: **4039**, owner Vamshi, sha `14f7ecf0fa38c088cdaeed1d5f142bcf04082d4d54600159f2e7e05fa3a7cc6a`.
- Source: `python tools/ng_current_bar.py --task 157 --json` (manifest authoritative, updated `2026-07-12T15:15:54Z`).
- Real-input profile of current `Team_Best_Onnx/task157.onnx`: `static=4039 profile=4039 (memory 3847 + params 192; 265 real inputs)`.
- Effective staging rule this pass: only a profile cost **<4039** can be reviewer-ready.

## Probes attempted

### 1. Final Pad full-rank pads probe
- Builder: `prototypes/build_task157_vamshi4039_tail_probes.py`.
- ONNX: `prototypes/task157_vamshi4039_padfull_probe.onnx`.
- SHA-256: `2db796572ef1a6c762ff3abb7116b48169244ef941506a963545d5d3d6c5b165`.
- Route: replace final Pad axes form `pads[6] + axes_123[3]` with full-rank `pads[8]`.
- Profile gate: `task157 task157_vamshi4039_padfull_probe.onnx: static=4041 profile=4041 (memory 3847 + params 194; 265 real inputs)`.
- Visible gate: `[visible] 265/265 (100.00%)  PASS`.
- Cost diff: `NET COST DELTA +2`; the shared slice axes initializer is still needed, so full-rank Pad only adds two params.
- Verdict: **REJECT_LOCAL / no-gain**. Correct but strictly worse than the 4039 bar.

### 2. Quadratic QLinearConv final decoder probe
- Builder: `prototypes/build_task157_vamshi4039_tail_probes.py`.
- ONNX: `prototypes/task157_vamshi4039_qdecode_tail_probe.onnx`.
- SHA-256: `23e2cfdd271e281065d7174ec551f6ee2b4ccf6c88fd934195d6386de2ffeab5`.
- Route: replace `Equal(label, channel_ids) -> small_b -> Pad -> output` with `{label,label^2} -> QLinearConv -> output`.
- Profile gate: `task157 task157_vamshi4039_qdecode_tail_probe.onnx: static=4061 profile=4061 (memory 3847 + params 214; 265 real inputs)`.
- Visible gate: `[visible] 0/265 (0.00%)  FAIL`.
- Static graph diff: `small_b` (-450B) is exactly replaced by `label_sq` (+150B) + `label2ch` (+300B), then decoder params add +21. Real profile does not expose a hidden saving.
- Verdict: **REJECT_LOCAL / visible-fail and no-gain**. Per visible-fail axiom, do not run truth/AB/battery on this representation.

## Separability sanity check
- Stored ARC-GEN rank sample over visible `task157.json` outputs: color-mask ranks on the 10x15 crop reached rank 5 (`arc-gen` ranks `{1:1, 2:71, 3:308, 4:111, 5:33}` over first 262 stored rows).
- `python tools/arc_gen_enrich.py --task 157 --num 20 --dry-run` confirmed ARC-GEN availability and would append 20 fresh pairs without writing.
- This matches techniques #109's rank/runtime-selector kill condition: the output is not a low-term fixed separable row/column product; selectors are runtime-decoded component placements.

## Why this route is dead
- The current Vamshi graph already sits on the #81 small-grid side: decoded `[1,3,10,15]` BOOL one-hot (450B) plus free final Pad beats a full `[1,1,30,30]` value canvas.
- The obvious free-output decoder swap does not reduce memory; it only changes which 450B pre-output surface is charged, then adds params and fails visible.
- The final Pad axes form is already minimal because `axes_123_i64` is shared by the input Slices.

## Remaining hypothesis
Only a matcher-side rewrite can plausibly beat 4039: one of `scores4_0/1/2` (240/180/120B), the F32 input slices (240/180B), or the component weight chain would have to disappear or shrink while preserving Vamshi byte behavior. Tail-only rewrites are exhausted by this probe.
