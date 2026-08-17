# task173 hard record — CarrierNarrowEqualTail route (DROP)

- **worker / date**: bo_ng_cld_worker3 (Claude lane), 2026-07-10.
- **bar used (refreshed live)**: `python tools/ng_current_bar.py --task 173 --json` →
  authoritative **manifest live_cost `10853`** (owner Sho, sha256 `0456dd75…`, updated 2026-07-09T01:09:17Z).
  Effective best to beat = **local staged ceiling `8710`**, sha256
  `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`
  (`staged_candidates/task173_regionfillrewrite_cost8710/`, byte-exact refactor of the 8829 oracle,
  gold ship class). Leaderboard note fresh: public 7559.16 (2026-07-11). Stage condition: profile cost `<8710`.
- **prompt target**: ~900 (false-floor "collapse 10-channel/F32 spatial carrier to U8/color-index before
  Conv/Reduce/Where + Equal-tail; delete charged carrier/canvas/template").

## Attempted route(s) and exact failed gates

**(1) The named charged component is ALREADY GONE from the live graph.**
`largest_intermediate.py` on the staged 8710:
```
  1   3,600  51.7%  F32  [10,10,3,3]   ScatterND  dynamic_w    (render WEIGHT, not a spatial carrier)
  2   2,116  30.4%  F32  [1,1,23,23]   Conv       detect_map   (already a 1-channel plane)
  (everything else ≤432 B scaffolding; total intermediate 6,963 + params 1,747 = 8,710)
```
First node is `Conv(input, detect_w) -> detect_map [1,1,23,23]`: detection reads the **free** F32 input
directly and emits a 1-channel map. There is **no 10-channel spatial carrier intermediate**, and no
`ScatterElements`/`QLinearConv` carrier (those exist only in the obsolete v21/local-live). The Direction
narrows a tensor that the team-best does not pay for. (Matches `rankchain_dce_scatter_qlinear.md`.)

**(2) Fast probe — bare carrier + Equal-tail (BUILT + MEASURED).**
`prototypes/carrier_narrow_fastprobe.onnx` = `Conv(input,[1,10,1,1]) -> carrier[1,1,30,30]` (color index)
then `Equal(carrier, colors[10,1,1]) -> output[1,10,30,30] BOOL` (free graph output). No detect/render.
- `largest_intermediate`: sole charged tensor `carrier [1,1,30,30] F32 = 3,600 B` (+~20 params ≈ 3,620).
- `ng_gate visible`: `[visible] 0/266 (0.00%) FAIL`.
- **Both kill-rule conditions fire**: price 3,600 ≫ 900, and falsifies. Worse, the 3,600-B carrier is a **net
  ADD** over the current no-carrier 8710 graph — you still owe detection+render on top. A U8 carrier can't
  escape it: the first op reading the F32 input outputs F32 (Conv → 3,600 B) or pays the ≥9,000-B
  F32→U8 QuantizeLinear/Cast tax; there is no single op giving U8 `[1,1,30,30]` from F32 `[1,10,30,30]`.

**(3) Two-stage split render (genuinely new lever, priced dead).**
Split the fused `dynamic_w [10,10,3,3]` Conv into (cheap dynamic 1×1 color-map `[10,10,1,1]`=400 B) +
(fixed 3×3 depthwise spread `[10,1,3,3]`=90 params). Either ordering materializes a `[1,10,30,30]` F32
**inter-stage canvas = 36,000 B** (only the final `output` is free) — worse than the 3,600-B fused weight by
+32,400 B. The fused single-Conv render is optimal precisely because it fuses color-map + spatial spread
with no intermediate canvas.

## Why dead (forced-floor argument)
- **Render floor = 3,600 B.** The fill is a data-dependent 10→10 color-changing 3×3 Conv. F32 forced
  (weight dtype must match F32 input; F16 needs an 18k-B input Cast). Dense forced (Conv needs a dense
  weight; sparse (C,F) can't feed Conv). 10×10 forced (10 colors in/out; grouped/depthwise can't move C→F).
  3×3 forced (fill reaches r±1,c±1). No cheaper op class: split→36k canvas; Einsum→8,949 lb; ScatterND→
  dynamic-shape cost fiction. **Target ~900 < this single tensor (3,600).**
- **Detector floor = 2,116 B.** Full-grid F32 Conv, k=8 23×23 proven minimal (k=9 → visible 262/266 FAIL,
  `detector_k9_visible_fail.md`). U8 pair-code detector alone lower-bounds at 12,237 (`paircode_direct_
  renderer_lower_bound.md`). Carrier-first adds 3,600 before detecting (probe above).
- Render 3,600 + detect 2,116 are **independent and forced** → they dominate; +scaffolding+params ≈ 8,710.

## Verdict
**DROP / WALL at 8710.** The charged component named by the Direction does not exist in the live graph; the
carrier-narrow route is a net ADD (fast probe 3,600 B, visible 0/266); the ~900 target sits below a single
forced tensor. This confirms — from three independent formulations (named carrier, two-stage split, plus the
4 prior records: k9 detector / free-output Einsum / pair-code U8 detector / rank-chain DCE) — that 8710 is the
byte-exact floor for this task's cost model.

## One remaining hypothesis (for a future batch — not this route)
A render op class that is data-dependent yet materializes <3,600 B AND writes the one-hot output for free
without a full `[10,10,3,3]` F32 weight or a `[1,10,30,30]` canvas — none found across Conv/Einsum/Scatter.
Absent such an op, do not re-open carrier-narrow / U8-detector / dtype-narrowing routes: the F32-input tax
prices every one of them above 8710.

## Note for captain (out of task scope, flagged only)
Manifest live is Sho's `10853` (sha `0456dd75…`) while the validated gold-class `8710`
(sha `27f31830…`, byte-equal to the 8829 oracle, truth 20000/20000) is staged locally and −2,143 B better.
The 8710 appears not yet promoted into the submission manifest. Promotion decision is the captain's.
