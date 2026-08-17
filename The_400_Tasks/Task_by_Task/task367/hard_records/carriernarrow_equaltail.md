# HARD RECORD — task367 CarrierNarrowEqualTail (DROP / false floor)

- **Date**: 2026-07-11 · worker `bo_ng_cld_worker2` (Claude lane)
- **Live bar (sha)**: cost **15,088** (manifest, `ng_current_bar`, owner Sho, updated 2026-07-09) · onnx `Team_Best_Onnx/task367.onnx` sha256 `fa6f2f8b7e50af97ae06664c43983f11f1669be7aed3ab5da6737209dd557c7c` · `_score_v2` recompute = memory 14,500 + params 582 = **15,082**.
- **Prompt target / "floor_est"**: ~900 (gap ~16.8×). Direction: "carrier-narrow sweep — collapse 10-channel/F32 spatial carrier to U8/color-index before Conv/Reduce/Where, restore w. Delete charged carrier/canvas/template."

## Attempted route
CarrierNarrowEqualTail: narrow the 10-channel F32 spatial carrier to a single U8 color-index plane and render the fill with a separable Equal-tail, aiming at the ~900 forced floor.

## Why it is dead
1. **Charged component already eliminated in live.** Sho's 15,082 graph (`inspect_graph.py`) is already the carrier-narrow / forced-floor architecture: node0 `Conv Wp[1,10,1,1]` collapses the [1,10,30,30] F32 one-hot → `pf` F32[1,1,20,20] (negpad crop 30→20) → `Cast` UINT8 `p`. Per the 2026-07-10 workflow update, "if [the charged component] is already gone... return DROP with the refreshed floor note." It is gone.
2. **The ~900 single-plane floor is priced/falsified.** Fresh-seed fast probe (`prototypes/probe_task367_carriernarrow_equaltail_fastprobe.py`, seed `3672026071100`, 3000 fresh ARC-GEN pairs):
   - cheapest separable single-plane rule (nearest-gray-wall ≤5 in 4 dirs + all-zero rect, ±1-D guard) = **1566/3000 (52%)** → over-fills connector pockets (first_fail grid 2).
   - fill mask **rank>1 in 99.2%** of fill-grids (real & GF(2), mean 2.9, max 4). A single `Equal(row_pred, col_pred)` render is rank-1 → **#109**: a 2-D region (union of 2–4 rectangles) cannot be one separable Equal render; it forces a canvas / multi-channel detector.
   - even the exact numpy reference is 2997/3000 on fresh — connector edge-cases are subtle (non-separable continuation guard).
3. **Forced-floor components** (both non-collapsible): (a) local windowed **seed detector** — box-vs-connector discrimination is non-separable (corroborated by exterior-flood 108/266, nearest-wall 117/266, corner-imos 35/266, CC-rect needs flood/CC machinery); Sho's `v_main` [1,10,20,20] U8 = 4,000 B. (b) **bounded 5-step propagation** (`r1..r5`, 3×3 Conv + `Min` wall-clamp), forced by max interior size 5; ~4,000 B. Together they dominate the 14,500-B memory.

## Exact failed gate / measurement
- `ng_gate visible --task 367 --cand Team_Best_Onnx/task367.onnx` → `266/266 PASS` (live is the correct reference).
- fast probe A/B separable single-plane → `1566/3000` on fresh ARC-GEN (kill-rule: falsifies).
- No candidate built below 15,082; no truth/ab/battery run (nothing to gate — priced wall reached at the probe stage, per "one crack only, stop on a priced wall").

## One remaining hypothesis (fragile, deprioritized)
A rung-shave — fewer than 10 detector channels, or coarser (5×5, 2-cell/step) propagation with fewer iterations — could in principle reach ~13–14k, but requires reverse-engineering Sho's quantized `Wm`/`Wc` weights and re-proving 200k byte-equality, carries real connector-leak risk, and the 2026-07-10 workflow update deprioritizes rung-epsilon shaves on this false-floor prompt. Not a floor route; do not grind under this prompt.

## Forced 2nd pass (2026-07-11) — gated forced-component confirmation
Pushed harder per the review mandate; decoded Sho's exact mechanism and gated genuinely-different assemblies:
- **Mechanism**: `qs=1, zu=zi=0` → all QLinearConv are clamped integer convs. `Wp=[1,0,0,0,0,2,…]` → index plane (1=bg,2=gray); `c5=p>>1`=is-gray; `v_z=p%2`=is-bg. Body is a **bounded exterior flood**: detector `v_a2` seeds a flood, plus-kernel `Wc=[[0,1,0],[1,1,1],[0,1,0]]` dilates 5×, each `Min`-clamped to bg; `v_fill = v_z > r5c` = bg cells the flood can't reach = sealed box interiors. The 10-ch ReLU detector seeds connector false-pockets so they flood (not filled).
- **Detector nonlinearity is real (measured, 266 official)**: pre-clamp `v_main` range [−7,2]; the UINT8 ReLU-at-0 clamp fires on **all 10 channels** (13k–56k cells/ch of 59k). So `Wm→Wcol` is a genuine 2-layer quantized ReLU net — **cannot be folded**; `v_main` [1,10,20,20]=4,000 B is forced.
- **Flood depth is corpus-forced (gated)**: rewired the flood tail to N dilations (`build_task367_flooddepth_variants.py`). flood5 sanity = 266/266 @ 15,082 (validates rewiring == live); **flood4 = 243/266 visible-FAIL @ 14,282**; flood3 = 206/266 @ 13,482. A visible miss = certain hidden 0 (§VISIBLE-FAIL) → 5 steps minimal; did NOT run truth/ab on the visible-fail variants (pure burn).
- **Per-step floor**: library l.3619 — a flood step is `Min(dilate,mask)` = 2 planes; Sho is at it. l.2407-2412/4701 — geodesic step count can't be shortcut. l.1847 — 1-ch U8 flood ~2 planes/step confirmed.
- **Front-end bridge forced**: no single ONNX op yields U8 from F32 input without a ≥9,000 B [1,10,…] intermediate; the single F32 index plane `pf`=1,600 (cropped 20×20, min spatial) + `Cast`→U8 is the cheapest bridge.

**Forced-component floor**: front-end 2,000 + carrier/zmask 1,600 + detector 6,000 + flood 4,000 + tail-pad 900 = memory 14,500 + params 582 = **15,082**. Every term pinned by measurement or gate.

## Verdict
**WALL (was DROP).** Cheapest correct formulation found = live **15,082** (5-step bounded exterior flood + 10-ch ReLU seed detector); its profile == static == 15,082 (score_v2 exact, no grader gap). No sub-15,082 correct formulation exists on this route: the ~900 floor_est is false, the charged 10-ch F32 carrier is already eliminated in live, and every dominant component is forced (F32→U8 bridge; active-ReLU 10-ch detector; 2-plane × 5-step geodesic flood; output-shape tail-pad). The only residual slack — fewer detector channels — requires reverse-engineering + retraining Sho's tuned quantized net with re-proven 200k byte-equality; not reachable by this route and deprioritized by the 2026-07-10 false-floor workflow.
