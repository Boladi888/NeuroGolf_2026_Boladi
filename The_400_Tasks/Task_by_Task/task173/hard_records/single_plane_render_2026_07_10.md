# task173 hard record — single color-index plane render (forced 2nd pass, DROP/WALL)

- **worker / date**: bo_ng_cld_worker3 (Claude lane), 2026-07-10 (second, forced pass).
- **bar used (refreshed live)**: `ng_current_bar.py --task 173 --json` → manifest live_cost **10853**
  (Sho, `0456dd75…`, 2026-07-09); effective best ceiling **8710** (`27f31830…`). Leaderboard fresh
  (public 7559.16, 2026-07-11, via BoBrain provider context task173). Stage condition: profile `<8710`.

## New diligence this pass (not done in pass 1)
- **Read the ARC-GEN generator** `external/ARC-GEN/tasks/task_72322fa7.py` (ground-truth distribution):
  - **4 sprite types**: X(corners), plus(+, 4 orthogonal), horiz(−), vert(|). 1–3 distinct per grid.
  - Each sprite type draws a **unique** (color0, color1) from `shuffle(range(1,10))` ⇒ up to **3 distinct
    (C,F,shape) triples**, all-distinct colors ⇒ scalar-CF/global-ReduceMax is provably insufficient;
    **TopK-3 is forced** (matches the current arch + the v22/v23 "scalar-CF BROKEN" don't-do).
  - **Bidirectional fill**: `megashows` ms=0 = complete template (teaches C,F,shape); ms=1 = center-only
    (fill FRAME with F); ms=2 = frame-only (fill CENTER with C). The render must move color BOTH ways
    across a 3×3 neighborhood ⇒ a cross-channel 10→10 map, not a one-directional stamp.
- **Mined library**: `#53` — "the `input` is NOT counted; always Conv/Reduce directly on it; **never
  pre-cast/pre-crop the 10-channel tensor — a working tensor is a net loss**" (= the F32-input tax, confirms
  pass-1 carrier verdict). `#54` — runtime-weight Conv IS the documented render lever; "all intermediates
  tiny **except the unavoidable match map**" (= detect_map is forced).

## Genuinely-different formulation tested + gated (dominant intermediate = a [1,1,30,30] plane, not a [10,10,3,3] weight)
**Single color-index plane via one runtime `Conv[1,10,3,3]` over the free input + Equal-tail** — the "~900
dream" (one narrow plane that IS the answer). Built `prototypes/single_plane_render_probe.onnx` +
`bo_ng_cld_worker3_single_plane_render_probe_2026_07_10.py`.
- **Correctness (numeric, ORT-equivalent numpy on a controlled grid)**: a summing Conv-to-plane is
  **structurally wrong**. At a COMPLETE plus template (C=6,F=8) the center sums `C + 4·C = 30` (not 6) and a
  frame cell sums `16` (not 8); `Equal(plane,[0..9])` of 30/16 is **all-false ⇒ cell dropped to BG**. A Conv
  ADDS contributions; the current `[10,10,3,3]` one-hot render is correct only because each output channel is
  an independent BOOL-OR resolved by the final Equal/Max — a single scalar plane cannot disambiguate counts.
- **Cost (measured)**: `largest_intermediate` → sole tensor `plane [1,1,30,30] F32 = 3,600 B` = **identical**
  to the `dynamic_w [10,10,3,3]` (3,600 B) it would replace. No saving, and the detector (2,116 B) is still
  required on top ⇒ net **≥ 8710**, with a broken render.
- ConvTranspose-stamp variant = same runtime `[10,10,3,3]` = 3,600 B (transpose of the same Conv); no saving.

## Why lower is not reachable (information-theoretic / forced-component argument)
- **Render floor = 3,600 B.** Up to 3 data-dependent (C,F,shape) triples, bidirectional cross-channel color
  routing over a 3×3 neighborhood, emitting a FREE `[1,10,30,30]` one-hot output. The cheapest ONNX op is a
  single runtime `[10,10,3,3]` F32 Conv/ConvTranspose. **F32 forced** (weight dtype = free-F32-input dtype;
  narrowing pays the ≥9k F32-tax, lib #53). **Dense 10×10 forced** (Conv needs a dense weight; grouped can't
  move C→F; a 1×1 color-map + spread split materialises a 12k–36k canvas; a scalar summing plane is *wrong*,
  proven above). **3×3 forced** (fill reaches r±1,c±1). ⇒ 3,600 B irreducible; the ~900 target is below this
  single tensor.
- **Detector floor = 2,116 B.** Scan all 529 candidate centers, emit a per-position code distinguishing
  4 shapes + (C,F) + validity; minimal charged output `[1,1,23,23]` F32 (k=8 minimal; k=9 → visible 262/266).
- Render (channel space) and detect (spatial space) are **independent and both forced** ⇒ they sum:
  3,600 + 2,116 = 5,716 B dominate; + forced scaffolding/params = **8,710 = byte-exact floor**.

## Verdict
**WALL at 8710.** Six independent formulations now priced dead (this pass: single-plane render broken+no-save;
+ pass-1 carrier 3,600 net-add / two-stage-split 36k canvas; + prior free-output Einsum 8,949 / pair-code U8
detector 12,237 / k=9 detector visible-fail / rank-chain-DCE ops-absent). Cheapest correct formulation found =
the current **8710** graph (profile static=8710 profile=8710). Lower is unreachable by any route: the ~900
target sits below the single forced render tensor (3,600 B).

## One remaining hypothesis (future batch only — do not re-open the priced routes)
A render op that is data-dependent, writes the one-hot output for free, and materialises <3,600 B without a
`[10,10,3,3]` F32 weight or a full canvas — none exists across Conv / ConvTranspose / Einsum / Scatter / plane.
