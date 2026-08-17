# task331 — F32-carrier-narrow / CompassCastU8 → WALL (forced floor 612)

- **Live bar:** cost **612** (manifest, authoritative), sha256 `c70b91043c26e92dc06a82adab2443992b442776da447fd647916d2da905cb3a`, owner Boladi, updated 2026-07-02T20:17:39Z. `ng_current_bar.py` reconciles; local task_state (910/11948) is STALE, ignore.
- **Profile == static == 612** (memory 500 + params 112, 265 real inputs). No grader gap.

## Route attempted
Prompt CompassCastU8: "Cast color0_crop F32→U8 (400→100B) before the shifts, drive toward floor ~300." Premise: the 400 B F32 crop is a formulation artifact that a Cast/U8 carrier deletes.

## Why it is DEAD (forced-component proof)
Cost model (`neurogolf_utils.calculate_memory/params`): `memory` = SUM over EVERY intermediate tensor of `elements × itemsize` (max of static/runtime), excluding only tensors literally named `input`/`output`. `params` = SUM of initializer **element counts** (dtype-independent).

Live graph = `Slice(input F32)→color0_crop F32[1,1,10,10]` → `QuantizeLinear→signed_state_u8 U8[1,1,10,10]` → `QLinearConv→output U8[1,10,30,30]` (free output). Decomposition:
- color0_crop F32 100 elem × 4 = **400** (charged)
- signed_state_u8 U8 100 elem × 1 = **100** (charged)
- params **112** = Wq[10,1,3,3]=90 + Bq[10]=10 + starts4[4]+ends4[4]=8 + 4 quant scalars
- **Total 400+100+112 = 612.**

The prompt's Cast is ALREADY what the QuantizeLinear does. Cast/Quantize is elementwise → same shape → it cannot delete the F32 crop, because the F32 crop is a Slice node output that is **charged before** any cast can read it. The only way to never materialize an F32 crop is to cast the WHOLE F32 input first, which is strictly worse:

**Forced components (each independently minimal):**
1. Input is fixed F32[1,10,30,30]. The rule reads all 100 channel-0 cells (markers may sit anywhere in the 10×10 active grid) → a ≥400 B F32 first-touch tensor is forced. The 10×10 F32 crop (400 B) is the minimal F32 slice; any larger slice (e.g. [1,1,30,30]=3600) is worse. Slice preserves F32 dtype (library #5: F32 Slice/Gather stays F32).
2. Output must be UINT8[1,10,30,30]. Producing it **free** requires QLinearConv (only op giving a free U8 render); QLinearConv requires a U8 input → a 100 B U8 crop is forced. Alternative (plain Conv → F32[1,10,30,30]=9000 B intermediate → Cast) is strictly worse.
3. Render params all forced: 10-channel one-hot output ⇒ Wq M=10; cardinal (4-neighbor) offsets ⇒ 3×3 kernel; per-channel threshold ⇒ Bq[10] (QLinearConv y_zero_point is scalar, cannot host per-channel offsets); minimal 4D Slice = 8; 4 distinct quant scalars.

Sum of forced components = **612**. No lower configuration exists. The ~300 floor_est in the prompt is a formulation error (assumed Cast removes the F32 carrier; it cannot).

## Formulations priced (independent, from-scratch)
| # | Formulation | Charged memory | params | Total | Gate |
|---|---|---:|---:|---:|---|
| A | Slice-F32 → QuantizeLinear-U8 → QLinearConv (LIVE = prompt's own "Cast to U8" graph) | 400+100=500 | 112 | **612** | static=profile=612 |
| X | Quantize FULL input → Slice U8 → QLinearConv (never make F32 crop) | 9000+100=9100 | 112 | **9212** | score_v2=(0,9100,112) — cast-full loses (library #5, task255/370) |
| Y | Slice-F32 → plain Cast-U8 → QLinearConv | 400+100=500 | 112 | 612 | plain Cast breaks quant zp semantics → visible-fail; memory identical to A regardless |
| B | 4 Slice/Pad compass-shift planes | 500 + N×100 (each shifted U8 plane charged) ≥ 900 | — | ≥900 | scratch notes + memory accounting: LOSES (each shift = new 100 B charged plane; QLinearConv fuses all 4 shifts+one-hot+embed in one free-output op) |
| — | QLinearConv w/ U8-in/I8-w/I8-out (drop a scalar) | — | — | — | prior probe `task331_qlinear_i8zp_probe.onnx`: no ORT kernel → score None, visible errors |

## Verdict
**WALL at 612 = forced floor.** color0_crop F32 (400 B) is a forced component, not an artifact. Do NOT re-explore the U8-carrier / Cast-pre-shift route: the F32 crop is charged before any cast, and full-input cast is 9100. Do NOT re-run the I8-zp QLinearConv (no ORT kernel).

## One remaining hypothesis (low confidence)
Only sub-612 lever would be shrinking **params** below 112 — but Wq(90)/Bq(10)/Slice(8)/scalars(4) are each structurally pinned by the 10-channel one-hot + 3×3 cardinal render (see forced-component #3). A per-channel bias cannot fold into scalar y_zero_point; a smaller kernel cannot reach 4 cardinal neighbors. No credible params reduction identified.
