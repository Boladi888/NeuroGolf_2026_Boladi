# task125 hard_record — Conv/Mul shrink of the Vamshi 2482 render pathway = WALL

- **Live bar:** 2482 (Vamshi), sha12 `9ab2a793fc7b`, manifest.csv `2026-07-02T02:35:15Z`. Profile == static == 2482 (memory 2366 + params 116, 265 real inputs). Team-best onnx 1851 bytes.
- **Verdict:** WALL. No candidate strictly below 2482 found. Champion is a tight local optimum for the MaxPool-directional + QLinearConv-render family.
- **Batch/worker:** batch42 slot07, `bo_ng_cld_worker2` (Claude). Corroborates the prior batch's scratch plan (`prototypes/task125_scratch_plan_20260702.md`) and its failed `task125_v9_direct_pb_renderer.onnx`.

## Champion architecture (behavior oracle + cost baseline)
`Slice ch6->13x13 F32` → `Cast u8` → 4× `MaxPool` (directional, kernel 1x6/6x1) → `Min` (bounded) →
`Concat(p_u8,bounded)=pb` → `QLinearConv 1x1 code_conv -> codes` → `QLinearConv 3x3 group=2 render -> output[1,10,30,30] u8`.
Decode (grader): `run_raw = (out>0)` then **exactly ONE positive channel per cell over the FULL 30x30**
(`ng_harness/encode.py decode_to_grid` / `session.run_raw`). Padding rows/cols 15..29 must be all-zero.

## Forced-component cost (Σ = 2482, no slack)
memory 2366:
- Entry ≥ 845: `Slice ch6->13x13 F32` = 169*4 = **676** + `Cast u8` = **169**. Only Cast changes dtype
  (shape-preserving), only Slice/Gather crop (dtype-preserving) ⇒ 676 F32 crop + 169 cast forced. 13x13
  core forced by margin-1 (6s reach rows/cols 1 and 13). Casting the full input first = 9000 (dead).
- Boxed ≥ 845: 4 directional `MaxPool` (u8, 169 ea) = **676** + `Min` collapse = **169**. All 4 directions
  required (every 3-subset ray-AND ≥ 359,612 FP / 200k per prior floor note); Min is the minimum tensor
  for a 4-way AND. Pools must be u8 (F32 pools = 4*676, dead).
- Render pathway ≥ 676: `pb`(Concat) **338** + `codes`(saturating remap) **338**.
params 116: crop 8 + scalars 4 (scale_one, zero_u8, zero_i8, code_zero) + code_weights 4 +
render_weights[10,1,3,3] group=2 = 90 (10 out ch forced by grader; group=2 min for 2->10; 3x3 forced by
halo 8-neighborhood) + render_bias[10] = 10.

## Why the render pathway can't drop below 676 (the attacked lever — Conv/Mul shrink)
Goal was to delete the 338-byte `codes` plane by rendering directly from `pb=(p,bounded)` (group=1) or a
cheaper remap. All routes fail:

1. **Single combined plane (m=p+bounded, argmax)** — cost 1970, 265/265 by argmax, but the grader decode
   is one-hot `(>0)` with exactly-one channel, NOT argmax. A single plane has 3 distinct center states
   (bg/boxed/six); "boxed" is the MIDDLE value and a linear tap cannot make one channel `>0` at the middle
   while `<=0` at both extremes (non-monotonic). DEAD.
2. **group=1 render on raw `pb` (drop code conv)** — cost 2229, passes a local argmax test but **visible
   0/265**. Two failures: (a) padding: with any scalar `x_zp`, real bg=(0,0) collides with padding under
   `x_zp=0`, or six=(1,1) collides under `x_zp=1` — so some firing state is indistinguishable from the
   all-zero padding window (biases must be ≤0 or padding lights up). (b) even ignoring padding, ch3(halo)
   is int8-INFEASIBLE from `{0,1}` channels (weak six-neighbor signal drowned by bg-neighbor noise).
3. **group=1 + ONE cheap remap op (A=k·p, B=k·bounded, or m=p+bounded; scan over k∈{1,2,5,7,10,13},
   x_zp∈{0,1,2,3})** — `solve_render.py` LP over the full 265 corpus (238,500 output cells): **ch3(halo)
   int8-infeasible for every config**. Keeping p and bounded in separate channels gives no cross-mixing;
   scaling for a strong six signal (10p) still can't separate halo within |w|≤127.
4. **Per-channel `x_zp`** (the natural fix for the (2)a padding collision) — prior worker's
   `task125_v9_direct_pb_renderer.onnx`: ORT rejects it — `QLinearConv: input zero point must be a scalar
   or 1D tensor of size 1`. DEAD at the runtime level.
5. **Saturating remap is REQUIRED and costs the plane.** Feeding the LP the champion's *saturated* codes
   (`clip(-p+7b), clip(5p+2b)`) makes ch3(halo) int8-feasible with |w|≤4 (vs infeasible for all linear
   remaps). The saturation nonlinearity that enables separability comes only from a QLinearConv/Clip op,
   which materialises a 338-byte 2-channel `codes` plane. Producing `codes` by arithmetic (Mul/Add/Sub)
   needs ≥3–4 planes (≥507) + concat ≥ champion. Net: no route removes a 338 pack without adding ≥ its
   cost back.

## Best theoretical (still unreachable cleanly)
Group=1 + one remap would be mem 2197 + params ~202 = ~2399 (−83) IF the render were expressible — but it
is not (ch3 infeasible + padding collision). So 2399 is a paper number, not a candidate.

## One remaining hypothesis (low confidence)
The champion's directional `MaxPool` uses kernel 1x6/6x1 (reach 5 cells) — an in-distribution assumption
(boxed cells' bounding 6s are within 5 cells; verified on 200k as it is the live best). A full-reach
kernel 1x13/13x1 is same memory (169) and strictly more robust — but does NOT reduce cost, so it is not a
win, only a robustness note if the champion is ever contested on hidden data. No cost lever remains.

## Gates run
- profile_real_cost champion: `static=2482 profile=2482 (memory 2366 + params 116; 265 real inputs)`.
- ng_gate visible champion: `265/265 (100.00%) PASS`.
- ng_gate visible v3_onehot (group=1, my best decode-correct build): `0/265 FAIL` (padding lights via bias;
  confirms the decode/padding wall). No truth/ab/battery run on visible-zero candidates (traps §VISIBLE-FAIL).

Evidence scripts: `prototypes/solve_render.py` (LP feasibility scan), `prototypes/build_v3_onehot.py`,
`prototypes/build_v2_singleplane.py`, `prototypes/rule_ref.py` (numpy rule 265/265 + grader one-hot decode).
