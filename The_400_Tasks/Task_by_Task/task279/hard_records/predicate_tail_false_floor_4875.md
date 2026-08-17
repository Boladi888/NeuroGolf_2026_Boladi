# task279 hard record — predicate-only tail / ~900 false-floor (DROP)

**Session:** `bo_ng_cld_worker2` (Claude lane), 2026-07-11.
**Verdict: DROP / WALL.** The 2026-07-09 prompt's ~900 `floor_est` is a **false floor**,
and the charged component named in its Direction is **already gone** from the live graph.

## Live bar (re-measured)
- **Authoritative live cost = 4,875** (`ng_current_bar.py --task 279`: manifest, owner **Sho**,
  updated 2026-07-09T01:09Z), sha256 `e281359d428f0b084b6f609c90bdd2996e814526dd112bd15de8d911ead7230c`.
- `_score_v2` on the live `Team_Best_Onnx/task279.onnx`: `(macs=0, memory=4740, params=135)` →
  **static == profile == 4,875** (exact; no grader gap).
- ⚠️ `task279.md` is STALE: its `## Cost analysis` still says team-best **5,057 (Vamshi)** and
  STATUS says LIVE 14,067. Both are superseded. Sho's **4,875** is live. (Left for runner to fix;
  worker prompt scoped writes to `task279/`.)

## Why the 5,057 "WALL" fell — the pattern that repeats here
The 2026-07-02 floor proof declared **WALL at 5,057** via a complementary-error argument on the
`erode×4 + jump×2 + flood×2` structure. Sho beat it anyway by **changing structure**, not by
shaving: replace the 2-jump + 2-flood regrow (4 planes, 1,024 B) with a **single 7×7 `QLinearConv`
"recover"** (radius-3 regrow in one op; components are ≤5×5 so radius-3 covers). That is a
structural false-floor crack, exactly what this task class is about. It is the reason old
per-structure floor proofs are "tried-path evidence only."

## Decoded live 4,875 graph (Sho) — the charged component is ALREADY gone
```
Slice(input, c1 ch, 16x16) -> blue_f32 F32[1,1,16,16]      1024  (FORCED, F32 input)
Cast -> blue U8[1,1,16,16]                                   256
6x QLinearConv peel  k=[[0,1,0],[1,3,1],[0,1,0]] scale 9 -> core_0..core_5   6*256=1536  (erode open arms, keep cycle cores)
Concat(core_5, blue) -> recover_feat U8[1,2,16,16]          512
QLinearConv 7x7 recover (scale 47, irregular 2-ch weights) -> cycle_component U8[1,1,16,16]  256
Greater(>0) -> cond_crop BOOL[1,1,16,16]                    256
Pad 16->30 -> cond BOOL[1,1,30,30]                          900  (FORCED, Where broadcast)
Where(cond, color8_onehot, input) -> output F32[1,10,30,30]
```
memory 4740 = 1024+256+1536+512+256+256+900. **No materialized canvas, no template bank, no
repeated Pad/Concat planes** — the output is a single `Where(predicate)`. The Direction's target
("avoid stored/materialized canvas, generate predicate/index and let final output op materialize;
delete charged carrier/canvas/template") is **already implemented**. Per the 2026-07-10 workflow
update: *"if it is already gone … return DROP with the refreshed floor note."*

## Fast probe (the ~900 route), priced
`prototypes/build_predicate_floor_probe.py` → `predicate_floor_probe.onnx`: the absolute-minimum
predicate-only Pad/Concat-free tail (`Slice → Greater → Pad → Where`) with **ZERO cycle
detection** (recolors every c1 cell; 100% wrong).
- `_score_v2` = `(0, 2180, 25)` → **cost 2,205**.
- 2,205 **> 900 ⇒ fast-probe KILL rule fires.** Even the degenerate, correctness-free tail is
  2.45× the claimed floor.

## Forced-component floor (why <900 is impossible; dodges priced)
- `blue_f32` F32 16×16 = **1,024 FORCED**. Read c1 from F32[1,10,30,30] input ⇒ Slice keeps F32.
  Dodge (cast full input to U8 first) = 9,000 B. Worse.
- `cond` BOOL 30×30 = **900 FORCED**. `Where(cond, ·, input[1,10,30,30])` needs the mask at output
  resolution. Dodge (Where at 16×16 then pad the 10-ch output) = F32 10×16×16 = 10,240 → far worse.
- **Forced sum = 1,024 + 900 = 1,924 > 900 before any cycle detection.** The ~900 estimate counted
  only the output predicate plane and dropped the forced F32 input crop.

## Cycle detection is irreducible (can't cut the working set) — prior falsifications still binding vs 4,875
These were falsified vs the 5,057 bar; they are CORRECTNESS/structure floors (visible-fail = certain
hidden 0), so they bind against 4,875 too (all also price > 4,875):
- **6-shrink minimum** (complementary-error floor + Sho): drop to 5 shrink ⇒ long straight open
  arms survive ⇒ false positive ⇒ 265/266 visible-fail.
- **Local/aggregate cycle classifier** (`hard_records/topology_cycle_probe.md`): exact
  `edges≥vertices` per component can't be broadcast to all its pixels without materialized labels
  or flood/regrow planes; minimal local topology (9 planes) = 264/266 visible-fail; adding the
  needed 2nd regrow ≈ 5,544.
- **Ring-template seed bank** (`hard_records/ringtemplateqlinear.md`): `ring_bank[1,16,16,16]` =
  4,096 B and still 264/266 visible-fail on larger/irregular closed components.
- **Regrow micro-shave** (Concat→dilate+Mul, ~−256 B → 4,619): `recover_w` is an irregular
  2-channel weighted threshold (channel-1 blue gate center weight 16 + scattered channel-0 seed
  weights, y_scale 47), NOT a trivial dilate — a Mul-gate swap is not obviously byte-equal, is a
  forbidden rung-ε shave (+0.05 pts), and is nowhere near the ~900 target. Not pursued (prompt:
  "one crack only … stop on a priced wall").

## Bottom line
Live 4,875 is the practical floor of the QLinear cycle-detector family: forced `blue_f32`(1024) +
forced `cond`(900) + irreducible {blue U8 256, 6 erosions 1536, cheapest-known regrow 768,
cond_crop 256} + params. No known structure reaches the ~900 estimate; that estimate is a false
floor (omits the forced 1,024 F32 crop). **DROP.**

## One remaining hypothesis (for a future batch, low odds)
The only path below ~2,000 would eliminate BOTH the F32 read AND the iterative erosion set. That
needs a legal ONNX op that computes a per-4-connected-component cycle aggregate and broadcasts the
per-component decision to every pixel WITHOUT materialized labels or flood/regrow planes, AND a way
to obtain the c1 mask without an F32 slice of the F32 input. No ONNX primitive does the former
(Euler-char gives only a global count — don't-do log); the latter is impossible with F32 input.
Revisit only if the input encoding changes or a component-aggregate op appears. Otherwise WALL.

## Pass-2 empirical confirmation (2026-07-11, forced 2nd pass) — working set is irreducible
Built a numpy simulator of Sho's EXACT pipeline (`prototypes/sim_regrow_variants.py`) and a
faithful ONNX reconstructor (`prototypes/build_shofamily.py`, reuses Sho's initializers verbatim).
Validation: numpy `sho_regrow_numpy` = **266/266**; ONNX 6-peel rebuild scores **exactly 4,875**
(memory 4740, params 135 — cost-identical to Sho) and REAL `ng_gate visible` = **266/266 PASS**.
So both the sim and the rebuild are faithful to the live grader.

- **Regrow-cut route DEAD.** Swept 5 cheaper regrows meant to drop the 2-channel Concat plane
  (−256 → 4,619): `maxpool7&blue` 67/266, `maxpool5&blue` 146/266, `conv7>=2&blue` 87/266,
  `conv7>=3&blue` 96/266, `conv5>=2&blue` 187/266 — all FAIL, most at pair 0. Sho's 2-channel 7×7
  weighted `QLinearConv` (recover_scale 47) is load-bearing per-cell logic, NOT a dilate+gate; no
  cheaper regrow reproduces it.
- **Peel-cut route DEAD.** Peel-count sweep with Sho's exact regrow: p3 94/266, p4 198/266,
  **p5 242/266**, p6 266/266, p7 266/266 (numpy). Built `shofamily_p5.onnx` (static **4,619**,
  −256 vs bar) and ran the REAL gate: `[visible] 242/266 (90.98%) FAIL` — matches numpy exactly.
  5 peels ⇒ certain hidden 0 (VISIBLE-FAIL axiom). 6 is the tight minimum for Sho's family too.
- **cond_crop-cut DEAD** (analytic): Pad-then-Greater (900+900) > Greater-then-Pad (256+900).
- Net: working set {blue 256, 6 peels 1536, regrow 768, cond_crop 256} is fully load-bearing;
  every removable tensor breaks a different official class. **Cheapest correct formulation = Sho's
  6-peel + 7×7 regrow = 4,875 == live.** No strict win exists on this route. WALL at 4,875.

## Gates run
- `ng_current_bar.py --task 279 --json` → live 4,875 / sha e281359… (bar confirmed).
- `_score_v2` live graph → 4,875 (bar), predicate_floor_probe → 2,205 (fast-probe wall),
  shofamily_p6 → 4,875 (cost-faithful rebuild), shofamily_p5 → 4,619 (−256 cut, visible-fail).
- REAL `ng_gate visible`: shofamily_p6 **266/266 PASS**; shofamily_p5 **242/266 FAIL**.
- No `truth`/`ab`/`battery`/`profile_real_cost`: the only sub-bar-cost candidate (p5) is a
  visible-fail ⇒ certain hidden 0; advancing it to heavy gates is pure burn (gate-ladder rule).
  No winning candidate exists ⇒ no `ng_safety` (nothing to ship-check).
