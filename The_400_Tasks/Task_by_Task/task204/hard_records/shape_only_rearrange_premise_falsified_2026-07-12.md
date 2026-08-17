# DEAD ROUTE — task204 "ShapeOnlyRearrange" (card premise falsified)

**Recorded:** 2026-07-12 by `bo_ng_cld_worker2` (Claude lane), task204 batch.
**Companion record:** `carrier_narrow_equal_tail_falsefloor_2026-07-10.md` (prior batch, different
prompt angle "CarrierNarrowEqualTail"). This record covers the **ShapeOnlyRearrange** card.

## Live bar (refreshed at session start, measured this session)
- **live_cost = 7682** — `tools/ng_current_bar.py --task 204 --json` → **manifest authoritative, NOT stale**.
  BoBrain provider `consistency task204`: team_manifest / task_state / legacy / library all = **7682**.
- **live sha256 = d2f77ac3e6bca0c1e5afdb88378da368ae17805bb70bb62651e5d9d10b150c75**
- Prototype `task204_v2.onnx` confirmed **byte-equal to LIVE** (sha match this session).
- **Fresh measurement (this session):** `profile_real_cost --arcgen 266` → `static=7682 profile=7682`
  (memory 6652 + params 1030; 268 real inputs; **no reprice gap**). `ng_gate visible` = **268/268 PASS**.
- No staged sidecar (`Team_Submission/Staged_Boladi/task204.json` absent) ⇒ 7682 is the effective best.

## The card premise ("shape-only rearrange/crop, 7682 → floor_est ~1500")

**FALSIFIED by first-hand data.** The card is a corpus-trigger artifact (hits: slice/gather/pad/concat).
The task is NOT a shape rearrangement.

### Rule (ARC-GEN 868de0fa), visually + programmatically confirmed this session
Hollow blue (1) square outlines (side 3..10) on black (0). Output preserves outlines; fills each
interior **orange (7) if side length is odd, red (2) if even**.

### Primary falsification (parse of the 5 train pairs, this session)
Every pair: `in_colors = [0, 1]` (black + blue ONLY) → `out_colors = [0, 1, 2, 7]`,
**`NEW_in_output = [2, 7]`** — the fill colors red(2) and orange(7) exist **nowhere in the input**.

A pure Slice/Gather/Pad/Concat/Reshape/Transpose graph can only **copy** values (from input or
constant initializers) to output positions using **static** indices. Two independent blockers:
1. The fill colors aren't in the input. (Solvable — palette initializers. Not the wall.)
2. **WHICH** fill color at each interior cell = the enclosing square's **side-length parity**, a
   value computed by reduction over that square's geometry. Selecting red-vs-orange per cell needs
   **data-dependent** indices/branch — exactly what shape-only ops (static indices) cannot express.
   Computing those indices is arithmetic/Conv/reduction ⇒ you have left the shape-only family.

This is the prompt's own kill rule firing: *"Kill if correctness needs per-cell decisions rather than
rearrangement … Do not turn a geometry task into a color-index or Conv classifier before proving the
shape-only family cannot express it."* Proven: the shape-only family cannot express it. **DROP the ~1500
floor_est as fantasy** (it prices a copy that is impossible here). The two-color size-parity fill is a
compute task; its exact-priced floor is the v2 template-match at 7682.

## Floor confirmed — independent component audit (this session, from the LIVE build script)

`prototypes/build_task204_v2.py`. Every tensor reconstructed and proven load-bearing:

| Component | Bytes | Forced because |
|---|---|---|
| `ch1_det_f32` [1,1,18,20] F32 | 1440 | input is F32; blue read → Slice(F32) preserves dtype. 20 cols needed to see full top-edges (kernel W=12, squares reach col 19); 18 rows = top-left row domain |
| `topleft_len_u8` [1,8,18,18] U8 | 2592 | 8 distinct sizes × 18×18 top-left domain; post-threshold detector signal |
| `paint_w_u8` [1,8,10,10] param | 800 | redraws blue border + fill in ONE stamp (avoids a 2nd, larger blue read); 10×10 = max square |
| bg/valid machinery (rows + outer-product `QLinearMatMul`) | 560 | grid size is variable 10..20 in a 30×30 canvas; background channel needs the true-grid mask |
| `paint_code`/`label_code` [1,1,20,20] U8 | 400+400 | paint output + Max-merge canvas |
| `label30_u8` [1,1,30,30] U8 | 900 | Pad-before-Equal (900) beats Equal-before-Pad (2000); output canvas is 30×30 |
| small U8 rows/cols | 120 | valid-row/col derivation |
| params (topleft 192 + paint 800 + codes/pads/idx 38) | 1030 | all minimal |
| **Total** | **7682** | **= 6652 mem + 1030 params (matches manifest, byte-equal LIVE)** |

## Forced-component argument (why NO route beats 7682) — sharpened this session

Two-color fill forces a per-square **side-length-parity measurement spanning the whole square**.
Exactly two realization families exist:

- **(A) per-length template match (v2, CURRENT):** the entire size axis is packed into ONE dense
  U8 tensor `[1,8,18,18]` (**1 byte/elem** = 2592) + 8-ch paint 800. Inherently robust to multiple
  squares: every stamp is placed at a *verified* (top-left, size), so no cross-square interference.
  → 7682. **Cheapest.**
- **(B) scan / prefix (v3, MEASURED 18636):** interior via crossing-parity needs a horizontal
  prefix-**SUM**, and **ONNX `CumSum` has no U8 kernel** ⇒ the size axis spreads across F16
  `[1,1,20,20]` planes (**2 bytes/elem**), one per scan step. Measured 18636 (2.43×), a regression.

### NEW this session — closing the "all-U8 scan" hole the prior record left open
One could imagine dodging (B)'s F16 penalty with an **all-U8 MaxPool** pipeline: color via
vertical-span parity `parity((near_dn − near_up))` using causal MaxPool of a U8 row-ramp × blue
(Sub/BitwiseAnd run on U8 at opset 14+). But that only handles the COLOR. It still needs an
**interior mask**, and interior is where U8 dies:
- **crossing-parity interior** → needs the horizontal prefix-SUM ⇒ F16/INT32 (no U8 CumSum). Back to (B).
- **4-way "blocked up/down/left/right" MaxPool interior** (all-U8, cheap) → has **multi-square false
  positives**: a cell surrounded by the edges of 4 *different* squares is falsely flagged interior.
  With 229/300 fresh ARC-GEN pairs holding multiple squares, this is a real visible-fail risk, not a
  corner case. So the cheap all-U8 interior is **incorrect**, and the correct interior forces F16/INT32.

⇒ There is no cheap all-U8 scan. (A) packs position×size into one 1-byte-per-elem tensor AND is the
only family robust to multi-square configs without extra disambiguation ops. Zero-sum theorem
(2026-06-30) closes the rest: param↔memory reshuffles are neutral; the only sub-floor levers are
deleting a load-bearing tensor or shrinking the forced F32 read — both proven impossible.

## Micro-levers re-checked and rejected (this session)
- Shrink `ch1_det_f32` to [1,1,18,18] (−144): NO — 20 cols required to see top-edges reaching col 19.
- Shrink paint kernel below 10×10 (−152..288): NO — dropping the blue-redraw forces a 2nd/larger blue
  read (net ≥ 0); already the documented optimum.
- Right-size / split detector channels per L: NO — zero-sum (Don't-do log); ragged channels need
  separate convs ⇒ extra merge/paint planes cancel the param saving.

## VERDICT: WALL. 7682 (v2 template-match) is the exactly-priced floor and the cheapest correct
## formulation. The ShapeOnlyRearrange premise is falsified (fill color is computed size-parity, not a
## copyable/rearrangeable value). floor_est ~1500 is fantasy. No route beats 7682. NO CANDIDATE staged.

## SECOND PASS (2026-07-12, forced re-attempt) — all-U8 MaxPool span family, PROBED + PRICED

Reviewer pushed for a genuinely-different from-scratch formulation with a different dominant
intermediate. The one hole worth probing: the 2026-07-10 record argued the scan family loses because
crossing-parity needs a prefix-SUM and **CumSum has no U8 kernel** ⇒ F16 (2 B/elem). But **MaxPool
DOES accept UINT8** (`ort_reality.md:60`, opset 12+). So an all-U8 MaxPool span pipeline is
*dtype-feasible* — could it dodge the F16 penalty and beat 7682?

**Fast numpy probes (before any ONNX build — the prompt's mandated discipline), full 268-pair corpus:**
- **Cheap 4-way interior** (`interior = hasBlueUp & Dn & Left & Right & ¬blue`, all cheap MaxPool):
  **INCORRECT — 84 false-positive cells across 18/268 pairs.** Cells in the cross-hairs of 4 *different*
  squares get falsely filled. A visible miss ⇒ certain hidden 0. The cheap all-U8 interior does not work.
- **Vertical-span parity color** (`L = nearBelowRow − nearAboveRow + 1`; orange odd / red even):
  **CORRECT — 0/6016 fill cells wrong.** (Reconfirms v3's color finding.) But only correct *on* the
  interior; it doesn't identify the interior.
- **Span-consistency-corrected interior** (`… & (vspan == hspan) & (vspan>0)`, squares are square):
  **BYTE-EXACT — 0 mismatched cells across all 268 pairs.** A correct all-U8 formulation EXISTS.

**Priced (explicit plane inventory, best-case fusion):** the correction forces **four** directional
nearest-edge distance fields (near_up/dn via causal MaxPool of a row-ramp×blue; near_left/right via
horizontal MaxPool of a col-ramp×blue), each a full `[1,1,20,20]` canvas-plane, + the two spans, the
`Equal`, the 4-way `has` reduction, the parity `Mod`, the fill `Mul`, the label merge, `label30` (900),
and a 20-row F32 blue read (1600, *bigger* than v2's 1440 since vertical spans need all 20 rows).
**≈ 11,990 memory + ~90 params ≈ 12,080 — a 1.57× LOSS.** No ONNX built: the numpy proof of
correctness + the plane-by-plane price is an exact priced failure (protocol: "report exact priced
failure instead of grinding"); v3's already-measured F16 sibling (18,636) corroborates the family loses.

### The forced-component argument, now information-theoretic (dimensional packing)
Color = each square's side-length parity — a per-square scalar requiring a whole-square extent
measurement. Making that measurement available at every interior cell has exactly two realizations:
- **(A) pack it into the CHANNEL axis (template, v2):** one `[1,8,18,18]` U8 tensor where channel k ==
  "size k+3 detected here" — the size axis costs **8 B/position** (8 channels × 1 B), 2592 total, and a
  paint conv renders each size's exact interior. Inherently robust to multi-square (verified top-lefts).
- **(B) spread it across SPATIAL scan-planes (v3 F16 / this all-U8):** compute nearest-edge distances by
  prefix ops — **four** directional distance fields + products + spans + comparisons ≈ **28 canvas-planes**
  ≈ 28 B/position. Measured 18,636 (F16) / priced ~12,080 (U8).

Packing 8 size-classes into 8 channels of ONE tensor (**8 B/position**) is ~3.5× denser than unrolling the
size measurement across ~28 spatial planes (**28 B/position**). This is fundamental to how the scorer
charges (every materialized intermediate is a full canvas ≥ its dtype×shape) — NOT a dtype artifact
(all-U8 confirmed feasible, still ~12k). Combined with the canonical floor laws — `cost_floor_reference.md:270`
lists task204 in the floor-verdict ledger at **7682** ("8 length channels + 10×10 paint kernel forced"),
and `:165` proves the single-channel color-index canvas = 900 B output-tail floor — the template-match v2
is the exactly-priced floor.

## VERDICT (pass 2 confirms pass 1): WALL. Cheapest correct formulation = v2 template-match, profile
## 7682. Three from-scratch families now measured/priced (template 7682 < all-U8-span ~12,080 < F16-scan
## 18,636); all decompositions of the size axis into scan-planes lose to channel-packing. No route beats 7682.

## One remaining hypothesis (only if this ever revives)
Same as the 2026-07-10 record: the single forced F32 read `ch1_det_f32` [1,1,18,20] = 1440 (19% of
cost) is the only non-U8 charged tensor. IF a future opset/kernel allowed reading one input channel
straight to U8 without materializing an F32 slice (none at opset 13/14 — `Gather`/`Slice` output dtype
== data dtype == F32), ~1400 B could drop. Not currently expressible. Everything else is load-bearing.
