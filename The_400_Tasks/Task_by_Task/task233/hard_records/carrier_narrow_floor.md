# hard_record: task233 carrier-narrow / ~900 floor route — DEAD (2026-07-11, bo_ng_cld_worker3)

## Live bar (refreshed 2026-07-11)
- **live_cost 24624** (ng_current_bar.py, authoritative=manifest, owner Sho, updated 2026-07-09).
- live_sha256 `2b822338b46698102baef1dfbac706879e93b0df529c28c06473cc795356e121`.
- live graph = `task233_h_global_cost_min_v8` (algorithmic H-Global cost-min; structural_kaggle_delta +13.53 LB ⇒ this rule is CONFIRMED-correct on hidden, NOT the old 266-entry lookup).
- input FLOAT[1,10,30,30]; output BOOL[1,10,30,30].

## Route attempted
Prompt Direction: "carrier-narrow sweep — collapse 10-channel/F32 spatial carrier to U8/color-index
before Conv/Reduce/Where, restore w. Delete charged carrier/canvas/template." Target floor ~900.

## Why it is DEAD

### 1. The charged component named in the Direction is ALREADY GONE
Full F32 intermediate footprint of the live graph is **7,088B total**, and there is **no
10-channel F32 carrier**. The largest F32 tensors are:
- `ch2_f32` [1,1,30,30] F32 = **3,600B** — single-channel red read (Gather ch2), forced for the
  box-crop bounds + hole detection.
- `source_score_3_f32` [1,1,28,28] F32 = **3,136B** — a single 3×3 Conv that already reads all 10
  channels ONCE and outputs the narrowest useful F32 (28×28).
- everything else F32 is ≤336B.
The graph reads a single channel and works in U8/F16 downstream. There is nothing to "collapse."
Per the 2026-07-10 workflow rule ("if [the charged component] is already gone … return DROP with
the refreshed floor note") this alone forces DROP.

### 2. The collapse-to-carrier lever is a NET LOSS (empirically priced)
Fast probe `prototypes/worker3_20260711_floorprobe/` — measured collapse-intermediate cost of every
way to produce a narrow color-index carrier from FLOAT[1,10,30,30]:

| formulation | collapse mem | vs incumbent 3,136B |
|---|---|---|
| A ArgMax → INT64[1,1,30,30] → Cast U8 | **8,100B** | +4,964 LOSS |
| B 1×1 Conv → F32[1,1,30,30] → Cast U8 | **4,500B** | +1,364 LOSS |
| C current 3×3 source-Conv (incumbent) | 3,136B | — |
| D forced red read Gather ch2 → F32 | 3,600B | forced |

Any collapse of 10 F32 channels yields a [1,1,30,30] intermediate ≥3,600B (F32) or ≥7,200B (INT64),
which is **already larger** than the 3,136B the current Conv achieves directly on the input. You
cannot narrow the carrier more cheaply than the graph already does. QLinearConv-on-color-index
(is-decoration map) was also rejected in the 2026-06-02 floor proof: 3×3 sum varies 1–5 vs the
robust uniform-9 detector → weaker signal, risks the 99.9% truth for ~+0.04 LB.

### 3. The ~900 target is below a single forced tensor
Forced floor = red read (3,600B) + source Conv (3,136B) = **≥6,736B** before ANY matching/scatter
machinery. 900 < 3,600, so ~900 is provably unreachable for any graph that reads the F32 input.
(external best_ext_cost 110 in task_state is not reproducible for a static graph — a static graph
prices exactly and cannot beat the 3,600B single-channel F32 read; treat 110 as a dynamic-shape/
estimate artifact, not a real floor.)

### 4. Rung-epsilon shaves below the current rung visible-fail
Prior worker3 (2026-06-28): all 40 single patch-entry drops score ~24,322 but visible-fail 265/266;
the staged `worker3_patch1_onefp0_cost24334` is a −12 non-byte-equal shave (deprecated by the
2026-07-10 "not a rung-epsilon shave" rule). sigmap family (19.5k–21.4k) visible-fails from false
positives. No SAFE structural route below the current rung exists.

## Verdict
**WALL / DROP.** Charged component gone; carrier-narrow is a net loss; ~900 below the forced 3,600B
read. Real forced floor ≥6,736B + proven-required matching machinery (every prior removal regressed
truth). Cheap route (lookup 6120) is the documented self-consistency hidden-test gamble — captain-held.

## SECOND PASS (2026-07-11, bo_ng_cld_worker3) — four more families closed, floor now information-theoretic

**5a. Sigmap signature-gather (the sub-bar family) — visible-fails, CONFIRMED dead.**
All static sigmap probes score below the bar but visible-fail catastrophically:
`v116_under20` 19,514 → visible **17/266 (6.4%)**; `v117_nomask_nohas` 20,184 → visible **151/266
(56.8%)**; v113/v112/v109 (20.5–23.1k) same family. Per VISIBLE-FAIL AXIOM = certain hidden-0.
The dynamic-shape sigmap probes (v106/v107/v113s) ERROR on load ({1,1,0,7} shape mismatch) =
local-cost fiction (traps §GRADER-GAP). Not fixable — a fundamental false-positive gap, not a shave.

**5b. Lookup-table (6,120–9,523) — hidden-risky, captain-held.** Self-consistency trap; lib_query
flags task233 under §14e LEARNING-ARTIFACT (+192 recurrence, 261/2000 truth = 13%). Not a safe route.

**5c. Fused color-index carrier (read replacement) — max saving 1,452B, and the U8 source detector is
PROVABLY too weak.** Structure: `ch2_f32`(3600) feeds one Cast→ch2_u8; `source_score_3_f32`(3136)
feeds one TopK. Fusing both reads into one color-index plane saves at most 1,452B (24,624→~23,172),
NOT a floor route. But it can't even be built safely: the F32 source kernel `[1,10,3,3]` uses **per-color
FRACTIONAL weights** (black=0, red c2=1.00, other colors 1.01–1.07) so sprites out-rank the all-red box
by a hair. A single-value U8 color-index carrier cannot hold fractional weights: they round to 1
(tiebreak lost) or, scaled to integers, a 9-cell 3×3 sum reaches 963 > 255 (U8 overflow). Numerical
proof on 600 real arc-gen pairs: F32 fractional detector separates sprites-vs-box 81.8%; U8 integer
detector **24.8%** (3× collapse). Preserving the tiebreak needs INT32/F32 = 3,136B → zero saving.

## Information-theoretic forced floor (both F32 reads are truly forced)
- **Red read 3,600B**: cheapest path to the U8 red mask is Gather→F32[1,1,30,30]→Cast; casting the
  input first is 9,000B. Any color-index carrier is itself DERIVED from a ≥3,600B F32/INT64
  intermediate — carrier-narrow never escapes the F32 read floor.
- **Source Conv 3,136B**: robust sprite/box separation needs per-color fractional weights over the
  10-channel ONE-HOT (a collapsed single-value carrier destroys per-color separability), at
  ≥F32/INT32 precision (U8 rounds away the tiebreak / overflows). Forced.
- Matching/scatter machinery ~14–17KB: proven-required (every removal truth-regresses; all cheaper
  sigmap assemblies visible-fail 6–57%).

Forced floor ≈ 6,736B + ~14–17KB ≈ low-20KB; current safe rung ~24,334 is near it. **~900 is below a
single forced tensor and is information-theoretically unreachable by any faithful graph.**

**FINAL VERDICT: WALL.** No safe candidate strictly below 24,624 exists on any of the five families.
Do not re-explore carrier-narrow, sigmap, or lookup for this task.
