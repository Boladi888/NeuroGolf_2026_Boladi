# hard_record: task233 MorphologyPool / local-neighborhood rewrite — WALL (2026-07-12, bo_ng_cld_worker1)

Prompt: `Prompt_20260711T000000Z_6409_Task233_MorphologyPool.md` (STRUCTURAL card, floor_est ~1500).
This is the **morphology/pooling** angle. Carrier-narrow / sigmap / lookup were closed separately in
`carrier_narrow_floor.md` (bo_ng_cld_worker3, 2026-07-11); this record closes the local-morphology
family with first-hand probe evidence and corrects the bar.

## Bar correction (IMPORTANT — flag for runner/captain)
- `ng_current_bar.py` authoritative live_cost = **24624**, but its recorded `live_sha256`
  `2b822338…` points to `…/Backend/Team_Best_Onnx/task233.onnx` (10,729 B) which is a **dynamic-shape
  graph that ERRORS on the 30×30 officials**: `Pad node 'safe_name_160' … Shape mismatch {1,1,20,20}
  != {1,1,30,30}`; `score_v2` → `None`. Its cost is dynamic-shape FICTION (traps §GRADER-GAP).
- The **real shipped/working best** is `…/Staged_Boladi/_shipped/task233.onnx`, sha `fbe0832f…`,
  52,515 B, `score_v2 = (0, 24307, 317)` = **cost 24624**, static, `ng_gate visible 266/266 PASS`.
  My bar = **24624** (this file). Copy for reference: `prototypes/_live_shipped_24624.onnx`.
- The sub-bar non-byte-equal shaves `task233_worker3_patch1_onefp0_cost24334` (sha `11603d…`) and
  `…_cost24346` (`7b4d08e7…`) now sit in `…/Staged_Boladi/_hidden_failed/2026070*/`. The 24334 rejection
  record `Agenters/…/_consumed/20260628T230252Z-task233.rejected_hidden_failed_overlay_20260702.json`
  shows it PROBE-passed in isolation (score 78.66) but was **rejected on overlay** (truth 99.819% vs
  live, non-byte-equal, 189 ab-divs) and the team reverted to the byte-equal 24624. Do NOT restage them.

## Route attempted: local MorphologyPool (crop + 3×3 window match + paint), no global label state
Numpy first-hand probe on ALL visible + arc-gen (`prototypes/worker1_morph_local_probe.py`):
- crop = bbox of the largest 4-connected color-2 component → **output shape exact** on 4/4 real +
  262/262 arc-gen (crop rule is trivially correct & cheap; a reduction, not the cost driver).
- rule = fill crop holes with 2, then stamp each source-tile's decoration color where the tile's
  color-2 stencil (any of 4 rotations) lands on holes and its deco cells land on filled cells.

| mode | train | test | arc-gen | verdict |
|---|---|---|---|---|
| **A pure-local** (paint every match, NO conflict state) | 2/3 | 1/1 | **231/262 (88.2%)** | VISIBLE-FAIL → certain hidden-0 |
| **B greedy conflict-resolution** (accept non-overlapping) | 3/3 | 1/1 | **257/262 (98.1%)** | still VISIBLE-FAIL; needs exact cost-min |

**Conclusion:** the prompt's explicit kill condition is MET — the rule *does* need global
conflict-resolution / assignment state. Pure-local morphology visible-fails at 88%; even greedy
global resolution leaves 5 arc-gen misses that only the live graph's TopK **cost-min** assignment
gets right. This independently reconfirms the 2026-06-02 floor proof (removing conflict resolution →
truth 97.95%, 27/1164 regressions; removing cell_valid → 96%).

## Why no cheaper faithful graph exists (priced, information-theoretic floor)
The live graph IS the faithful cost-min implementation of exactly this rule (crop + 5-source match +
cost-min resolve + scatter + one-hot), and it is the endpoint of a documented 7-stage byte-equal
compression chain (v53 28,095 → v55 27,718 → v56 26,837 → … → 24,624). Forced components:

| component | bytes | why forced |
|---|---|---|
| `ch2_f32` red read [1,1,30,30] F32 | 3,600 | crop bbox + hole detection; casting input narrow = 9,000 |
| `source_score_3_f32` Conv [1,1,28,28] F32 | 3,136 | source detect needs per-color FRACTIONAL weights over all 10 ch (U8 collapses sprite-vs-box 81.8%→24.8%); Conv F16-out on F32-in impossible |
| 5× `match_score_1_N_*` F16 [1,324] | 3,240 | **U8 TopK is `NOT_IMPLEMENTED` in ORT** (`TopK(11)`), reconfirmed by v54+v98 probes; F16 is the smallest supported TopK score type |
| `slot_sig` F16 + `ch2_crop_f16` | ~2,096 | 3×3 pattern signature = a **9-bit position encoder** (kernel `[[1,2,4],[8,16,32],[64,128,256]]`, max 511 > 255) → U8 QLinearConv OVERFLOWS; F16 forced |
| conflict-resolution + scatter + one-hot | ~9k | proven-required (floor proof + probe above); 5 sources sized EXACTLY to the max (58 arc-gen pairs carry 5 tiles); K_TOP=3 (pairs 194/243 need the 3rd fallback) |

Forced read floor alone = **6,736 B**; with the proven-required machinery the safe minimum sits right
at the current **24,624**. The prompt's ~1500 floor_est is below a single forced tensor (3,600 B) —
**fantasy**. `task_state best_ext_cost 110` is a dynamic-shape estimate, not reproducible statically.

## Second pass (2026-07-12, forced re-attempt) — cost model nailed + every alt assembly priced
Read the scorer (`neurogolf_utils.calculate_memory`): **tensors named `input`/`output` are EXCLUDED**
(the [1,10,30,30] one-hot output is FREE), cost = Σ(elements×dtype.itemsize) over intermediates
(max static/runtime-profiled) + Σ(init element counts). So the ENTIRE 24,624 is intermediates; the
one-hot Equal-fold at the tail costs nothing. Full distribution: FLOAT 7,088 · F16 5,982 · U8 5,763 ·
I32 2,320 · BOOL 2,034 · I64 1,120; top-2 = the two reads (6,736), then a 448-tensor plumbing tail (2,842).

Genuinely-different assemblies I priced and REJECTED (all tie/exceed or break byte-equality):
- **Row/col projections for bbox** (`ReduceSum axes=[3]/[2]` → [1,10,30,1] 1,200 ea) instead of the
  3,600 read: still need the 2D crop for holes → projections DUPLICATE the read (≥4,000 > 3,600). Worse.
- **Dynamic Slice crop (#146 task036/079 crop law)**: the crop uses **clamp-padding** — rows past the
  rectangle are `Where(row_mask, r0+range, r0)` = duplicate of row r0 (needed for sub-20 crops: 8×8,
  9×9, 8×12 are all <20). A contiguous Slice takes real out-of-rectangle rows instead → NOT byte-equal.
  (And GatherND for the crop needs [20,20,2] I64 = 6,400 B ≫ the 600 B `ch2_rows20` it would replace.)
- **U8 source detector / two-step decoration-presence map**: 3,600+3,136 > the single 3,136 all-10-ch
  fractional Conv; U8 collapses sprite-vs-box (worker3). Conv F16-out on F32-in impossible.
- **ScatterND paint** (avoid the flat round-trip): indices [85,2] I64 = 1,360 B > the 340 I32 flat +
  ~800 reshape it would save. Worse.
- **Empirical byte-equal redundancy scan**: **0 CSE duplicates, 0 duplicate initializers**, and the
  ONLY dead tensor (`constraint_sort_values` F16[1,5]=10 B, an unused TopK Values output) is
  **non-removable** — ORT/onnx.checker reject an empty required TopK output (`marked single but has an
  empty string`). The graph has literally **zero removable slack**, confirming the 7-stage compression
  reached the exact floor.

Net: the live 24,624 is the exact minimal cost of the faithful cost-min assembly. No built candidate
beats it; no `ng_graph_cost_diff` claim to make (no candidate < bar).

## Verdict: WALL / DROP
No SAFE candidate strictly below 24,624 exists on the morphology/pooling family (or any other):
local morphology visible-fails, byte-equal narrowing is ORT/encoding-blocked, sub-bar shaves
hidden-failed, lookup (6120/9523) is the §14e self-consistency gamble (13% truth on 2k). 

## Do-not-replay for the next batch
- Do NOT re-probe local/greedy morphology without conflict state — visible-fails 88%/98% (priced above).
- Do NOT retry U8/INT match TopK (ORT `NOT_IMPLEMENTED`) or U8 QLinearConv slot_sig (511>255 overflow).
- Do NOT restage 24334/24346 — they are in `_hidden_failed/`.
- Bar is `fbe0832f` (24624), NOT the broken dynamic `2b822338` Team_Best_Onnx file.
