# task279 hard record — regrow kernel-size lever (K7→K5, 4828) + separability floor proof (WALL)

**Session:** `bo_ng_cld_worker1` (Claude lane), 2026-07-12.
**Verdict: WALL at 4875 for a shippable cost win.** A genuine −47 B param lever exists
(regrow 7×7→5×5, prior workers never tested regrow kernel size) and was fully built + gated,
but it trades correctness below LIVE and is not byte-equal, so it fails both ship paths.

## Live bar (re-confirmed)
- `ng_current_bar.py --task 279 --json` → **live 4,875** (manifest, owner **Sho**, 2026-07-09),
  sha256 `e281359d428f0b084b6f609c90bdd2996e814526dd112bd15de8d911ead7230c`.
- `_score_v2` LIVE = `(macs 0, memory 4740, params 135)` = **4,875**, static==profile.
- Sho graph = `Slice→Cast→6×QLinearConv peel(3×3)→Concat(core_5,blue)→QLinearConv recover(7×7,2ch)
  →Greater→Pad16→30→Where(cyan_vec,input)`. Forced: `blue_f32` F32 16×16 = **1024**, `cond` BOOL
  30×30 = **900**. (Prior floor record `predicate_tail_false_floor_4875.md` still valid; this
  record ADDS the kernel-size lever + a stronger separability proof + the LIVE-is-fitted finding.)

## The TRUE rule (independently derived, 100% verified)
Generator `task_b2862040`: axis-aligned color-1 box rings 3×3..5×5, margin 2, holes ≤3×3, plus
isolated single-cell "barnacles" (`remove_neighbors` ⇒ never chained, radius-1 tails inheriting
parent color). Open box = one **non-corner edge** cell removed. Rule = **a 4-connected color-1
component → 8 iff it contains a cycle (edges ≥ vertices)**. A `scipy.ndimage.label` 4-conn +
E≥V oracle scores **0/2000 disagreements** with generator truth (`diag_sho_disagree.py`).

## NEW finding 1 — the peel+single-conv-regrow family CANNOT be 100%
A QLinearConv regrow is a **linear threshold** on its window (signed effective weights available
via weight zero-point: `conv_eff = Σ x·(w_q − w_zp)`). Reframing the regrow as per-cell linear
separability over the K×K window of (core after 6 peels, blue):
- **Window-contradiction check** (definitive; features are binary so each window packs to an
  int64): **0 contradictions** over visible+15k fresh for p6 K5 — the window DETERMINES the true
  label, so a *function* exists.
- **LinearSVC at scale** (visible+6k fresh, deduped unique windows): `K5 train_err=8`,
  `K7 train_err=3` — **NEITHER is linearly separable.** So **no single conv (any K) is a 100%
  regrow.** This is exactly why LIVE tops out ~99.7% truth: it is a *fitted approximation*, not
  the rule. (A 100% graph needs component labeling / flood — the ≥14k family, DEAD on cost.)

## NEW finding 2 — the peel/kernel separability boundary (memory floor is rigorous)
`map_separability_boundary.py` (concat mode, visible+3k fresh):
- peel count: **p4 = CONV-IMPOSSIBLE** (window contradictions — 4 peels lose info);
  **p5 = NOT linearly separable** (err 52 @K7, 42 @K9, 37 @K11 — bigger kernel does NOT rescue);
  **p6 = separable @K7.** ⇒ 6 peels is the tight minimum ⇒ **memory 4740 immovable.**
- regrow kernel @p6: **K3 = CONV-IMPOSSIBLE** (contradictions); **K5 = separable**; K7 = separable.
- single-plane combined regrow (`Add(core,blue)`, drops the 512-B Concat): NOT separable at any
  peel/K ⇒ the 2-channel Concat is load-bearing. Confirms prior regrow-cut DEAD from a new angle.

## The lever: regrow K7→K5 = **4828** (−47 B) — BUILT and GATED, near-miss
Sho's `recover_w` genuinely uses the 7×7 outer ring (ch0 core seeds at radius 3), so **no
byte-equal 5×5 crop.** But K5 is separable, so a *re-fitted* 5×5 regrow exists. Fitted a signed
integer K5 (visible×200 weighted + 9k fresh, LinearSVC → integer realize via `w_zp=6`, y_scale
249), built by mutating Sho's graph (`build_k5.py`, ORT-vs-numpy self-check 300/300 faithful).
- `task279_k5.onnx` sha256 `12f6835c3bbdda75890d2ccbf62869ad83de4f32af69f57dc982730c266c2af4`.
- `_score_v2` / profile = **4828** (memory 4740 + params 88; recover_w 98→50, +1 for recover_wzp).
  static==profile, 266 real inputs. **Cost strictly < 4875.**
- `ng_gate visible` = **266/266 PASS**. On the 266 visible K5 output == Sho output (0 divergences).
- **Ship gate `truth --n 2000` = 1993/2000 (99.650%), deterministic** (re-run identical).
  **LIVE `truth --n 2000` = 1994/2000 (99.700%) on the same sample — Sho beats K5 by 1 pair.**
- `ab --n 2000` vs LIVE = 2259/2266, **divergences=7, byte_equal=False.** Since cand_misses(7) >
  LIVE_misses(6), `ab_divs ≠ LIVE_misses − cand_misses` ⇒ **NOT more-correct-than-LIVE.**

### Why it's not a shippable win (both ship paths fail)
LIVE is <100% ⇒ ambiguous-class ⇒ gold gate = **byte-equal-vs-LIVE** (K5 fails: not byte-equal,
7 divs) **OR more-correct-than-LIVE** (K5 fails: 99.650% < 99.700%, net +1 error). K5's held-out
edge on my *own* generated pairs (99.54–99.73% vs Sho) did NOT transfer to the official ARC-GEN
gate — a fitted-artifact overfit. Root cause: K5's irreducible error floor (8 non-separable
windows) > K7's (3), so K5 cannot match a good K7's correctness. Matching Sho's correctness needs
radius-3 = K7 = 4875 (cost tie, not a win). **No cost win keeps correctness ≥ LIVE.**

## Bottom line
Memory 4740 forced + params floor: the sole sub-4875 lever (K5 regrow) sacrifices correctness
below LIVE and is not byte-equal ⇒ unshippable. **WALL at 4875.** Cheapest correct (LIVE-matching)
formulation remains Sho's 6-peel + 7×7 regrow.

## Remaining hypotheses (for a future batch)
1. **Correctness-only (NOT a cost win, cost tie 4875):** Sho is a *suboptimal* K7 fit — LIVE 99.70%
   vs K7 irreducible floor of only 3 non-separable windows. A better-fit K7 at the SAME 4875 cost
   could reach ~99.8% truth. If team scoring is `points × truth%`, that raises score at zero cost
   change. Not staged here (my mandate is cost; staging a tie is barred), but flagged for a
   correctness pass. Reuse `fit_realize_k5.py` with K=7.
2. **A genuine cost win requires escaping the peel+conv family entirely** — a legal ONNX op giving
   a per-4-connected-component cycle aggregate broadcast to all pixels WITHOUT flood/label/regrow
   planes, AND the c1 mask without an F32 slice. No such primitive exists (Euler-char = global
   count only). Revisit only if the input encoding changes or such an op appears.

## Gates run (this session)
- `ng_current_bar` → 4875/e281359… ; `_score_v2` LIVE 4875, K5 4828 ; `profile_real_cost` K5 4828.
- `ng_gate visible` K5 = 266/266 PASS.
- `ng_gate truth --n 2000` K5 = 1993/2000 (99.650%, ×2 deterministic); LIVE = 1994/2000 (99.700%).
- `ng_gate ab --n 2000` K5 vs LIVE = 2259/2266, 7 divs, byte_equal=False.
- No `ng_safety` (K5 not ship-ready — fails ship gate; gate-ladder: no heavy gates on a loser).

---

## PASS-2 addendum (forced second pass, 2026-07-12) — library mining + forced-component proof, WALL holds

Pushed for a genuinely DIFFERENT from-scratch formulation (different dominant intermediate
shape/dtype or assembly) and mined the library/BoBrain. **No cheaper formulation exists.**

**Library/BoBrain mining (no lever found):**
- `techniques.md #37` (flood/CC iteration floors, task279-specific): confirms the peel/flood count
  is corpus-measured at the floor; **"erode-4×4 seed enrichment to cut steps = −EV (saves 256 B,
  breaks 237/200k)"**; the rectangle-geometry contract only legalized the *8-conn CC step* (512 B),
  already in the family. None reduces below 4875.
- `#38 ScatterND` / `#39 GatherND` sparse canvas/read idioms: task279's changed-cell set is
  **data-dependent variable count** ⇒ dynamic shapes ⇒ the §39 calibration warning (task014:
  local 4,380 → Kaggle 6,093) says these cost MORE on the grader. Dead.
- `#123` predicate/coordinate idioms (interval/Mod/AveragePool-histogram/exp-position): none
  applies to a per-pixel cycle-membership recolor.
- BoBrain provider `context/consistency/search`: bar 4875 (Sho) reconfirmed; library route 50,462
  is stale/contextual.

**Forced-component proof (memory floor 4740, empirically verified):**
- **F32 read = 1024 FORCED.** Scanned 8266 grids (officials+8k fresh): max grid **16×16**, ALL
  non-maroon content within `[0:16,0:16]` (top-left). ⇒ the 16×16 crop is necessary (max dim 16)
  AND sufficient. Input is F32 ⇒ `Slice` keeps F32 ⇒ `blue_f32` F32 16×16 = 1024. No op does
  F32→U8+crop in one; cast-first on full input = 9,000; dynamic bbox-crop = grader cost fiction
  (§39 warning). Second channel read (c9 for a hole approach) = +1024 F32 ⇒ hole route dead.
- **Output cond = 900 FORCED.** `Where(cond,cyan_vec,input[1,10,30,30])` needs cond at output
  resolution `[1,1,30,30]` BOOL = 900. Priced alternatives all worse: Equal-label-canvas = 900 +
  must rebuild all colors (needs c9); slice+Concat reassembly = 16,616; additive `Mul(mask,swap)`
  delta = 36,000. BOOL is min dtype (1 B).
- **Detection = 2816 IRREDUCIBLE** (blue 256 + 6 peels 1536 + regrow 768 + cond_crop 256). Peel
  count 6 is the info-theoretic floor: cycle-vs-path is a *global* per-component property; each
  QLinearConv is a single linear threshold = ONE topological peel-layer (a 2-level-nested "keep iff
  ≥2 nbrs each with ≥2 nbrs" cannot collapse into one conv), so K layers need K planes; p5 is
  provably NOT linearly separable (52 err-windows; `shofamily_p5` = 242/266 visible-FAIL), p4
  window-contradicts. Global CC labeling (the only non-iterative route) has no cheap ONNX primitive.
- **Dominant intermediate is forced to 2D 16×16 U8.** A per-pixel recolor decision needs 2D spatial
  planes; channel-histogram `[1,10,1,1]` (#123) or 1-D projections lose the spatial structure. U8
  (256/plane) is the min dtype×area. So no "different dominant shape/dtype" formulation is viable.

**Conclusion: WALL at 4875.** memory 1024+900+256+1536+768+256 = 4740, params ~135 (all minimal,
no byte-equal shave), = 4875. The only sub-4875 lever (K5 regrow, 4828) regresses correctness below
LIVE. Lower cost is **not reachable** by any peel/flood/hole/template/projection route; a break
requires an ONNX primitive that computes a per-4-connected-component cycle aggregate and broadcasts
it without flood/label/regrow planes AND a non-F32 read of the F32 input — neither exists.
