# Lessons — NG

Durable lessons accumulated across sessions for the NeuroGolf project. Append at session end if a lesson is worth carrying forward (and is not already covered by the universal master).

Format:

```markdown

> Phase-4 (2026-06-05): pre-June entries -> `_archive/lessons_pre0601.md` (grep-only); 0 duplicate-titled entries removed; strict newest-first.

## 2026-07-15 - archive cleanup: convention docs need audits; empty load-bearing files are worse than absent files (Agent D)

- **A convention doc is a claim about behavior, not proof of behavior.** `Logs/README.md` described live session logs, dated scan buckets, and a `runs.jsonl` telemetry stream, but the session-log convention died after 2026-05-25 and 413 root-level scratch files accumulated in direct violation of the "do not write at Logs/ top level" rule. Future cleanup should audit adoption before preserving a process doc.
- **An empty "load-bearing" file is worse than no file.** `Logs/runs.jsonl` was 0 bytes while the README promised per-run telemetry. Keeping the empty file made agents trust a non-source. Archive rule: either keep the data with an honest freshness note, or remove the file and fix the pointer.
- **Fold the durable lesson, delete the run-history.** The useful transferable bit from the root cleanup logs was not the file-by-file blow-by-blow; it was the failure mode: rules without enforcement decay, especially under parallel agent pressure. That lesson now lives here; the disposable cleanup logs and scan/session dumps can stay in git history.

## 2026-06-10 — task258 Conv bias threshold for §FREE-OUTPUT BOUNDARY LEAK fix; prompt's ~90 target was overestimated (worker-3)

- **BIAS IS THE FIX FOR §FREE-OUTPUT BOUNDARY LEAK in output-writing Conv**: when a Conv has positive off-center taps (K[ch,c,0,+1] > 0 or K[ch,c,0,-1] > 0) that fire at out-of-grid padding cells, adding a BIAS that requires the CENTER cell to also activate solves the problem. The bias threshold forces ch_center ≥ 1 (active cell), so padding cells (all inputs = 0) can never reach the threshold. Previously only negative off-center taps were considered safe; the bias-threshold approach extends this to positive taps.
  - Concrete: ch2 = center_ch0 + left_ch1 + right_ch1 + bias=-2.5. At padding cell: 0+1+0-2.5=-1.5≤0. At interior: 1+1+1-2.5=0.5>0. Bias -2.5 is the threshold that separates the cases.
- **Pad [1,1,30,30] BOOL = 900B intermediate dominates multi-node cost floor**: for tasks with 10×10 active grid but 30×30 output, the Pad step from active-grid intermediate to full canvas costs 900B memory, making any multi-node approach cost ≥ 930. Single-node approaches (cost = params only) are the only path to cost < 160 for such tasks.
- **~90 estimates in task prompts may undercount Pad intermediate**: the prompt for task258 claimed ~90 based on the CumSum/MaxPool span approach. The actual cost floor for a correct multi-node approach is ~1828 (MaxPool @ 10×10 with Pad). The 900B Pad intermediate was not counted.
- **Prompt's "~90 target" vs actual: 90 assumes cost ≈ params only, but cost = memory + params. Whenever a Pad to canvas-size BOOL is required, the Pad output = 900B intermediate trumps all param savings.**

## 2026-06-07 — task209 byte-equal re-plumb 9,487→5,826→5,251: clip-dup phantoms, dead i8 Where, nibble-packed planes, and "the moment-check is load-bearing" (worker-7)

- **A byte-equal target makes radical re-plumbs cheap to verify**: 4 successive full rewrites
  (v3/v4/v5/v6, −45% total; v5 SHIPPED +0.49 EXACT mid-session, v6 staged byte-equal on top)
  each gated `ab --full` byte-equal in ~2 min — the 200k byte-equality gate turns "risky
  structural rewrite" into "mechanical refactor with a perfect oracle". When LIVE is already
  a correct solver, hunt byte-equal-at-lower-cost FIRST; bet-class divergence only after that
  well is dry.
- **A "pinned" floor component fell to a transport-packing idea-class on the 3rd forced pass**:
  the entry conv was "pinned at 1,620B" in two written floor analyses before the nibble-pack
  (2 cells/byte via kernel [1,10,1,2] stride-2, `packed>0 ⟺ content` keeps all boolean
  consumers packed) halved it. Anti-quit pings that force re-deriving a pinned component from
  scratch are worth honoring — floors are formulation-relative even when YOU wrote the proof.
- **CLIP-DUP PHANTOM class (new trap; cost 4 visible fails before diagnosis)**: shrinking a read
  plane changes `Clip`-overflow semantics from read-an-empty-row to DUPLICATE-the-edge-row;
  phantom duplicated guide cells made wrong hypotheses pass an exact moment test. Any plane-shrink
  must re-prove every Clip's overflow target empty or mask it. Invisible on corpora that never
  reach the edge — check edge-geometry pairs explicitly.
- **The bug double-paid as a floor proof**: the clip-dup failure mode IS what a weakened
  consistency check does (constraint-only selection mispicked real officials) — killing the
  "drop the moment-matcher, save 600B" ablation without building it.
- **i8 lanes are real but `Where(bool,i8,i8)` has NO ORT kernel** — route selects through u8
  (cast before the Where; wrapped negatives only land in unselected lanes). With `Min(u8,1)`-ArgMax
  extents and einsum rank-drops (`'bchw,c->h'` → [30]; `'bchw,bchw->'` self-dot) a geometry
  solver's scalar/vector plumbing shrinks ~25% with zero semantic delta.
  (Kit: `.update/2026-06-07_task209_i8lanes_clipdup_byteequal_replumb.md`.)

## 2026-06-07 — FREE-OUTPUT sweep: 12 conversions in one session, the pad-floor law, and the 0/9-with-perfect-intermediates trap (worker-7)

- **Sweeps pay when the recipe is mechanical and the gate is byte-equal**: 12 boladi `Pad→output`
  tails converted (+10.24 pts staged), every one ab --full 200k byte-equal on the FIRST gated build.
  The non-first-try builds failed at OFFICIALS (104 synthesized-mask guess, 353 missing active
  mask, 194 quadrant misread) — cheap local failures, zero corpus burns.
- **The pad-floor crossover law prices conversions in 10 seconds**: Equal-tail = fixed ~900 B
  padded canvas + ~60 B chain. prepad ≤ ~1,000 B ⟹ skip (or find a different #67/#78 family).
  13 of 25 boladi candidates died to this test BEFORE building — the skip list took minutes.
- **Perfect intermediates + 0/9 officials = decode-geometry bug, not logic bug** (353): canvas
  value 0 decodes ch0=True across the whole canvas region; varying-size grids need the
  active-rectangle sentinel mask (full law in `.update/2026-06-07_freeout_sweep_padfloor_law_worker7.md`).
- **Never synthesize what LIVE already stores** (104 anti-diagonal masks, 194 rot-vs-transpose):
  read the canonical inits/grids and derive the EXACT map; 1-2 train pairs under-determine it.
- **Tool persistence**: this sweep's scanner was written 2026-06-06 and LOST (job-tmp). Rewritten
  into `tools/scan_freeout_candidates.py` + the previously-missing gate-5 tool
  `tools/agent_helpers/profile_real_cost.py`. If a prompt says "find or rewrite X", PERSIST the rewrite.

## 2026-06-07 — Scalar dyadic-moment decode deletes the last 2-D detection tensor; diagonals still force the value-canvas output (task260, 12,074 → 1,522 byte-equal, 6 rungs)

- **Dyadic moment decode (the −600 v5 breakthrough)**: when a shape family has a small
  enumerable parameter domain (gray-wedge gaps {0}∪[2,8], brute-forced from the generator's
  anchor constraints), ONE scalar einsum `Σ_shape 2^±(r−c)` encodes BOTH sides at once as
  `2^±d·(code + frac)` — codes {0,8,22,52,114,240,494,1004} spaced ≥14, fractional bleed
  < 7.004 → 7 mid-band thresholds + ΣBool rank recover the gap EXACTLY. F32-safe because the
  ULP-span (code-range/granularity = 2^18) < 2^24 and the uniform 2^d factor scales out;
  normalize via Gather(pow2-table, d-index) = exact exponent shift. The radical rung stayed
  BYTE-EQUAL vs LIVE over 200,266 pairs. Beats perfect-hash LUTs (no linear hash p≤128
  exists for the same key; p=219 → 438p) with zero search risk.
- **Information-conservation pricing**: params and intermediate bytes price IDENTICALLY per
  element, so init-vs-runtime swaps of the same map are dead heats (±6 measured twice). Real
  wins exploit STRUCTURE: constant regions → Pad; small domains → k-space [19] vectors +
  one Gather; diagonals are permutation-rank → no factored final op, value-canvas u8 →
  Pad(sentinel) → Equal(ch_ids) is the output floor (1,019 B + 105p for 10×10-in-30×30).
- **Scorer dummy-probe robustness trap**: `score_network_v2` runs the graph on zeros where
  derived quantities leave their contract range (lc=0 → idx 19) → data-dependent Gather
  index OOB → scorer None DESPITE visible 266/266 + truth 200k/200k green. Clip-bound every
  data-dependent index so the graph is total on ANY input.
- **ReduceSum(u8): NO kernel** (spec rejects u8 — probed 2026-06-07); ReduceMax/Min u8 fine.
- **Affine F16 einsum-dot folds bool-rank ladders**: `Greater[k] → Cast F16 → Einsum('k,k->')`
  with weights like [4,1,1,…] (rank+3-or-0) or [252,−1,…] (253−rank-or-0) replaces
  ReduceSum+gate+Add/Sub+Where per decode at half the cast bytes (F16 ≤2048-exact, #77).
- Corpus-perfect 1-D shortcuts can be officials-fatal (§GENERATOR-INVARIANT): marginal-only
  anchor extraction and the triangular-number scalar route both verify on ALL 200k random
  pairs (exactly 1 wedge) and provably break on 2-wedge officials (train[2]). The dyadic
  decode survives BECAUSE both wedges ride the two einsum orientations simultaneously.

## 2026-06-07 — Free-output convs leak through positive off-center taps; 204k green gates cannot see it (task125, 12,905 → 3,668, 5 rungs)

- **QLinearConv-as-output generalizes #67 beyond Equal-tails**: X = Concat(s, e, P) u8
  [1,3,15,15] + W[10,3,3,3] i8 + asymmetric pads ⇒ the [1,10,30,30] one-hot explosion AND
  the 3×3 neighbor-count A never exist as tensors. Sign-consuming rules want sign-arithmetic
  end-to-end: the Equal-tail variant priced 5,700 vs 3,668 because exact canvas VALUES need
  exact 0/1 indicators (threshold tax). u8 lanes halve every plane (don't-do "MaxPool ≠ Conv
  counts" only forbids value-level byte-equality vs LIVE — indicator-sign consumers use
  MaxPool dilations freely, 0 params).
- **THE TRAP (new, library note `.update/2026-06-07_task125_qconv_freeout_boundary_leak.md`)**:
  any free-output conv channel with POSITIVE off-center taps on a content channel fires
  phantom cells OUTSIDE the active grid when content sits at the grid edge — and every
  standard gate (visible 265, truth 200k, ab 200k) stays green because the generator never
  places content there (0/200k). Found only by directed synthetic border stress (500 grids:
  v1 315/500, v4 223/500, leak-free v5 500/500). Negative off-center taps are safe (u8
  saturation); center-only-positive (1×1 kernel + explicit a01 = MaxPool 3×3 channel) is the
  structural fix at +434 cost. Pattern for the fleet: corpus-certified margin rung as
  primary + center-only-positive sibling staged beside it; and NEVER write "assumption-free"
  in a manifest without an adversarial test that targets the specific assumption.
- **In-graph constants are the in-grid witness**: ch8 ("far" cells keep bg) must fire inside
  the grid on all-zero neighborhoods but stay silent outside — impossible from biases (leak
  everywhere) or any linear taps on content channels (LP-infeasible, twice); the only
  carrier is an e = 1−s channel whose Pad-ring/Sub-ring is 1 exactly on the grid. Budget it
  from the start when designing free-output convs.
- **Round-2 (sub-2,000 push, same day)**: variadic `Min(u8×4)` works in ORT at opset 14 —
  n-way AND of indicator planes = ONE charged tensor (−338 here; Boolean cousin of #75's
  `Sum`). Dir-subset discount measured DEAD (any 3-of-4 ray-AND: ≥359k FP/200k — boxed needs
  all four). Exact-value canvases (Equal/OneHot tails) need ~2 interaction channels (aP, sP)
  beyond sign-predicates — for a `>0`-decoded task, never build values. OneHot maps
  out-of-range indices to ALL-OFF (free padding-class handling for other tasks, but i32-only
  indices ⇒ 3,600 B canvas). Floor lower-bound matches the v6 construction tightly at 3,330
  (25.8%): F32 entry 845 + 4-dir+Min 845 + e 225 + pads/X 1,125 + params 290. Under-2,000 is
  structurally unreachable for a corpus-correct task125 model — documented so nobody chases it.

- **Diagonals defeat every factored final op.** A full-extent diagonal is permutation-rank as
  an (r,c) pattern: `Equal(P[1,ch,30,1], Q[1,1,1,30])` fires one cell/row max, BitwiseAnd
  bit-sets need rank-1-per-bit, broadcast-Add rows must be inclusion-nested — none can write
  a diagonal (let alone its bg complement). For diagonal/permutation-class outputs go straight
  to #67 Equal-tail (u8 value canvas → Pad(sentinel) → Equal(ch_ids)); pre-output floor is
  the [1,30,30] u8 plane (900 B) + its cheapest producer (Pad of a 10×10 canvas, 100 B).
- **§GENERATOR-INVARIANT trap, live instance**: gray-triangle anchors are extractable from
  1-D marginals alone (`anchorDiag = maxGrayRow − minGrayCol`) and that verifies on ALL 200k
  random corpus pairs (random path emits exactly 1 triangle) — but 2-triangle pairs exist in
  OFFICIALS ONLY (train[2]) and the marginals mix the triangles → VISIBLE-FAIL. Price the
  officials-only quirk set before deleting any 2-D view a corpus-only proof says is removable.
- Detection-side pattern that priced best: free-input einsums straight off `input`
  (4-op crop-plane for joint-(r,c) extremes; SCALAR einsums `'bchw,c->b'`/`'bchw,c,h->b'`
  for counts and positional probes) + validity-collapsed Equal targets
  (`Where(valid, t, t0)` dups a guaranteed-true mask instead of And-gating each plane).
- **Derive scalars from structural counts before building marginal machinery**: the line is
  full-extent for every generated pair ⟹ |d| = 10 − lineCount and sign = one row-0 scalar
  probe — deleted a 246 B marginal block down to 22 B. Same trick for the color id:
  one-nonzero-entry vectors give `value = Σ(idx·count)/Σ(count)` as an EXACT small-int F32
  Div (≤90, integral quotient, Cast-U8 safe) — cheaper than ArgMax on a [1,10] vector.
- **#74 u8 lanes pay across a whole graph** (opset 14): u8 init planes for compare targets,
  Where sentinels that double as Pad values (0/255 reused), u8 wrap-around `Add(x, 254)` in
  place of an unprobed `Sub(u8)`. Extraction lane F16→u8 halved every plane (−~460 over v1).

## 2026-06-06 — Sign-trick planes + digit-packed counts: counting rules need no clamp chains (task078, 15,756 → 842)

For any per-row/per-column COUNTING rule (gravity/sort/stack families), the `>0.0`
decode means output channels only need the right SIGN — never re-clamp to one-hot:
1. **Sign-trick planes (F16-safe)**: `g = count − (r+.5)` IS the channel plane; the
   half-integer offset makes every value tie-free (|margin| ≥ 0.5, F16-exact). A
   middle band (color between two thresholds) is `ReduceProd(axes=[1])` of the two
   boundary margins — both-positive provably impossible ⟹ product>0 ⟺ in-band.
   Killed LIVE's 8 Sub + 5 Relu clamp chains (task078 v1: 15,756 → 1,611, ab byte-equal).
2. **Bool tail beats F16 arithmetic when comparisons are tie-free**: `Greater(n1, r+.5)`,
   `Greater(r+.5, n12)`, middle band = `Equal(b_lo, b_hi)` (XNOR; both-true impossible).
   traps §12 escape clause applies — bools DEFINED by exact comparisons, not
   reinterpreted floats; gate per §NOVEL-OP = full-corpus ab (200,266/200,266). (v3, 961)
3. **Digit-pack multiple counts into ONE einsum stream**: sel coefficients {c1: 101,
   c2: 100} emit `packed = n1 + 100·n12` ([1,1,1,30] = 120 B vs 240 for two). The
   hundreds-scale compare works DIRECTLY on packed (low digit ≤ 10 < 50 can't flip
   it); only the low digit needs `Mod(packed, 100, fmod=1)`. (v4, 842 = mem 800 + 42p)
4. **Einsum size-1 free subscript**: equation `bchw,kjc->bkjw` with sel [k,1,10] emits
   the rank-4 [1,k,1,30] directly — deletes the Unsqueeze a rank-3 output would need.
5. Matrix gaps verified on CPU EP: **ReduceProd F16 ✓** (absent from ort_reality §7
   until now), **Mod F32 fmod=1 const-divisor ✓** (exact on integer-valued floats ≤ 2^24).
Ladder staged: v1 1,611 safe primary (+2.28) → v4 842 free probe (+2.93 total predicted).

## 2026-06-06 — Hypothesis-select graphs: scalar-einsum tests + N−1 elimination + MAPS-init gather + centroid bits (task124, 15,387 → 2,149)

For "test K completions, output the consistent one" graphs (LIVE task124 ran 6 parallel
Gather→Sub→Mul→ReduceSum→Less SSD branches + Where/Max merge), five composable moves:
1. **Equality tests as SCALAR einsums — intermediates live in kernel scratch, cost-free**
   (#46/§42 addendum): SSD between row-slices = a factored quadratic form
   `Einsum('rc,abcd,re,abed->', L, P, L, P) → Less(·, 0.5)` where L is the [3,5]/[2,5]
   row-DIFFERENCE factor (store L, 15p/10p — not K = LᵀL, 25p). The [5,5] Gram and every
   marginal exist only inside the einsum kernel; `calculate_memory` never sees them. Each
   fusion rung (3-op, 4-op, repeated operands, repeated subscripts across operands)
   landed EXACTLY at its static prediction. Whole detection stage = 4 einsum scalars.
2. **N hypotheses need N−1 tests** (elimination): when the generator guarantees some
   hypothesis matches, the last is the Where-chain default — a whole test deleted.
3. **Late tests run in a PROVEN regime — replace tensor equality with a scalar moment.**
   Once the exact tests fail, the survivor set may be 1-bit ("shift is 1 or 2"), and a
   shifted row obeys `W₂−W₀ = s·S₀`: `dW = Einsum('abcd,c,wdxy->', P, rowsel±, Wc)`
   (the colorize Conv weight REUSED via size-1 subscripts, 0 new weight params),
   `S₀ = Einsum('abcd,c->')`, `m21 = Less(dW, 1.5·S₀)` — margin ≥0.5·S₀, F32-exact:
   13 B replaces a 244-unit slice/SSD test. Moments are ONLY safe intra-regime —
   cross-class moment collisions are real bugs; keep exact tests as the outer gates.
4. **Select the index MAP, not the candidate**: `Gather(MAPS [K,…] INT32-init, h) →
   Gather(flat, M)`, vs materializing K candidates + Where/Max merge (≈3,600 B at LIVE's
   K=6). Scalar h from a `Where(m,i,…)` chain (12 B). Priced alternatives (arithmetic map
   synthesis in every layout incl. zero-tail negative-wrap — row-1 OOB wraps positive
   into row 0, 4 disproofs; ConvTranspose pattern-stamping with runtime-gathered
   placement kernels; runtime-starts Slice → 025-class None): all net-worse — the
   [rows,10] INT32 index tensor is unavoidable and params==memory in cost units.
Bonus moves: a generator-proven always-zero cell (task124: row 0 cols 6–9, stamp col ≤ 5)
makes a free OOB sentinel — point OOB map entries at it and delete the Concat(0) (−205 B);
hypothesis-INVARIANT output rows (the dr=0 region) come verbatim from the input plane —
slice them and shrink MAPS to the variant rows (−57); §41 negpad Conv accepts negative
BEGIN pads too (ORT-probe-verified, techniques §41 addendum — superseded rungs needed it,
the final doesn't). When verification has a window-size choice, prove the minimum: 4 rows
admit a real (3,0)-vs-(2,s) false-positive family; row 5 makes it jointly impossible.
Ladder + floor argument: `task124.md`; staged `staged_submits/task124_v11_mapselect_lfactor_cost2149/`
+ v6 fallback (every rung v8-v11 fully gated: visible 267/267, truth 200k/200k, ab
byte_equal; final −13,238 = ln +1.97).

## 2026-06-06 — U8 vector machinery + 1-based-iota anchor compare beats F32+CumSum for single-anchor location logic (task199, 16,157 → 3,179)

For "find the one anchor's row/col, then build row/col gates" graphs, don't run LIVE's
F32 chain (full-plane dot mask → Cast F32 → CumSum → Cast BOOL, plus Mul/ReduceSum for a
weighted scalar). Cast the encoded plane to U8 once and stay there: per-axis
`ReduceMax(u8)` 15-B vectors → `Greater(u8)` indicators → `Where(ind, iota1_u8, 0) →
ReduceMax → LessOrEqual(iota1, r_s)` for the "rows ≤ r" gate (46 B vs 135), and
`Where(ind, parity_u8, 0) → ReduceMax` for the anchor's parity/weight scalar (46 B vs 139).
**Use 1-BASED iota**: `r_s ∈ {0=no-anchor, r+1}` makes the no-anchor case all-false —
byte-matching CumSum-suffix semantics even on off-contract inputs (0-based iota leaked
row 0 and cost task199-v2 its shift/shrink adversarial parity; same cost to do it right).
`LessOrEqual(u8)` opset-13 kernel confirmed (matrix row added to `ort_reality.md`).
Combined with §41 negpad-Conv encode + the free-output Equal tail (the task148 entry
below), the whole family prices at ~2× the two 900-B planes (15×15 F32 encode + 30×30 U8
canvas). Worked ladder + floor proof: `task199.md`; staged at
`staged_submits/task199_v3_u8vectors_cost3179/` (byte-equal 200,266/200,266 vs LIVE).

## 2026-06-06 — Make the EXPENSIVE op the output node: free-output Equal halves an Equal-broadcast tail (task148, 18,388 → 4,461)

§39 says tensors literally named `output` are free in BOTH the static walk and the profiler
trace — the unexploited corollary is an op-ORDERING rule: when a tail ends
`canvas → Equal(ids) → [1,10,h,w] → Pad → output`, flip it to
`canvas → Pad(value=sentinel) → [1,1,30,30] u8 (900 B) → Equal(ids) == output` and the
10-channel one-hot explosion costs nothing (task148: 2,880 → 900). Pad value must be a
non-color sentinel (reuse the canvas Conv's out-of-grid bias value) so padding decodes
all-False. Generalizes to every #23 Equal-tail / one-hot-assembly task — grep for
`Equal → Pad → output` endings; spread-pattern candidate. Three more facts banked from the
same ladder (probes + grader-exact static scoring): 3-operand free-input Einsum
`'bchw,c,w->bh'` is temp-free like §42's 2-operand form (joint channel+weighted-axis row
reads at 120 B); INT8/U8 row-vector lanes halve F16 for small-integer machinery — but
narrow-int Add/Sub/Mul need OPSET 14 (13 fails shape inference), ArgMax(INT8) works,
GatherElements indices stay INT32; **u8 wraparound interval** `Less(Sub(c,lo), span)` does a
2-threshold band mask in TWO plane ops (c<lo wraps ≥226 > span — GE+GE+Xor 864B → 576B);
**rank-3 einsum birth** `'bchw,cx,w->bhx'` (selector [10,1]) emits [1,30,1] row vectors
directly, deleting the Slice→Unsqueeze rank-lift pair. Method note: the whole 6-rung ladder
(v5 8,302 → v10 4,461, all byte-equal/200k) took one session BECAUSE the corpus analysis ran
first — 9 generator invariants (equal group heights, seeds-one-group-only, never col 0,
groups never touch borders) each deleted ops/masks/Clips before any ONNX was built.
Details: `task148.md` + `Onnx_Advanced/.update/2026-06-06_task148_freeout_equal_einsum3_i8lanes.md`.

## 2026-06-06 — Officials can VIOLATE generator invariants: gate lab candidates on corpus AND officials (task077 v1, 19,422 → 7,235)

ARC-GEN `generate(width=None)` runs constraint loops (task077: "every box row/col has an
empty cell" legality retry); but officials are reproduced via `generate(colors=[...])`,
which SKIPS the whole construction block — hand-made ARC pairs need not satisfy ANY
invariant the generator enforces on its own draws. task077's corpus is 200k/200k clean for
a formulation that officials[0]/[1] break (boxes with a no-2s middle row / no-2s column).
A corpus-perfect candidate still dies at visible (§VISIBLE-FAIL) — the rule lab must score
candidates on BOTH from the first iteration; the officials' quirk classes (here: 1 missing
row + 4 missing cols across 266) then become explicit repair sub-rules (width-1 V5-hole
fill + one vertical between-fill, each provably harmless on the generator distribution).
Second confirmation of the task205 lesson below (numpy-lab-first: v1 ONNX passed every
gate on the first build). Bonus gadgets that shipped: (a) QLinearConv w=[1,2,1] y_scale=3
computes round((a+2b+c)/3) = b OR (a AND c) — a 1-op hole-fill/between-fill that replaces
4 shift/Min/Max planes per use; (b) §DUAL-TRUTH counting certified more-correct-than-LIVE
with zero per-pair inspection (ab_divs 3,074 == LIVE_misses 3,074 − cand_misses 0); (c)
orientation lemma for separated-component fills: mask the TIGHT axis (±2 rows), reach along
the LOOSE axis (±6 cols) — the mirrored design admits third-box bridging and is unsound.

## 2026-06-06 — Simulate the restructure on the corpus BEFORE building the graph (task205 v18–v25)

The locator-crop rewrite (27,525 → 10,528 byte-equal-FULL / 9,741 risk rung) was designed and
de-risked entirely in a numpy corpus simulation first (`loc_sim.py`: full LIVE pipeline vs
crop-windowed variant, 200k pairs, margins swept, tie/no-fire semantics modeled down to the
accumulator level) — the first ONNX build then passed visible + ab byte-equal on the first try,
and 7 rungs landed in one session with zero ORT debugging rounds. ~3 min of sim per design
question vs a build+gate cycle each. Corollary: read the ARC-GEN generator source FIRST
(task_8731374e.py) — bg density, cross-pixel bounds (1..3, interior, distinct rows/cols), and
offset≥1 were the facts the whole design hung on. Library intake:
`.update/DONE_2026-06-06_task205_locator_crop_v18-v25.md` (locate-then-detect, separable
ones-kernel banks, 0-dim-init scorer trap, rank-4 broadcast trap, tie-clamp byte-equality pins).

## 2026-06-05 — Local truth-full timeout ≠ grader risk: divide by corpus size first (task270)

A 2,978 B candidate whose 200k truth run timed out at 904s scored EXACT on Kaggle: 904s/200k =
4.5 ms/pair; the grader evaluates ~1k hidden pairs ≈ 4.5s. Rule: before treating a local
timeout as a risk signal, compute per-pair latency — fear it only if (per-pair × ~2k) approaches
grader limits. Gate slow candidates with truth --n 20000 instead of --full and note the cap.
## 2026-06-04 — Captain session: live-zip audit via score-delta reconciliation recovered +2.39

Walking the Kaggle submissions list and reconciling each past submit against base+Σln(deltas)
certified six hidden-confirmed winners (238 v4 / 383 v5 / 286 exact-53814 / 054 v2 / 018 v10 /
233 v5) that had never been folded into the local zip; one fold-submit recovered +2.39. An exact
delta-match doubles as a hidden-correctness certificate even when the raw score LOOKED like a
regression (stale rebased base). Full method + the session's other captain lessons (ladder-verdict
probes, sibling discipline, 13/13 exact calibration): `The_400_Tasks/current_strategy.md`
§captain-session-2026-06-04.

## 2026-06-04 -- Euler number for 4-conn CC counting: formula sign, border padding, hole detection

**Rule:** count 4-connected components of color 8 -> N; output NxN identity (task325).
Three traps discovered building the ONNX model from scratch:

1. **Euler formula sign for 4-conn foreground is +2*nD (not -2*nD).**
   (Q1 - Q3 + 2*QD) / 4 gives E4 = C4(fg) - H4(fg) with +sign for 4-conn fg.
   Using -2*QD gives the 8-conn formula -- wrong here.

2. **Must pad mask with 1-zero border before computing 2x2 quad sums.**
   Grids that use all 16 columns (e.g., 9x16) have fg cells at the computation boundary.
   The local Euler formula assumes infinite background; without padding, boundary fg cells
   corrupt the count (E=2 instead of 5 for a 5-component grid). Fix: Conv with pads=[1,1,1,1]
   on the 16x16 mask instead of separate Pad node + valid Conv.

3. **Hole detection requires 8-neighbor check (nbr8_sum > 7.5), NOT 4-cardinal.**
   A single-cell bg cell is an enclosed 8-connected bg region (H4) only if ALL 8 neighbors
   are fg. Using 4-cardinal check (nbr4 > 3.5) gives false positives: a cell with all 4
   cardinal nbrs=fg but a diagonal bg neighbor can escape via 8-connectivity, so it's NOT
   an enclosed hole.

**Cost result:** 50733 -> 19528 B (-62%, +0.95 pts). Build: task325/prototypes/build_task325_v2.py.
Euler approach costs ~19k B vs label-propagation which would cost 30-65k B (32 unrolled MaxPool steps).

## 2026-06-04 — task279 flood-fill: 8-conn for BFS border-flood, 4-conn for CC flood

**Rule:** a c1 component encloses a 9-hole iff there is a 9-cell that can't reach the 16x16 border via an 8-conn c9-path without crossing a c1 wall. Key implementation lessons:

1. **8-conn MaxPool for border BFS (c9-flood), but 4-conn Conv for CC expansion.** Using 8-conn MaxPool for CC expansion merges diagonally-adjacent but 4-disconnected c1 components, causing false positives (wrong cells becoming 8). LIVE uses a 4-conn kernel `[[0,1,0],[1,1,1],[0,1,0]]` for its Conv CC flood.

2. **Padding cells drive flood seeding: use `1 - c1 - c9` not ch0.** For the seeding of the border flood, the padding cells (ch0=1 in the ARC scorer's encoding but all-zero in `encode_grid`'s encoding) can be identified as `1 - c1 - c9` = 1 wherever neither color-1 nor color-9 is present. This works regardless of whether padding cells have ch0=1 or all-zero.

3. **Gate on N_CC=4 steps (matching LIVE), not 2.** Corpus measurement says max 2 4-conn CC steps from adj_1, but hand-crafted visible pairs need up to 4 steps from enc_9 (3 expansion steps after initial seed).

4. **Canvas+Equal output saves ~10k bytes over 10-channel Concat.** For tasks with ≤4 output colors: U8 single-channel canvas → Pad to 30×30 → `Equal(canvas, ch_ids[1,10,1,1])` = BOOL output (free, not counted). Mask padding cells to sentinel 255 so Equal gives all-False there.

5. **v1 achieves 25,273 vs LIVE's 50,462 (-50%). 0 real bugs in 10k synth; v1 actually MORE correct than LIVE (1/10k case where LIVE is wrong, v1 is right).** Expected +0.69 LB.


## 2026-06-04 -- K=1 MaxPool label propagation fails; K=5 U8 needed for task325

**K=1 with [3x3] 8-connectivity MaxPool is NOT sufficient for 4-connected CC counting**,
even when no two 4-connected components are 8-adjacent (verified 0/200k pairs).

Root cause: a fg cell can be a local 8-neighborhood maximum but NOT the global maximum
of its 4-connected component. Example: L-shaped component where the highest-ID cell
is >1 8-step away from another local max.

**Proof by construction:** If cell Y has no fg 8-neighbor with higher ID, that means:
Y has no fg cell at (r,c+1), (r+1,c-1), (r+1,c), (r+1,c+1). The path from Y to the
global max M must go UP or LEFT first (lower IDs) -- but then it can go RIGHT along
a higher row (increasing IDs) to reach M. So Y can be a local max without being global max.

**Fix:** K = max 8-conn eccentricity of the highest-ID cell across all components = 5 (on 200k pairs).
Use K=5 steps with U8 IDs (r*16+c, 0..255). Between steps, Where(mask, MaxPool, 0) prevents
cross-component contamination via bg cells.

**U8 vs F16 for IDs**: U8 (1B/elem) = 256B per [1,1,16,16] intermediate vs F16 (2B) = 512B.
MaxPool supports U8 in ORT. IDs r*16+c range 0..255, fitting U8 exactly.

**Canvas output**: Replace Concat with zeros_7ch (1792B param) via: canvas U8 [1,1,16,16]
{diagonal=8, off-diag=0, inactive=10} -> Equal(canvas, ch_ids[0..9]) -> [1,10,16,16] BOOL.
Saves ~2000B in params. ch_ids only needs [1,10,1,1] U8 = 10B.

Result: task325 cost 50733 -> 8813 B (-83%), +1.75 pts. Build: task325/prototypes/build_task325_v4.py.

## 2026-06-04 -- task325 v5: corpus-bounded output canvas + constant-folding Wheres into initializers

Three cheap wins stacked on the K=5 U8 MaxPool propagation (8813 -> 5646 B, -36%):

1. **Size the output canvas to corpus maxN, not the propagation grid.** Propagation needs
   16x16 (input bound) but the output identity never exceeds 6x6 (corpus maxN=6 over 200k).
   Building [1,10,6,6] BOOL (360 B) + Pad [0,0,24,24] instead of [1,10,16,16] (2,560 B)
   saved ~2,900 B across output_16/canvas/active_2d.

2. **Fold data-independent Where chains into initializers.** diag_canvas = Where(diag_const, 8, 0)
   has no data dependency -- precompute as a U8 initializer (same params, -256 B mem, -1 node).

3. **Shape range constants for direct broadcast.** Storing range6 as [1,1,6,1] and [1,1,1,6]
   F16 inits lets Less(range, N_scalar) produce row/col masks directly -- drops active_1d
   plus 2 Reshape nodes and their shape inits.

Also: Cast(F32->BOOL) replaces Greater(x, 0.5) for one-hot masks (drops the 0.5 scalar; Cast
to bool is exact on one-hot 0.0/1.0 inputs).

**Floor-proof method worth repeating**: vectorized numpy replay of the exact ONNX pipeline
(stacked-shift max for MaxPool) over the full 200k corpus, sweeping K: K=3 -> 4,508 wrong,
K=4 -> 399 wrong, K=5 -> 0/200,000. Names the minimum iteration count with evidence instead
of a hand-wave. Same pass re-confirms maxN=6. Runtime ~2 min.

## 2026-06-04 — task279 v2: UINT8 masked-flood at opset 14 nearly halves flood cost vs F16

F16 flood intermediates (2 B/elem) are NOT the floor for MaxPool-based flood fill: **MaxPool(UINT8) works at opset 12+, Mul/Add/Sub(UINT8) at opset 14+, Max(UINT8) at opset 12+** (all per `ort_op_dtype_matrix.md`, now Kaggle-track-verified by the task279 v2 scorer run). Flood steps drop 1,024→512 B each. Four composable tricks from the same build:

1. **Border seeding without a border-mask init**: `Pad(padding_mask, 1ring, value=1) → MaxPool(3×3, valid)` — out-of-canvas IS padding semantically, so the padded ring seeds the 16×16 border for free. Kills a 256-param init + a Max node (−188 net).
2. **4-conn dilation without Conv**: `Max(MaxPool[3,1], MaxPool[1,3])` in UINT8. Conv is float-only; this keeps the CC flood at 1 B/elem. (4-conn is mandatory for CC — 8-conn merges diagonally-touching foreign components.)
3. **Arithmetic canvas beats Where-chain**: `canvas = c1 + 7·cc + 9·c9` (u8) — padding lands at 0 naturally, no BOOL casts, no Where, no 255 sentinel.
4. **Impossible-value ch_ids sentinel**: set `ch_ids[0]=200` so canvas=0 (padding) matches NO channel in the final `Equal` → all-zero one-hot at padding without masking ops.

Bit-packing the grid into uint32 lanes was priced and REJECTED: ~10 BitShift/BitwiseOr/Slice ops × 64-72 B ≈ 650 B/step vs MaxPool+Mul's 512 B/step. MaxPool does the whole 2D dilation in one op; op-count × tensor-size is the real metric. task279: 50,462 → 24,243 (v1 F16) → **15,860 (v2 u8)**, byte-equal to LIVE 2266/2266 + 200k/200k full truth.

## 2026-06-04 — task279 v3: read the ARC-GEN generator BEFORE optimizing flood structure

Generator mining (`external/ARC-GEN/tasks/task_b2862040.py`) turned two structural invariants into bytes where corpus statistics alone could not:

1. **8-conn-first CC step justified by box margin**: every component is a rectangular ring 3×3..5×5 with margin ≥2 between boxes, holes ≤3×3. So dilate-8(interior) covers the ENTIRE ring in one step AND provably stays inside the box bbox (cannot touch foreign components). One 8-conn MaxPool+Mul (512 B) replaces two 4-conn quartets (2,048 B). The remaining 2 quartets cover barnacles + the officials' handcrafted 2-px tails. Gate evidence, not just theory: byte-equal 2266/2266 + 200k/200k truth.
2. **Depth-histogram EV analysis kills seed enrichment**: erode-4×4 (all-9 windows are provably outside when holes ≤3×3) is a SOUND seed enrichment but the corpus depth tail {9:32, 10:8, 11:1 of 200k} lives in narrow inter-box channels with no open window — the worst case stays 11 steps. Cutting to 8 breaks 237/200k vs LIVE to save 256 B. Enrichment that doesn't move the WORST case is worthless when byte-equality is the bar; check the full histogram, not the mean.

Also: when officials are handcrafted (validate() with explicit pixel lists), they can contain structures the random generator never emits (the 2-px tails forcing CC depth 4→ even with ring-in-one-step, keep 2 extra 4-conn steps). Officials gate ≠ generator gate.

task279: 50,462 → 15,860 (v2) → **14,067 (v3)**. Floor proof: scaffolding 8,392 + corpus-mandated 11×512 flood = 14,024. Sub-10k requires ≤3 flood steps = breaks 45% of corpus. Sub-5k is below scaffolding alone.

## 2026-06-04 — Quantified iteration-floor proofs (task325 K-minimum, task279 depth histogram)

Two flood/CC tasks closed with floor proofs that are MEASUREMENTS, not assertions: task325
proved K=5 minimum (K=4 fails 399/200k) and killed the 2-hop shortcut by measuring gap-2
density (79,986/200k); task279 histogrammed required flood depth and showed a <10k budget
breaks ~45% of generator draws. Promoted to `Onnx_Advanced/techniques.md` §37. This is the
new bar for "proven floor" on iterative tasks.

## 2026-06-04 — Dual-truth divergence resolution (task279 capstone)

Worker ab windows can under-sample: captain's deeper ab (n=3000) found 1 divergence the
worker's 2,266-window missed. Resolution recipe: run `ng_gate truth` on BOTH the candidate
AND the LIVE extract over the same window — v3 read 5000/5000, LIVE read 4999/5000 ⇒ the
divergence is a LIVE GT-bug the candidate FIXES (task377 more-correct-than-LIVE class, ships
clean; scored exactly +1.277). Always resolve an ab divergence to WHICH side is wrong before
deciding anything.

## 2026-06-04 — "Staged" must mean an artifact in staged_submits/ (task325 protocol miss)

task325's summary said "staged" but only the build script existed (no onnx, no staged dir).
Captain rebuilt from `build_task325_v6.py` and verified the exact cost — fine this time, but
the contract is: a stage = `staged_submits/<id>/` with the onnx + manifest. Build scripts are
reproducibility, not staging.

## 2026-06-04 PM — Dynamic-shape candidates: local cost is a LOWER BOUND on grader cost (task014)

First non-exact landing after 24 exact-calibration submits: task014 v3 (dynamic single-channel
crop) scored −0.33 below prediction — Kaggle's profiler credited ≈6,093 vs local 4,380. Hidden-
correct (byte-equal FULL), so keeps-highest banked +2.01. Rule: for dynslice/dynamic-Gather
constructions, treat `score_network_v2` as optimistic; build the static-shape sibling when it's
within ~30% and ship both. Promoted to techniques §39 warning.

## 2026-06-04 PM — Check artifact mtimes against the prompt date before treating work as "finished" (task187)

A "187 finished" report turned out to be two-week-old prototypes (2026-05-21) that were ALREADY
the live model (byte-equal at identical 49,813 cost). The tell: every file predated the batch-6
prompt, and the staged dir's base ("on_5874_78") was an ancient LB. Captain check before firing:
`ls -la --time-style=long-iso prototypes/` + compare against the prompt timestamp + score-match
vs live (cost == live + byte_equal ⇒ it IS live, not a candidate).

## 2026-06-04 night — The visible-miss axiom is now beyond relitigating (3rd proof, task363 probes)

task363's no_diamond probe passed **synth 200,000/200,000** and still scored the exact fail value
from its ONE visible miss (264/265). Third independent confirmation (task193 ladder, task145,
task363 ×3): officials ⊆ hidden; one visible miss = certain 0; NO amount of generator coverage
overrides it. Promoted as an axiom in traps §VISIBLE-FAIL.

## 2026-06-04 night — Dual-truth by ARITHMETIC: counts alone can certify zero-real-bugs (task187 scanline)

When (ab divergences) == (LIVE GT-misses − candidate GT-misses) over the same window and the
candidate's misses are a subset with identical outputs, every divergence is LIVE-wrong→candidate-
right — 0 real bugs WITHOUT per-pair inspection. task187 scanline: 21 = 29 − 8 ⇒ fired, scored
+0.850 exact. Added to traps §DUAL-TRUTH.

## 2026-06-04 t119 drive-to-floor (43,122 → 3,390, +2.54 LB predicted)

- **Probe accounting questions before pricing architectures.** Two 5-minute probes settled
  what hours of pricing couldn't: (a) Einsum on the free graph input counts ZERO temps —
  `(0, 0, 10)` for a bare marginal (techniques §42, campaign-wide lever); (b) multi-output
  models score `None` — the "declare intermediates as outputs" exemption does not exist
  (traps §14z). One opened a −205 B subsystem; the other killed a whole idea class instantly.
- **A wrong heuristic can pass visible 266/266.** The V-detect-by-row-span shortcut survived
  visible and died only on the 500-pair corpus A/B (12 divergences). When a candidate is
  supposed to be byte-equal, A/B against the byte-equal parent is the real gate; visible is
  just a smoke test.
- **User pushback was right twice**: "try the einsum" overturned a NOT-WORTH-IT pricing whose
  two unknowns (temp accounting, V-flag soundness) both resolved favorably — the priced-out
  list is only as good as its unprobed assumptions. The sound V-flag needed rows OR cols
  (vertical-wall Vs repeat columns, not rows).
- **Infeasibility proofs are deliverables too.** Sub-1,500 for task119 is closed with
  receipts: 900-B full-coverage tensor floor + 432 two-line tail + ≥800 detection ⇒ dream
  floor ≈ 2,400. The single-Equal major-axis fold (disjointness insight) priced +156 OVER the
  two-Equal tail — written down so nobody re-derives it.

## 2026-06-04 night — §GRADER-GAP: the local scorer was fooled 37× (task398)

The v8–v15 boolpad/dummy-branch family scored 364–519 locally and ≈13,300 on Kaggle — every rung
equally. Local cost is a LOWER BOUND for any graph using dummy branches / optional inputs /
bool-Pad collapses / data-dependent shapes; parity is proven only for boring-static graphs (30+
exact landings). The saving graces: byte-equal floored the risk (+1.10 banked), and the worker's
fallback ladder let fixed-base probes decode the credited cost in one round. Static rebuild
(est. 1.5–2.5k honest) logged as the task398 follow-up.


## 2026-06-04 — task054 v5: strip-extracted runfill + ScatterND row-writes drop a "proven floor" 40%

The v1 floor analysis declared the 4x full-grid CumSum runfill "the irreducible core" (~27,900 B);
v5 reproduces it byte-equal (200,266/200,266 full AB) at ~6,100 B by running the SAME primitive on
Gather-extracted [4,30] marker strips and writing back via ScatterND row-writes + ScatterElements
column-writes (47,180 -> 26,043, +0.60 LB predicted). The floor wasn't the primitive — it was the assumption that the primitive
must run on the plane it paints. K-bounded sparse work (K=4 markers, sweep-proven) converts
full-canvas ops into strip ops. Two transfer-grade traps from the build: ScatterND duplicate
indices + per-strip exclusions silently lose cells (use MULTI-marker onehots so duplicate updates
are identical), and TopK indices have unspecified tie order (consume the values, never the
indices, when ties encode "absent"). Cost conservation laws are real: every algebraic
reformulation of the detection planes priced exactly 8x900 — write the conservation argument
down (task054.md Don't-do) instead of re-deriving it.


## 2026-06-04 night — The exact-margin close standard (task338)

task338 closed the right way: every remaining escape route priced with ARITHMETIC margins
("dead by ≥1,875 B"), kernel probes stamped into the dtype matrix as permanent knowledge, a
quantified lemma left in §40, and a closing verdict naming the missing ops that would change
the answer. ~38 dead-ends across 4 families + 3 non-linear routes. This — not "tried hard,
couldn't" — is what TASK CLOSED means now. (Floor proofs: §37/§16-grade; closes: 338-grade.)

## 2026-06-04 night — Absent-guard bug class: zeros/dummy inputs can satisfy presence guards (task065)

task065's root blocker: `ReduceMax(eq_canvas15)` as an "is color present" guard fires wrongly on
dummy/zeros profiling inputs because target-color selection collapses to background, making the
equality canvas all-True. Fix pattern: guard on `And(clr_pres, not_bg_mask)` — presence tests
must exclude the background hypothesis explicitly. Audit any guard whose truth value could
change between the zeros-profile input and real grids (same family as the §39 zeros-drift rules,
but for CORRECTNESS instead of cost).

## 2026-06-04 night — The convergence lesson (task094): dynamic local costs are fiction, not estimates

Three byte-equal models: dynamic 313 → credited ≈24,386 (78×); dynamic 451 → ≈19,182 (42×);
static 3,806 → EXACT and the winner by +1.6. The two dynamic credits CONVERGED to the same
absolute band — the grader traces true runtime shapes, so a zeros-run local profile of a
dynamic graph measures nothing. Static siblings aren't hedges; they're the expected ship.
The bracket-pair + boring-sibling pattern cost 3 free submits and decoded everything in one
round.

## 2026-06-04 night — The grader-input mystery SOLVED (task054 worker, source read)

`calculate_memory` profiles on real train+test+arc-gen inputs and charges max(static, runtime)
per tensor. Consequences now doctrine: (1) static graphs — local==Kaggle exact, submit-probes
buy NOTHING; (2) dynamic graphs — real inputs inflate runtime shapes, max() charges them
(the 094 convergence mechanism); (3) symbolic-dim exploits are patched (None). Also: a worker
proved a floor by READING THE GRADER plus scored probes — the new gold standard for "impossible".
And: task359's byte-equal +2.20 sat unshipped in staging behind a lossy summary channel —
when a summary cites a "safe vN", grep the task .md and staged_submits for un-shipped gold.

## 2026-06-03 — CumSum row/col rank is wrong when two 5-regions share a column (task368)

In the template-stamp rewrite for task368, computing `row_rank = CumSum(ch5, axis=2, exclusive=True)` seems like a cheap way to find each cell's row offset within its 5-region. It works when each column has at most one ch5 run, but breaks when two separate 5-regions share the same column (e.g., pair 1: regions at rows 0-2 AND 5-7 both span cols 6-7). The CumSum counts ALL ch5 cells above without resetting between runs. Fix: use the **consecutive-AND approach**: `above_k = And(ch5_b, cd1, ..., cd_k)` where cd_k = ch5 shifted down k rows. `row_rank = Cast(above1) + Cast(above2) + Cast(above3)` — this correctly gives 0 at the first row of any run because it checks that ALL of ch5[r-1,c], ch5[r-2,c], ... ch5[r-k,c] are 1, which resets at any gap. Cost ~5200B for the INT32 chain (ORT rejects INT8/UINT8 Add/Mul).

## 2026-06-03 — UINT8 Where/Gather/Equal all work at opset 13; UINT8 Add/Mul do not (task368)

In NeuroGolf, ORT rejects INT8/UINT8/INT16/UINT16 Add and Mul. Only INT32 and INT64 work for arithmetic ops. But UINT8 IS supported for: Where (opset 13+), Gather (data tensor), Equal, ReduceMax, Cast. This asymmetry means: use UINT8 for "select/compare/lookup" ops to save 75% vs INT32 (100B vs 400B per [1,1,10,10] tensor), but must Cast to INT32 before any arithmetic. Applied in task368 v5: gcol_u8, gcol_flat_u8, stamp_u8, ocol_u8, tmpl_u8 all stay UINT8 (100B each). The ReduceMax+ArgMax for template origin uses UINT8 ReduceMax → INT64 ArgMax cleanly.

## 2026-06-03 — Xor trick eliminates Not intermediates in BOOL mask chains (task368)

When computing "exact row rank" masks rm0..rm3 via consecutive-AND (above1=And(is_ph,cd1), above2=And(above1,cd2), etc.), the standard approach needs Not(cd1), Not(cd2), Not(cd3) as 100B intermediates each. Replace with Xor: rm0=Xor(is_ph, above1) = 1 exactly when is_ph=1 AND cd1=0 (i.e., first row of a run). Similarly rm1=Xor(above1,above2), rm2=Xor(above2,above3). Eliminates 3 Not tensors (300B) from each row+col axis. Also: when using a 3-step cwa with T_ty_0 as the default (cm0 not needed as a mask), the cm0 mask and its Not(cr1) prerequisite are also eliminated. Total savings: ~700B for this task.

## 2026-06-03 — Sentinel-Pad before channel expansion eliminates 10-channel intermediate (task368)

Pattern for "ocol_u8[1,1,10,10] → one-hot [1,10,30,30]": instead of Equal(ocol_u8, ci_u8)→out10b[1,10,10,10] BOOL (1000B) then Pad(out10b)→output[1,10,30,30] BOOL, use Pad(ocol_u8, pads, sentinel=10)→ocol_padded[1,1,30,30] UINT8 (900B) then Equal(ocol_padded, ci_u8)→output[1,10,30,30] BOOL (graph output, free). The sentinel value 10 (out of [0..9] range) produces all-False at padding positions, matching the expected all-zero encoding. Net: eliminates out10b (1000B), adds ocol_padded (900B). Saves 100B. The key: graph output tensors are NOT counted by the ORT profiler.

## 2026-06-03 — Asymmetric Conv kernel crops rows for free: [1,10,21,1] → output [1,1,10,30] (task368)

When a Conv 1x1 color-decode (weighted channel sum, weights=[0..9]) reads a [1,10,30,30] input whose active grid is only 10 rows, make the kernel [1,10,21,1] with only kr=0 nonzero: output rows = 30−21+1 = 10, so the conv output is [1,1,10,30] F32 (1200B) instead of [1,1,30,30] (3600B) — the row-crop is free. Kernel zeros COUNT as params (10·21·1 = 210), so cropping the columns the same way explodes: [1,10,21,21] = 4,410 params. The optimum is asymmetric kernel + Cast-to-U8 at [1,1,10,30] (300B) + col-Slice to U8 [1,1,10,10] (100B). Cast-before-Slice beats Slice-before-Cast (400B vs 500B). Combined with replacing Slice+ArgMax(INT64) entirely: the Conv weighted-sum gives exact color indices on one-hot inputs (exact fp32), eliminating both inp10 [1,10,10,10] (4000B) and gcol_i64 (800B).

## 2026-06-03 — Cumulative ≥k masks replace exact-rank masks when Where chains nest deepest-rank-outermost (task368)

To select per-cell among rank-indexed values (T[0..3] by row_rank), you do NOT need exact-rank masks (rank==k). Nest Where with the deepest rank OUTERMOST: `Where(ge3, T3, Where(ge2, T2, Where(ge1, T1, T0)))` — a rank-2 cell fails ge3, passes ge2 → T2. The cumulative masks (above_k = consecutive-AND of shifted is_ph) come straight out of the shift chain, eliminating the Xor-derived exact-rank masks (rm/cm) entirely, and the chain's innermost default can be the passthrough canvas (gcol) which also eliminates the separate final merge Where.

## 2026-06-02 worker-7 — task383 pure-compression pass (25,767 → 18,666, hidden-confirmed rule)

v1 (25,767) held hidden at 6455.99 on Kaggle → rule correct; reopened as pure compression. Drove to 18,666 (−66% vs the 55,467 byte-equal floor) at 100% truth. Reusable compression takeaways (general, not task-specific):

1. **Replace neighborhood detectors with bbox-corner Gathers when a property is positional.** Frame-vs-fill ("which color touches bg") was a 6-mask MaxPool dilation chain. But on a bordered rectangle the frame color is ALWAYS `cidx[first-nonbg-row, first-nonbg-col]` and the fill is ALWAYS `cidx[bbox-center]` — verified 0/200k. Get the indices via `ArgMax`(+`select_last_index=1`) on `Greater(ReduceMax(cidx_u, axis), 0)` and `Gather` the scalars. Tiny tensors replace 5×576-byte masks. **Bonus: it's shift-invariant** (follows the active bbox), so adversarial ±1 shift went 1011/1064 → 1064/1064 — a from-scratch rewrite can be MORE robust than the LIVE graph it matches.
2. **Once you have a scalar color, `Equal(cidx, B_sc)` IS the mask** — drops the whole "nonbg AND NOT A" (Equal+Xor) chain. Derive masks from the cheapest scalar you already extracted.
3. **Collapse rank-1 masks to vectors.** A cross-out mask is `cleared_rows[H] OR cleared_cols[W]` (an outer product). Compute/branch/combine on the tiny [H]/[W] vectors and materialize the 24×24 plane with ONE final `Or`. Don't build parallel full-plane marker/fallback/combine masks (that wasted ~2,880 B here).
4. **OneHot is the cheapest scalar→one-hot output; proven by building the alternative.** "Output is free, expand late" already describes OneHot (single-channel INT32 index 3,600 → free 10-channel output). The one-hot-free `Equal(canvas, channel_ids)` expansion was built+gated at 24,977 (WORSE): any 10-channel intermediate is ≥9,000 B. Don't replace OneHot to "save the index."
5. **The real floor of a global-rule-on-24×24-canvas is ~18k, and it's the I/O dtype kernels, not op count.** F32 `Conv` color-decode (3,600) + INT32 `OneHot` index (3,600) + F16 `ReduceSum` count source (1,152) = 8,352 (45%) are forced by ORT (Conv float-only, OneHot int32-only, ReduceSum no-sub-F16). MACs are free, so collapsing ops doesn't help; only shrinking/eliminating *intermediate tensors* does. Name the forced tensors + why each is forced; that's the floor proof.

## 2026-06-02 worker-7 — task383 output rewrite (18,666 → 13,112): canvas+Equal-output + MatMul-cleared

After compression stalled, two STRUCTURAL output-path rewrites (from cost_floor_reference §238) cut another 30%:

1. **Make `Equal(canvas, channel_ids)` the graph OUTPUT, not an intermediate** — the single-channel-canvas pattern. To emit a [1,10,30,30] one-hot output, build a [1,1,30,30] U8 color-index canvas and feed `Equal(canvas, ch_ids[1,10,1,1])`; the BOOL result IS `output` (outputs aren't counted), so the 10-channel expansion is FREE. This BEATS `OneHot` (which forces a 3,600-byte INT32 index). **The trap (cost me a wasted variant):** do NOT `Cast` the Equal to F32 afterward — that makes the 10-channel Equal a counted 9,000-byte intermediate (my v4_ohfree scored 24,977, WORSE than OneHot). Declare `output` dtype BOOL; the validator decodes `result>0`, gates pass (visible 266/266, truth 200k/200k). Padding cells carry an out-of-range sentinel (10) so `Equal` gives all-zero there.
2. **Reductions over a mask → MatMul with tiny output, reusing the F16 mask.** The marker cross `cleared_col[c]=Σ_r anom_row[r]·is_B[r,c]` was 2 `Where`+`ReduceMax` over 24×24 U8 planes (3×576 B). As `MatMul(anom_row[1,1,1,24], is_B_h[1,1,24,24])` it's a tiny [1,1,1,24] output reusing the F16 B-mask that already exists for counting — no new 24×24 plane. F16 MatMul is supported; outputs are MAC-free in cost.
3. **Floor honesty:** these got task383 to 13,112 (−76% vs LIVE 55,467; beats the cost-ref's old ~25k estimate ~2×). **<3k is unreachable**: the per-cell color-decode Conv (3,600 F32, ORT float-only) + per-cell B-count (1,152 F16, ReduceSum no sub-F16) are forced and sum to 4,752. A rule that reads per-cell colors AND counts per row/col can't drop below its color-index plane + one count plane + the output canvas. Don't promise low-k for per-cell-color + global-count rules on a ≥24×24 canvas.

## 2026-06-01 captain session — +29.39 LB (6401.05 → 6430.44), 15 ships (2 rounds)

Pointer entry; full cross-task detail in `The_400_Tasks/current_strategy.md §captain-session-2026-06-01` and the Onnx_Advanced files below.

1. **Cheapest-staged-probe SOLVED ⇒ the floor is BELOW your spread — probe cheaper, do NOT retire.** task286: reach2sym **m20 (63,211)**, the cheapest staged, solved → overrode the worker's "63k floor / retire". (Mirror of round-1 where bp50-fail / bp55-solve bracketed D∗.)
2. **A worker's declared floor is usually beatable — verify cheaply:** check actual grid dims (`corpus.build(N)['in_shapes']`; cropping only helps if grids < 30) + whether the cheapest spread probe solved. Real full-30×30-plane floors this session: task203 `mid_oh` 1,200, task400 `canvas` 2,304, task224 `border` 900, task137 2×900. Beaten "floors": task203 2,404→2,188 (fold + ScatterElements), task215 5,730→110 (rewrite). Taxonomy in `Onnx_Advanced/cost_floor_reference.md`.
3. **Path-B / op-swap ship safety = FULL-CORPUS correctness, not op-vocabulary.** task224 Einsum **40×** +3.69 (byte-equal 2266/2266), task377 from-scratch +1.86 (**0 real bugs vs LIVE / 200k**), task203 v6 opset-18 ScatterElements +0.10. Contrast task324 Einsum (failed: never byte-equal). `Onnx_Advanced/traps.md` SHIP DECISION REFERENCE.
4. **Grader runs opset-18** (`ScatterElements(reduction='max')`, task203 v6). New reusable trick: priority-scatter inverse permutation (`Onnx_Advanced/techniques.md`). Ships logged in `Logs/submissions_ledger.csv`.
5. **Round 2 (8 byte-equal ships, +16.99): sub-max crops are DEAD when grids reach the canvas max.** task340 17×12, task379 h16×16 + novert all FAILED on hidden (grids reach 20×20); the clean full-size byte-equal candidate won every time (task335 45×, task378 20×, task340/379/222/192/112/169 all byte-equal). **Verify `corpus.build(N)['in_shapes'].max(0)` before any crop**; "truth 2000/2000" does NOT prove a crop safe (task340's crop passed truth, failed hidden). task286 closed: m10–m19 ladder all failed → **D∗=20, 63,211 is the proven floor**. See `Onnx_Advanced/traps.md §CROP-CONTRACT` + `current_strategy.md §captain-session-2026-06-01` #5/#6.

---

## 2026-06-01 captain session (cont.) — task216/157/255/182 + "filtered-clean is task-specific"

Continues the same-day session above. Running high → ~6434.4 (see `Logs/submissions_ledger.csv`; doc-hygiene = don't hardcode the LB). Cross-task detail in `current_strategy.md §captain-session-2026-06-01` #8–#10.

1. **task216** — v7 (13,995, byte-equal) is the cheapest-CORRECT floor, +1.97. kiter1 (9,023, 35.6% truth) scored literal **0** (incorrect net gets no `max(1,)` floor) — sub-floor approximations are futile. Rule needs 4 sequential passes; no correct model < ~14k.
2. **task157** — combo-free greedy structural rewrite 227,123→37,310, +1.80 (Path-B, truth 782/782 + hard seeds).
3. **task255 (ambiguous fill)** — held-out gate OVER-rejects: v52 (17,604, worker-gate "91.8% → +0.00 trap") PASSED hidden and BEAT the worker's "18,145 definitive floor" (+0.03). Submit-first found the win the gate hid. Also **shipped ≠ live**: v1 was "shipped" but the live zip still had 19,052.
4. **task182 (memorized lookup, 267 exemplars)** — gate UNDER-rejects: v53 (4,683, truth 20k/20k, visible 264/267) scored **0**; for a lookup task every visible pair is in hidden and ARC-GEN truth ≠ the full official set. Shipped safe v48 (13,881 byte-equal LIVE, +0.145).
5. **Meta** — "filtered-clean hidden" is TASK-SPECIFIC (ambiguous ⇒ clean-relax; deterministic ⇒ 100% visible). Submit-first (keeps-highest) is the only reliable judge, both ways. Gate byte-equal vs LIVE, not just vs the generator. New technique: one-hot MatMul → ArgMax+Gather (byte-equal table lookup). See `traps.md` SHIP DECISION REFERENCE §FILTERED-CLEAN-vs-LOOKUP + `techniques.md`.


---

## 2026-06-01 worker-7 — task191 Path-B rewrite (63,060 → 32,841, +0.65 LB) + spatial-matcher floor proof

DriveToFloor prompt wanted <1k. **Proved <1k structurally impossible** (rigorous floor proof in `task191.md`), drove to the real floor instead. Reusable takeaways:

1. **"Distilled MLP / lookup" ≠ unsolvable-by-rewrite.** task191's LIVE graph is an MLP-style stamp bank (46 MatMul), but the GENERATOR's rule (`task_7df24a62`) is a clean variable-template D4 matcher. Decoded it from the generator source (not the LIVE graph), built a from-scratch minimal ONNX. Don't assume a trained-weight LIVE graph means the task is memorized — read the generator.
2. **NEW reusable technique — runtime variable-template extraction → Conv with runtime weights** (added to `techniques.md`): (a) bbox of a marker via `ReduceMax`(presence) + `ArgMax` asc (`select_last_index=0`) / desc (`=1`) → first/last index; (b) extract a variable patch into a FIXED-shape tensor via **`Gather` with runtime index VALUES** (Gather keeps static output shape even when indices are data-dependent — unlike `Slice`, whose dynamic starts/ends make the output shape dynamic → `score_network_v2`=None); clamp indices with `Min` so degenerate inputs don't crash the Gather; (c) build N transformed kernels via **build-time permutation tables selected by a runtime combo index** (`Gather(PERM[combos,N,K], combo)`), then `Gather(values, perm)` — handles variable kernel geometry with tiny tensors; (d) **ORT `Conv`/`ConvTranspose` accept RUNTIME-computed weights AND bias** (not just initializers).
3. **Floor proof for spatial matchers.** An N-orientation match map at full-grid F16 is irreducible: orientation-resolved correlation is **not linearly separable** (rotations don't combine into one kernel), ORT forces float Conv (no int/bool), and a variable match-window defeats any fixed-window LUT/encoding. Here: 8 D4 orientations REQUIRED (dropping transpose changes **2% of outputs**, measured 41/2000), 23×23 = full grid → `[1,8,23,23]` F16 = 8,464 B is the hard floor. Name the dominating tensor + why each axis (channels/spatial/dtype) is forced + why no reformulation removes it.
4. **Cost is intermediate-tensor memory + params, NOT op count** (MACs free since 2026-05-04). The original prompt's "collapse the 46-MatMul Einsum bank → sub-30k" was a non-starter twice over. Big wins came from: crop to the always-23×23 active grid, exploit always-zero channels ({0,1,4} only → ch2,3,5-9 are shared zero-INITS, free in memory), produce BOOL `output` directly, and the 8→2-channel paint-seed group-reduce.
5. **A correctness subtlety worth instrumenting:** the generator records only the LAST matching orientation per anchor; my Conv/ConvTranspose paints the union. Verified equivalent by instrumenting **0/20000 grids have an anchor matching >1 orientation with a differing rect**. Gate: `ng_gate truth 20000/20000` + visible 267/267.

---

## 2026-06-01 worker-7 — task383 Path-B rewrite (55,467 → 25,767, 100% truth) + 24×24-canvas floor proof

DriveToFloor prompt wanted <1k. **Proved <1k structurally impossible** (floor proof in `task383.md`); drove to the real floor at 25,767 (−53.5% vs the 55,467 byte-equal MaxPool ship, −83% vs original LIVE 152,592). Reusable takeaways:

1. **Decode the rule from the GENERATOR corpus, not just the 8–10 visible pairs — then verify against ALL 200k.** task383's old `v2` matched 266/266 visible but was hidden-bad (−12.31 LB). My first decode (sparse-line cross, threshold `count*3<max*2`) also hit 266/266 visible AND 262/262 stored arc-gen AND 99.884% on the 200k generator — but that last 0.116% (232 cases) is what separates a safe ship from a repeat of the v2 disaster. **Always run `ng_gate truth --full` (200k) and chase 100%, not just the 99% bar**, by loading the cached corpus `Q:\…\argen_cache\taskNNN.npz` directly and diffing your model vs GT to surface the failing patterns.
2. **A "good enough" threshold can hide a second rule mode.** The 232 failures were a *distinct puzzle sub-type* (single 1-cell **corner-notch** in the fill rectangle) the sparse-line rule can't see (no line is sparse enough; raising the threshold over-triggers — 0.8 → 96.9%). Real rule = **two modes**: sparse markers → cross perpendicular; else corner-notch → cross the FULL lines (count==max) along the **shorter bbox dimension**. Gate it with `has_marker = any sparse line` and `Where(has_marker, marker_cross, notch_cross)`. Combined → 200000/200000.
3. **LIVE-equivalence (visible 266/266) ≠ corpus-correct.** A from-scratch rewrite can match LIVE on all real inputs (visible + 100% truth) yet diverge on artificial adversarial perturbations (here ±1 `shift` 1011/1064 — the 53 are down/right shifts that move content off the top-left origin, which the bordered-grid generator never produces). For Path-B, **truth-vs-GT is the authority**, adversarial-vs-LIVE is a guardrail that legitimately diverges OOD.
4. **The padding/active-region trap (cost visible 0/266 if missed).** A cropped 24×24 canvas paints cross-out rays across the whole crop, but LIVE/GT leave padding (outside the real H×W) **all-zero**. Fix cheaply: active = `row_has AND col_has` from per-axis `ReduceMax(input, axes=[1,3])`/`[1,2]` (tiny [1,1,30,1] outputs, NOT a full 3,600-byte channel reduce), then mask padding to the **OneHot out-of-range sentinel** (index==depth → all-zero channels) so one op zeroes both the within-crop and 24→30 border padding.
5. **Floor proof for a 24×24-canvas multi-stage rule (~25k).** Two forced I/O tensors dominate (28%): `cidx_full` [1,1,30,30] F32 = 3,600 (ORT `Conv` float-only color-decode) + `out30` [1,30,30] INT32 = 3,600 (ORT `OneHot` needs int32 index). Plus ~22 single-byte 24×24 masks (576 each) the global per-row/col-count rule needs. Contrast task181/335 sub-1k: their active grids are ~6×9 (10× fewer cells/mask). **Rule of thumb: sub-1k needs a tiny active grid OR a near-zero-op shape transform; a count/geometry rule on a 24×24 canvas floors ~10–25k.**
6. **ORT dtype gotchas that shaped the build** (all in `Onnx_Advanced/ort_reality.md`): `ReduceSum` no UINT8 → counts in F16 (1,152 vs 2,304 int32); `ReduceMax` OK on UINT8 but NOT BOOL; `Mul`/`Sub` no UINT8 < opset14; `Where` data branches can't be BOOL (cast to U8); `Pad` no BOOL at opset 11; `OneHot` indices+depth must share dtype (INT32 indices work with INT32 depth — not only INT64) and **out-of-range index → all-zero output** (the padding-sentinel lever).

## 2026-06-01 — task238: Object_Based frame+scatter rule decoded; single-channel color-index map reduces cost 73%

Task238's rule: input has a rectangular frame (4 border colors T/B/L/R, square interior IH∈{3,4,5}) + a separate scatter of color-8 cells (bounding box = IH×IH = frame interior size). Output: frame border always colored; each active scatter cell colored by nearest border (tie zones keep color 8). Key lessons: (1) IH must be computed from actual per-channel cell counts (ReduceSum over spatial dims), NOT from row projections — projections give unique-colors-per-row (1 for top/bot border rows, 2 for side rows) which corrupts IH detection. (2) ArgMax on row projection must be applied after binarization (Greater>0.5) to get first non-zero row, not first max-value row. (3) The scatter bounding box is spatially SEPARATE from the frame; it must be located independently via ch8 projections. (4) Scatter patch needs a (+1,+1) shift (Pad+Slice) before region-mask multiplication since scatter cell (ir,ic) maps to output position (ir+1,ic+1). (5) The LIVE ONNX never outputs ch0=1 at bg cells without a separate mechanism; `run_raw` binarizes via >0.0, so bg cells need ch0=1 — handle with a sentinel value (10) at padding positions that makes Equal(10, 0..9)=False. (6) The big architectural win: single-channel [1,1,7,7] F16 color-index map (border colors × region masks + scatter colors × region masks) reduces the 13× [1,10,7,7]=1960-byte intermediates to 28× [1,1,7,7]=98-byte F16 intermediates = saves ~24,000 bytes memory. Final: 55,368 → 15,085 bytes (-72.8%), predicted +1.30 LB.

---

## 2026-06-01 — task018: drive-to-floor 62,027 → 58,336; empirical proof <30k impossible

DriveToFloor prompt wanted <1k; user pushed for <30k. **Drove the shipped 62,027 (compress_v5) to 58,336 (compress_v7, −3,691, 266/266 vs LIVE), then proved <30k is structurally impossible** for the flood+D4+GridSample rule (full proof in `task018.md`). Reusable takeaways:

1. **NEW reusable technique — replace per-row/per-template min/max-over-a-full-HxW-plane with a SHARED `ScatterElements(reduction=min|max)` per-category.** task018 found each template's reference-anchor position via `Where(mask, pos+1, sentinel)` → `ReduceMin/Max` over a `[1,1,24,24]` F16 plane (2 planes/template × 2 templates = 4,608 B, + per-template `anchor_ref_bool` 1,152). Replaced with ONE shared scatter: `ScatterElements([C], color_index_i32, pos_flat_p1_f16, reduction='min'/'max')` → `[C]`-vector of per-color extremes (computed once), then each consumer `Gather`s its category's value (~free). Net −3,403. **`ScatterElements(min, F16)` IS ORT-supported at opset 16** (probed; `max` already known). General rule: when you need per-category extremes of a positional quantity, scatter-by-category beats masking-then-reducing-a-plane once you have ≥2 consumers — turns N full planes into 1 shared `[C]` vector. The scatter UPDATE can be the 0-memory position INIT (non-category cells land in an unread bucket via the category index, so no masking needed for `max`; `min` needs a large data-init).

2. **Empirically confirmed: a data-dependent `Slice` (crop to runtime bbox) → `score_network_v2`=None.** Built a minimal probe (ArgMax→Slice start) and it scored None — the scorer rejects the symbolic output shape. This is THE blocker for cropping a scattered-object task to a smaller working canvas. Combined with the crop-contract (ARC-GEN grids reach the canvas max, so no fixed sub-max crop is byte-safe), the working resolution is locked at the generator's max grid size. For task018 that's 24×24, and ~30 distinct 24×24 planes (CC-flood, D4-detect, 2× GridSample warp, mask pipeline, output assembly) are each semantically required → a hard ~52-58k floor. **Lesson: before promising a 2-3× cut on a full-canvas multi-stage rule, count the irreducible HxW planes and confirm the crop path with the dynamic-Slice→None probe — dtype/op-fold shaving gets ~10%, not 2×.**

3. **GridSample is the optimal warp and its cluster barely shares across the 2 sprites.** Combined single-GridSample (one grid for both creatures) measured only ~−576: merging data/output (−4,608) is eaten by the per-cell region-select Where + the second grid still needing full construction (`base_rc_t` Gather + `grid_f16` Add = 2 `[1,24,24,2]` F16 each). Warping the colored creature (body+anchors in one warp, dropping the separate anchor-label path) saves only ~−1,152 with real F16-color-precision risk. Neither changes the floor materially.

4. **Don't eliminate a min/max pair as "redundant" without the 2-creature shared-anchor case.** compress_v7-pre attempted to drop the max-anchor check (single-pixel anchors guaranteed by the generator) — broke 124/266. When 2 creatures share anchor colors AND sit within the dilation radius, the reference color appears at 2 clone-anchor positions; `min` picks the wrong creature's, `max` the right one. The dual check is a correctness requirement, not redundancy.


## 2026-06-06 worker — task166 DriveToFloor: 21,648 → 2,441 byte-equal (−88.7%), floor-proven

Target <10k beaten 4×; full gates green (visible 266/266, truth 200k 100%, ab 200k byte_equal=True, DETERMINISTIC class). Staged `task166_v3_dualplane_cost2441`. Reusable takeaways:

1. **Where-as-final-node is the endpoint of techniques #4b — price it FIRST on any "recolor a data-dependent region" task.** `Where(cond [1,1,30,30] BOOL = 900 B, fill_onehot [10,1,1] broadcast, input)` as the LAST node assembles the full output for 900 B because the output tensor is memory-free and `input` is free. LIVE paid 19,600 B of 10-channel crop/delta/sum planes to do the same recolor by Add-paint — a 10× architecture tax. Assembly-floor lemma worth reusing: every alternative final op (Pad-of-inner ≥1,960 BOOL + the F32 crop to build it; Concat ≥3,600/channel; elementwise ≥9,000 full-size operand; Conv = shift-invariant, can't) loses to Where by ≥2×.
2. **Dual-use plane: one §41-negpad 1×1 Conv plane (`[1,1,14,14]` F32, 784 B, 10p) can serve row-reduce + col-reduce + per-cell compare simultaneously** — check for a shared-plane factoring before paying separate extraction paths (v2 paid +278 for a bg Slice that a `Less(plane, 0.5)` replaces). Negpad with PURELY negative end-pads + kernel 1×1 passes checker/strict-inference/ORT/scorer (new shape variant of the grader-confirmed §41 family).
3. **In-bbox ⇒ in-active-grid lemma kills the 1-ch0 guard**: the row/col cross-product of nonbg occupancy lies inside the active grid, so `bg ⇔ ¬nonbg` there — no active-cell mask, no ch0 extraction, immune to the padding trap *within that region*.
4. **"Bbox fill" tasks may actually be CROSS-PRODUCT fill** (occupied-rows × occupied-cols, what `Mul(row_has, col_has)` computes). Identical for one connected shape (row/col sets of a connected shape are intervals), divergent for disconnected content. Match LIVE's semantics and verify on 200k generator truth instead of trusting the visual name.
5. **F32 first-touch extraction tax: any per-cell signal derived from the F32 input costs ≥ H·W·4 at first materialization** (Conv/Slice keep F32; comparisons applied directly to `input` cost a full [1,10,30,30] BOOL = 9,000 B). Budget 784 B for a 14×14 region and stop hunting below it — no ONNX op fuses select+compare+downcast in one node.


## 2026-06-06 worker — task055 DriveToFloor: 31,628 -> 2,386 byte-equal (-92.5%), staircase-code assembly

Target <40% beaten 5x; full gates green (visible 263/263, truth 200k 100%, ab 200k byte_equal=True, DETERMINISTIC class). Staged `task055_staircase_v7_u8_cost2386` (primary) + `task055_staircase_v1_cost3801` (independently-certified F16/Sub conservative rung). New library entries: techniques #58, by_symptom "row-band x col-band" route, 2 ort_reality matrix corrections. Reusable takeaways:

1. **NEW technique #58 — staircase-code rank-1 assembly.** When every output channel is a row-band x col-band product (+ optional line-cross union), stack per-channel U8 row codes `X [1,10,30,1]` / col thresholds `Y [10,1,30]` and make `Greater(X,Y)` the graph output: zero planes materialize, the whole graph is 30-long vectors. The 3-level staircase (x=act+line vs y=!act+!line) expresses a UNION of two outer products in the same single op — something `Where(condR, A_c, B_c)` provably cannot (3 distinct row-patterns vs 2). Decoder `>0` = the comparison.
2. **CumSum F16 "no kernel" was an OPSET artifact, not a kernel gap** (matrix corrected): opset <=13 SCHEMA rejects F16 (added at 14); at opset >=14 the ORT CPU kernel exists and runs (LIVE task055 was already running it in production — READ THE LIVE GRAPH for op-support ground truth before trusting matrix rows). 1-D prefix chains CAN be 2 B/elem.
3. **Min(u8,u8) CPU kernel confirmed** (matrix row added): = AND on 0/1 masks in one op, replaces And(bool)+Cast(U8) (-30 B per use).
4. **Op-support probes are free inside the gate run** — v3 "probed" Cast(F32->BOOL), Greater(u8,u8), BOOL graph output simply by running visible; no separate probe harness needed.
5. **Unused initializers still count as params** (`calculate_params` sums ALL graph.initializer) — prune dead inits when folding constants away.
6. **traps S12 (bool reformulation hidden-bad) has a clean admissibility test**: it bans bool rewrites of graphs whose floats are NOT provably 0/1. Counts/prefix-sums/codes ARE provably exact small integers, and task013 shipped a BOOL-output And — BOOL output + integer staircase passed 200k byte-equal here.
7. **Adversarial split is diagnostic for structural-reconstruction candidates**: structure-PRESERVING modes (rotate/flip/shift/shrink) at 100% with structure-DESTROYING modes (stray cell, wiped quadrant) at ~0% is exactly the signature of "reconstructs from generator invariants instead of passing input through" — fine for byte-equal class, document the invariant in the manifest.

### 2026-06-06 addendum — task166 round 2 (owner push for <1,500): 2,441 → 2,057 via generator-source window; sub-1,500 proven impossible

6. **Read the ARC-GEN generator SOURCE before betting on corpus-measured bounds — it upgrades distribution bets to certainties.** `external/ARC-GEN/tasks/task_<uuid>.py` (uuid = `sorted(training/*.json)[N-1]`, files under `external/ARC-GEN/external/ARC-AGI/data/training`) gave task166 hard structural bounds: box placement margins ≥1 on all sides, width ≤12/height ≤14 ⇒ fill region ⊆ [1..12]² in every flip/transpose orientation. That turned a "200k pairs never show content at row 13" observation into a source-certified invariant (and explained it: tight randint bounds). Where-assembly makes this usable: the window crop only bounds *paintable* cells — passthrough is untouched — so historical canvas-shrink failures (which crop passthrough) don't apply.
7. **The honest-static floor for any full-canvas region-recolor is ~1,900–2,100, and the binding constraints are now grader-source-verified**: `calculate_memory` counts every named intermediate at `max(static, runtime-observed)` (only literal `input`/`output` skipped), `calculate_params` counts initializer ELEMENTS dtype-blind. Two irreducible terms: Where-cond [1,1,30,30] BOOL = 900 (unique sub-1,960 assembly — exhaustive final-op sweep: Pad-inner/Concat/OneHot/elementwise/MatMul/Scatter all ≥1,440+prereqs) + per-cell F32 first-touch 4·B² (no ONNX op fuses select+compare+downcast). Sub-1,500 asks therefore route ONLY through the owner-banned accounting exploits (scoring_internals A.2 "ethical play only") — decline and document rather than build.
8. **Real-submit probes have no information value for static graphs** — §39 mechanism (source-read): static ⇒ local == Kaggle exactly, so a probe can't reveal a lower number; dynamic ⇒ repriced UP toward runtime truth. The probe-worthy frontier is only constructions with unverified GRADER semantics (3-operand Einsum, CenterCropPad, both-operand-repeat Einsum) — none of which price below the static floor here anyway.

**SAME-DAY UPDATE (owner-directed push under 1,500)**: compare-arch floor 2,386 BROKEN by a second architecture — bitand bit-detectors (task055 v8-v11, best **1,174 = 3.7% of LIVE**, vis+truth-5k green, fulls running; v9 1,354 FULLY byte-equal-certified + staged). Additional reusable takeaways:

8. **NEW arch pattern — bitand bit-detector output (techniques #58 extension)**: give each (row-set x col-set) product detector ONE BIT of a U8; per-axis code vectors x,y; channel k fires iff `x(r)&y(c)&K[k] != 0`. The reassociation `output = BitwiseAnd(BitwiseAnd(x, K[10,1,1]), y)` makes channel expansion cost 300 B (one [1,10,30,1] u8) instead of the compare-arch's 600 B double-stack — and unions (ch8 line-cross) are free (K masks 2 bits). 25-combo table verified exhaustively pre-build.
9. **Nested Where(bool, u8-scalar, u8-scalar) is the cheapest k-way code selector** (30 B/select + 1 param/constant; selects forced = values-1). The affine alternative (code = A+B*pre) was searched over all 8! bit assignments incl. mod-256 wrap: zero solutions — popcount imbalance makes progressions impossible; don't re-search.
10. **ArgMax replaces CumSum for "first/last marker position" banding**: r1 = ArgMax(counts), r2 = ArgMax(select_last_index=1) — [1,1] I64 = 8 B each — then bands = Less(iota_init, r) compares. Kills the entire F16 cast+CumSum block (-180 B vs prefix route incl. comparisons). Works whenever the marker rows hold the strict max of a free signal (divider count W>=5 > crossing 2). ArgMax(F32 AND u8, select_last_index) execution-verified at opset 18.
11. **Opset-18 fear is dead for the grader**: task191 v6 shipped OPSET 20 Kaggle-confirmed (techniques #57 note) — Bitwise* (18) is within grader ORT coverage; the staged bitand rungs will confirm credit-side.
12. **Iota-init (30p) beats Range** (Range output is a counted intermediate: [30] I64 = 240 B vs 30 params) — and params count ELEMENTS, so iota dtype is cost-free; use I64 to compare against ArgMax outputs cast-free.

### 2026-06-06 addendum 2 — task166 round 3 (anti-quit ping paid off): 2,057 → 1,900 via Less-broadcast fusion; floor claims are soft until FUSION-AWARE

9. **NEW reusable technique — Less-broadcast fusion: scale a detector Conv so an axis-reduce of the SAME plane doubles as a broadcast comparison threshold.** task166: Conv weights 12.0 ⇒ plane values {0,12} ⇒ `col_has = ReduceMax(plane, axis=2)` ∈ {0,12} ⇒ `Less(plane, col_has)` computes (column-occupied ∧ ¬detector) in ONE op — false at detector cells (12<12, a detector cell's own column is always occupied), true at empty cells of occupied columns, false in empty columns. Kills a boolify plane + a cast + the threshold scalar (−157, −1p). Generalizes to any "axis-occupancy ∧ ¬per-cell" condition; fold ONE axis this way and And the other as a 12-byte BOOL vector (folding both needs a 2-D threshold plane — net worse).
10. **Stage-decomposition floor proofs are SOFT — only a fusion-aware bound (sweep every op PAIR for a fusing op) deserves the word "floor".** task166's floor claim fell twice in one day: 2,441 → 2,057 (external knowledge: generator-source window bounds) → 1,900 (cross-stage fusion: the boolify stage merged into the cross-product logic via broadcast Less). Write floor proofs with the epistemic note attached; an owner "it can be done cheaper" prod is empirically a good reason to re-sweep with fresh eyes.

### 2026-06-06 addendum 3 — task166 round 4: when does MILD-DYNAMIC (§39 discriminator) pay? A transfer-pricing rule

11. **Mild-dynamic crops (data-dependent shapes that stay small, 1.1-1.4x credited markup — the 398-v16/014/065 class) pay ONLY when the static alternative carries big canvases.** Price the transfer BEFORE building: dynamic costs a fixed extent-extraction tax (per axis: a full-length marginal vector + Sign + 2 ArgMax + int64 plumbing ~= 350-400 B runtime-observed) plus it CANNOT shrink any tensor the assembly forces at full canvas (task166: the Where-cond 900 B survives every dynamic variant). task398 won because its static rung held 625-3,600 B-class planes; task166 static window machinery is only ~590 B ⇒ dynamic nets ~+150 WORSE. Rule: sum the static tensors the dynamic crop would actually shrink; if < ~600 B, mild-dynamic loses before it starts.
12. **Equal-assembly (`Equal(cidx_u8_canvas, channel_ids)` as output, sentinel-10 padding — task398 v4) vs Where-assembly: decided by PASSTHROUGH.** Fully-generated outputs -> Equal-assembly (canvas 900 B u8, no input reuse needed). Outputs that pass input content through -> Where-assembly (input is a FREE operand; rebuilding it as a color map costs a 3,600 B F32 color-decode + active-mask). task166 is passthrough-heavy: Where wins 3x.
13. **Validation-tool trap found+fixed (traps #30)**: `adversarial_test.py` decoded with `np.isclose(vec, 1.0)` instead of the real `(out > 0.0)` — U8/F16 non-1.0-valued outputs scored 0/all despite 200k byte-equal. BOOL passed by luck. If an advisory gate contradicts the certified gates, READ THE TOOL before doubting the model. Fixed to `vec > 0.0`.
14. **OWNER-PUSH ROUND 2 (under-1000 directive) closed at 1,141 with a formal floor proof** (task055.md): final-stage lemma (binary broadcast to [1,10,30,30] needs >=330 B of operands; the (10 x spatial) operand is data-dependent hence computed), per-axis lemma (~376: einsum 120 forced 6 ways, 2 class compares, ArgMax+iota banding, 4-select tree forced 9 ways). Sub-1,000 is NOT expressible for this rule under this scorer. Two more probe-closures en route: Range outputs ARE charged (twin probe 181 vs 394 — zero_param_generators.md corrected), ScatterElements line-override breaks all-zero parity for -28 (declined). Meta-lesson: each "floor" fell to a new IDEA CLASS (architecture -> bookkeeping -> doc-claims); a floor claim is only as good as the alternative-class enumeration behind it — write the proof, not the number.
15. **Round-3 adversarial self-review: the 1,141 floor HELD.** Four new final-op families enumerated (Concat/Gather/OneHot/MatMul(Integer)-final) — MatMulInteger lands on EXACTLY the same 330 final-stage floor as BitwiseAnd (independent witness, weaker semantics). Position-weighted einsums (iota-weights) give r1+r2 in one value but bands need them separately. A floor that survives a hostile enumeration round is worth more than three that did not — the proof now lists every attack.

## 2026-06-06 worker-7 — task247 DriveToFloor: 18,184 → 543 byte-equal-FULL (−97.0%), first structural pass

Target <10k beaten 33×; full gates green on every rung (visible 269/269, truth 200k/200k, ab 200,269/200,269 divergences=0, DETERMINISTIC class). Ladder v2 2,401 → v3 606 → v4 563 → v5 543, all byte-equal; staged `task247_v5_eqassembly_cost543` (+3.51 LB raw-ln pred). Generator-source-certified (task_a3325580.py): inputs exactly 10×10, winners = the 1-3 max-boxes always all-tied, bg ≥ 67 ≫ M, winner boxes column-disjoint ⇒ leftmost ties impossible. Library note: `.update/DONE_2026-06-06_task247_dominance_topk.md`. Reusable takeaways:

1. **NEW technique — exponential positional dominance: first-occurrence ORDER as a free-input Einsum scalar.** "Order colors by leftmost column" = `f = Einsum('bchw,w->bc', input, 16^-j)` (40 B + 30p): `leftmost(c) < leftmost(d) ⇔ f(c) > f(d)` because the leading column exponentially dominates the tail. Margin condition: base b ≥ per-line-count + 2 (b=16 covers counts ≤ 14 with a ≥1.5× exact gap; weights 2^(-4j) stay F32-normal over 30 columns — b=32 would hit denormals/FTZ). Kills occupancy plane [1,10,1,30] 1,200 B + ArgMax + the whole pairwise-rank block. Generalizes to topmost (`'bchw,h->bc'`), rightmost (ascending weights).
2. **TopK as rank-selector beats pairwise-rank matrices whenever rank feeds a "top-K slots" one-hot**: `TopK(Where(ismax, f, 0), k=cap)` returns winners already sorted; `vals > 0` masks empty slots; `Equal(iota, idx)` is the channel one-hot. The pairwise route pays a 324-B BOOL→F32 Cast just to ReduceSum a 9×9 comparison matrix (ORT ReduceSum: float/double/int32/int64 only). Measured: 606 vs 1,115. Static-k TopK scores fine (task054 precedent re-confirmed through the full official path).
3. **Autopsy before architecture: 86% of LIVE was structurally dead** — a 4,000 B input crop whose every consumer is a tiny reduction (free-input stats make the crop pointless) and TWO identity Gathers (`indices=[0..9]` on axes 2,3 = 8,000 B) that every polish pass missed because no spread pattern checks Gather-index-is-identity (`gather_identity_elide` = candidate spread pattern). A prior worker note said "f16 NO WIN, spread 0 wins" — true, and it still hid a 30× structural win. "Polish found nothing" is evidence about polish, not about the floor.
4. **Deterministic-class + ab-FULL makes radical rewrites cheap to certify**: v3's only semantic divergence class vs LIVE (two winners sharing a leftmost column — dominance breaks the tie, ArgMax-rank duplicates the column) is provably unexercised in-distribution: 200,269 ab pairs, divergences=0. The rotate-adversarial 86% IS that class on off-contract inputs — advisory noise per core.md, documented in the manifest instead of feared.
5. **Einsum reduction beats ReduceSum at opset 13+ by 2 params**: the equation is an ATTRIBUTE (`'bchw->bc'` = 0p); ReduceSum-13 axes is an INPUT (init [2] = 2p). Same 40-B output off the free input. (ReduceMax keeps attribute axes until 18 — opset 13 is the sweet spot: attr-axes ReduceMax + input-form Pad(BOOL).)
6. **bg-dominance lemma — ch0 isolation can be FREE on bounded-M tasks** (the 606→563 cut): when the rule bounds the max object-color count (output-height cap M ≤ 9 here, provable from officials + LIVE's 9-row Pad holding hidden), `nonbg ≤ 9·M ⇒ bg ≥ 100−9·M > M` — bg is ALWAYS the strict global count max. So `M = TopK(counts10, k=2).values[:,1]` (bg owns slot 0; 28 B incl. the dead-but-counted indices output) replaces Slice+ReduceMax (40 B), and `Equal(c10, M)` needs no ch0 slice at all. Keep the one-hot iota at 1..9 as belt+braces. Generalizes to any "per-color stat where bg is structurally dominant" task.
7. **Unconsumed REQUIRED outputs (TopK indices) are still counted by `calculate_memory`** — shape inference places them in value_info. Budget dead outputs when comparing op alternatives (TopK-2 M-extraction wins over Slice+ReduceMax even carrying 16 dead bytes).
8. **u8 color-CODE canvas beats one-hot when channel expansion can ride the final pre-Pad op (the 563→543 cut)**: `Cast(idx→u8)` + 2 nested `Where` build a [1,1,9,3] code canvas (33 B) where the one-hot route paid Equal+And [1,9,1,3]×2 (54 B); ONE `Equal(canvas, chids 1..9)` expands to the [1,9,9,3] pre-Pad BOOL. Code 0 = "no color" matches no chid — valmask/row gating zero cells for free. Equal(u8,u8)@13 re-confirmed through the full scorer (already in `ort_reality.md:32` — and grep gotcha: the dtype matrix moved there; `ort_op_dtype_matrix.md` is a stub, search ort_reality.md FIRST).
9. **Floor-proof discipline echo (now 3 tasks in one day)**: my own fusion-aware hostile-enumerated "563 floor" fell within the hour to a tail-REPRESENTATION class (one-hot → u8 code) the enumeration had mispriced. Same meta-lesson as task055/166/256: a floor is a claim about the enumerated classes, not the task. Cross-checked the task256 broadcast-Add output class against 247: pays 30-extent marginals (≥660 F16) — only wins full-canvas content, not a 9×3 patch; Pad-final keeps the win.

## 2026-06-06 worker-7 — task256 DriveToFloor: 17,062 → 3,627 byte-equal-FULL (−78.7%), first structural pass

Target <10k beaten 2.8×; all gates green on both rungs (visible 266/266, truth 200k/200k, ab 200,266/200,266 divergences=0 byte_equal, DETERMINISTIC class). Ladder v2 3,938 → v3 3,627; staged `task256_v3_negmask_signarith_cost3627` (+1.55 LB raw-ln pred; v2 kept as rung). Library note: `.update/DONE_2026-06-06_task256_signarith_negmask.md`. Takeaways:

1. **Sign-correctness-only doctrine**: `run_network` decodes `(out > 0)` — output channels need the right SIGN, not {0,1}. Every Relu·2·Min clamp ladder in a float graph is deletable; raw affine fields suffice. task256 LIVE spent ~10,140 B (59% of memory) materializing a cropped input in two dtypes plus clamp planes the decoder re-binarizes anyway. Channel kit: threshold field `t = S−0.5−r−c`; bands as `Relu(affine(r))` (zero off-band = product-safe); "exactly-at" via the quadratic `0.25−k²` (one Mul of two existing vectors, no Abs); bg complement via ONE variadic `Min(−t, W−0.5−c, H−0.5−r)`.
2. **Negated-mask identity (the v2→v3 cut)**: when both a field and its negation are consumed, `t·Relu(x) ≡ (−t)·Min(x_flipped, 0)` — swap Sub operands, clamp at a shared zero scalar, no Neg ops → ONE plane serves all channels (−338 B for +26 B). Meta-confirmation of the task166 lesson "floor claims are soft until fusion-aware": my own v2 floor analysis fell within the hour to exactly this identity. Re-sweep before writing "floor".
3. **Grid extent as two scalars, not a presence plane**: `rowtot = Einsum('bchw->bh')` ⇒ `W = ReduceMax(rowtot)` (one-hot rows sum to W), `H = ReduceSum(rowtot)/W` (F32 Div; W≥1 structural, and F32 0-div → NaN not SIGFPE — the SIGFPE trap is integer Div only). In-grid clipping then rides existing Min/Mul ops as two 26-B rank-1 vectors, vs ~4.6 kB for reduce→crop→cast presence. Biggest single idea-class win of the rebuild.
4. **Generator-bound design contracts beat runtime guards**: 200k-corpus scans proved width≤W, S≤H−1, single 2-row, col-0 anchoring — so ch1/2/3 need NO grid clipping at all. Same confidence class as the 13×13 crop contract (LIVE itself crops). Verify the bound corpus-wide FIRST, then design as if it is an axiom, and write it in the manifest.
5. **Self-test encoder trap (cost one debug round)**: hand-rolled harnesses must zero ALL channels outside the active grid on BOTH input encoding AND expected one-hot — `np.eye(10)[idx]` alone marks padding ch0=1 and a flawless model reads 0/266. If a self-test contradicts a passing probe, suspect the harness before the model; `ng_gate` (official encoder) is the arbiter.

### 2026-06-06 worker-7 addendum — task256 rounds 2-4 (first anti-quit ping paid off AGAIN): 3,627 → 2,403 via a NEW ASSEMBLY CLASS; two floor proofs fell in one session

6. **NEW class — BROADCAST-ADD OUTPUT ASSEMBLY: make the FINAL op a broadcasting Add(A[1,10,30,1], B[1,10,1,30]) that writes `output` directly.** out[c,r,w] = A[c,r]+B[c,w] > 0; the [1,10,30,30] result is the free output, so NO 2-D plane exists anywhere in the graph. Any per-channel region of the form {w ⋛ f(r)} (half-planes, staircases, triangles) plus single-sided intervals (via a ramp cliff: B0 = Min(w+0.5, 64W−63w−32), one Min) assembles at A+B = 1,200 B — task256's plane formulation paid 3,042 B for the same content. Band gates ride pre-scaled aranges (64r+32; powers of two = F16-exact); "exactly-at-row" = Sum(above_off, below_off, +64) (variadic, 0 exactly on the row); dead channels/rows via Pad cv −64 on A, 0 on B. Margin discipline is the correctness load-bearer: prove off-region A + max(B) < 0 at actual F16 values (ghosts found at scale 32; 64 clears).
7. **Floor proofs are FORMULATION-RELATIVE (task055/166 lesson, now twice-confirmed in one session): enumerate FINAL-OP FAMILIES before pricing stages.** v2's plane floor (3,892) fell to a mask-polarity fold (v3 3,627); v3's "arithmetic-F16 floor" fell to the broadcast-Add class (2,403). Both proofs were right inside their formulation and blind to the formulation switch. The checklist that found the break: list every op family that can legally write `output` (Add/Mul/Min/Where/Pad/Resize/Expand/MatMul/Concat/OneHot/Scatter), price each family's minimum input set, THEN claim a floor — of the class, not the task.
8. **ORT CPU Sum(F16, 3 operands) works** (execution-verified through full gates) — variadic Sum is available for F16 gate arithmetic; LIVE-style Add chains can collapse.

### 2026-06-06 worker-7 addendum 2 — task256 round 5 (second anti-quit ping): 2,403 → 2,306; the enumerated class bound is REACHED (2,150 mem vs ~2,146 bound)

9. **COMBO HEAD — one dominance-weighted Einsum replaces two reductions**: `combo = Einsum('bchw,c->bh', 1 + 64·sel2)` ⇒ combo[r] = rowtot[r] + 64·count2[r]; R = ArgMax (object row = strict global max), W = ReduceMax−64L, HW = ReduceSum−64L, H = HW/W. One [1,30] F32 instead of two (−97 net). L itself is NOT recoverable from combo (the weighted scalar einsum ≡ ReduceSum(combo), circular) — the tiny sel2 scalar einsum is the irreducible companion. Keep α·maxcount F32-exact and α a power of two.
10. **Floor claims should come with the bound-gap stated**: v7 = 2,150 mem vs the itemized class bound 2,146 — 4 bytes of slack, written in the .md. When the gap is one scalar tensor, stop golfing and document; when it's hundreds, the formulation enumeration is probably incomplete (that gap-read is what motivated rounds 2-5: 17,062 → 3,892 → 3,580 → 2,528 → 2,370 → 2,150).
11. **Dominance-tricks need their preconditions checked against op-fusion alternatives**: task247's exponential-dominance R needs ch2-restricted row weights here = a 3-op Einsum (§40 pairwise caveat) or a [10,30] 300-param weight — while plain ArgMax on the combo marginal is 14 B and structural. Dominance wins when the marginal does NOT already exist; here it did.

### 2026-06-06 worker-7 addendum 3 — task256 owner-push round (sub-1500 directive): 2,306 → 1,592 via the u8-COMPARE class; risk-ladder staging

12. **NEW class below broadcast-Add — u8-COMPARE: `out = Less(C u8-codes, T u8-thresholds) → BOOL free output.`** A comparison final has identical rank-1 coupling power to Add-sign (a<b ⟺ a−b<0) but takes 1-byte operands and emits 1-byte output: the 600+600 F16 pads halve to 300+300. Bands share ONE static arange code init (0 bytes); only the interval-channel's code is computed (Max-form cliff `Max(base, 32(w−W)+26)`). Integer-native (strict < needs no half-offsets). task256: 2,306 → 1,679 (opset 12) → 1,592 (opset 19).
13. **U8-WRAP ASYMMETRY LEMMA**: Cast F16→u8 wraps mod 256 (probed: −3→253, 300→44). Wrapped negatives on the CODE side of a Less = huge codes = never-fire = SAFE; on the THRESHOLD side = fire-all = CATASTROPHIC. Hence Relu before every threshold cast — or kill the clamps entirely with **QuantizeLinear(·, scale=1) as a SATURATING F16→u8 cast** (probed: −30→0, 858→255; opset 19, QL-F16). Slope-32 cliffs make wrap residues provably safe on the code side (32 | 256 ⇒ residues cycle in {c+32j}, bound the minimum).
14. **Risk-ladder staging is the right shape for owner-sanctioned probes**: v7 SAFE (F16-Add, every op Kaggle-landed) / v8 MED (+BOOL-output class, u8-Less) / v9 HIGH (+QL-F16) — all visible 266/266, truth 200k/200k, ab 200,266/200,266 byte_equal. Since v8/v9 are decode-IDENTICAL to LIVE corpus-wide and DETERMINISTIC-class, a real-submit regression would isolate the task080 bool-risk to the grader/runtime rather than reformulation correctness — a sharper experiment than task080 itself ran. Cost-side has zero submit-risk (§39 static ⇒ exact).
15. **Sub-900 impossibility argument (task256-shaped tasks)**: any broadcasting binary final needs per-channel r-side AND w-side operands ≥ 10×30 elements ⇒ ≥ 600 B even at u8; + one [1,30] F32 reduction (Einsum is F32-only) + threshold construction ⇒ > 900 strictly. Write the impossibility WITH the enumeration list (the day's floor proofs fell twice to missing families; this one names every final-op family checked).

16. **Moment-correlation monochrome-window detection kills per-channel grouped-corr architectures** (task143: 15,965 -> 4,160, ab byte-equal 200k): on the color-index plane, window-is-one-solid-color <=> Conv(idx,Tw)^2 == Conv(idx^2,Tw)*nT and Conv(idx,Tw)>0 (variance-zero). F16-exact only while (9*nT)^2 < 2048 (safe nT<=5) -- sweep the corpus nT histogram FIRST. Occupancy-only and no-count-filter shortcuts both diverge at scale (49k/200k and 82k/200k) -- the generator builds nT+-1 near-miss decoys deliberately. Full recipe: Onnx_Advanced/.update/2026-06-06_task143_moment_corr_mono_windows.md.

### 2026-06-06 worker-7 addendum 4 — task256 round 6 (third anti-quit ping): SUB-1500 ACHIEVED (1,476 MED / 1,389 HIGH) via the C13 PAD-FILL

16. **C13 PAD-FILL — compute broadcast operands only over the REACHABLE fire-extent; let ONE multi-axis Pad constant-fill the provably-dead remainder.** task256: bands cannot fire past w=8 (width≤9) and bg past w=12 (W≤13), so the u8 code piece lives on [1,1,1,13] and `Pad(ch-end 6, w-end 17, cv=31)` fills both dead channels AND dead columns in one op — 31 > max-threshold 22 ⇒ never-fires by construction. −153B −50p (three 30-long inits became 13-long). General rule: when a comparison/additive final's operand has a *bounded active extent* (from corpus-verified generator bounds), split it computed-core + pad-fill; the pad constant just needs to clear the never-fire margin on its whole fill region.
17. **The pad-fill constant does double duty across axes** — one cv covers ch4-9 at every w AND ch0-3 at w≥13; verify the margin against EVERY threshold the filled cells can meet (here maxT0=22 < 31; bands ≤9 < 31). Cheaper and more robust than the v8-era per-region residue arguments (the no-QL tier keeps exactly ONE wrap case to check: 282→26>22).
18. **Risk-ladder upkeep: when a cheaper same-tier rung lands, supersede in place but keep the old rung staged** (v8→v11, v9→v10) — the captain gets strictly-better picks per tier without losing fallbacks if a kernel probe fails grader-side.

### 2026-06-06 worker-7 addendum 5 — task256 final ping: 1,389 → 1,334; the window closes with the u8-compare bound REACHED

19. **T2-from-band-mins (QL-land red-row fold)**: the two one-sided band mins `m1 = Min(S−r, 32(r−R))` and `m3 = Min(S−r, 32(R−r))` are BOTH exactly 0 at r=R (their gates cross zero there while smr=L>0), and at r=R±k one is ≤−32k while the other ≤9 — so the exact-row threshold is `T2 = QL(Sum(m1, m3, L_c))` (= L on the row, ≤−15 off, saturates to 0). Kills the Abs/atg pair. General shape: when two one-sided clamps bracket a point symmetrically, their SUM plus the wanted on-point value IS the point-selector — no |·| needed. (Cousin of the v5-era `above_off+below_off+64` trick, surviving translation into the compare class.)
20. **Session arc for the record (12 rungs, one day): 17,062 → 3,938 → 3,627 → 2,631 → 2,515 → 2,403 → 2,306 → 1,838 → 1,679/1,592 → 1,476 → 1,389 → 1,334 (−92.2%).** Four formulation classes (plane / negmask / broadcast-Add / u8-compare), THREE fallen floor proofs, every rung visible 266/266 + truth 200k/200k + ab 200,266/200,266 byte_equal — the entire ladder is decode-identical to LIVE. The owner's repeated "it can be done cheaper" was right four times; the floors fell to idea-classes, never to grinding. Certainty sweep: all 9 staged candidates sha+score re-verified against manifests.

### 2026-06-07 worker — task312 drive-to-floor: 12,211 → 2,777 (22.7%) in one rung-pair; floor proven by family enumeration

21. **Where-as-output with Y=input passthrough (the 5th free-output pattern, candidate for #67)**: for select-or-keep rules (`out = select(mask, replacement, input)`), lift the final Where to canvas scale — `Where(cond[1,1,30,30], X[1,10,30,1], input) -> output` makes BOTH full-canvas 10-ch tensors free (Y=input is the graph input; the result is `output`). Pay only cond + X. Inverts the crop→process→Pad-back shape that charges the 10-ch crop (5,760B F32) and the pre-pad result (2,880B). Pad the COND (bool, 900B), never the data. task312: truth 200k/200k, ab 200,265/200,265 byte-equal, all-static §39-exact. Details: `Onnx_Advanced/.update/2026-06-07_task312_where_as_output_y_input.md`.
22. **§GENERATOR-INVARIANT doctrine fired for real**: 200k corpus showed color-5 only at cols ≥2; officials have 4 cells with 5 at col 1 — a corpus-extent crop would have died on visible. But the col-0 exclusion survived on ALGEBRA (a 5 at col 0 replaces with itself — fixed point), making it crop-safe with zero distribution assumptions; verified with a 202-grid off-distribution probe vs LIVE. Corpus extents are bets; rule-algebra fixed points are proofs.
23. **Param-encoding micro-facts (broadly reusable)**: axes-omitted Slice (starts[4]+ends[4]=8p) beats axes-form (9p) when cropping ≥3 of 4 axes; `Gather(input,[k],axis=a)` (1p, axis is an attribute) beats Slice (3p) for single-index extraction — but keep indices shape [1], a scalar index drops the rank and right-aligned broadcast then mis-addresses.
24. **Floor-proof technique: price the INFORMATION separately from the OP-FLOOR** (task312 round-2, owner <1,400 push): rule information ≈1,272 B vs legal op-floor 2,760 — the 1,488 B gap decomposed into named broadcast taxes (768 B structurally-false cond padding + 720 B structurally-zero col0 rows), each forced because Where broadcast is the only free expansion mechanism at opset ≤18. A floor claim is FINAL when the info-to-op gap is attributed to specific missing fusion ops (sub-region-cond select / code-expansion-in-output) rather than to "tried hard". Owner-number coincidences are diagnostic: 1,400 ≈ the info bound — the ask was structurally right, the op set says no.
25. **task312 floor fell 2,775 -> 600 to the #68-ADDENDUM contraction-final the owner pointed at** — the worker-side blind spot was a "passthrough theorem" (final op must consume input directly) that silently assumed COPY semantics. The #65 question — is the output FULLY-GENERATED-expressible (every cell a map of input, identity included)? — must be asked BEFORE any passthrough/copy floor argument. The per-row color map became ONE 6-operand Einsum-as-output (input x2 free, output free, memory ZERO): selector e0 reads the col0 key INSIDE the contraction; CP-factored map bank (T = 9 identity + 10 key-rank = 19) cut 1,000 dense params to 570. Operand order smallest-first keeps ORT pairwise internals tiny; 0/1-exact monomials keep F32 exact. All three contraction rungs (600/1,030/2,201) gold on truth-full + ab-full first try. Library note: Onnx_Advanced/.update/2026-06-07_task312_colormap_einsum_contraction.md.

## 2026-06-07 (afternoon) — reorg/audit session lessons

- **Folder renames must sweep references repo-wide, same commit.** The top-level Research/ -> Research_Archive/ rename (pre-June) left 717 dead pointers across 349 live files — including real recipe links in 217 task-mds (workers chasing dead paths or, worse, never finding verified rule hypotheses like recipe-task369). Fixed by guarded regex repoint (both / and \ separator styles; Onnx_Advanced/Research excluded). If you rename a tree: grep both separators before calling it done.
- **Manifest schemas vary per worker — never sum dict fields blindly.** Three layouts in staged_submits the same afternoon: top-level cost+profile_cost (246-style), nested cost{score_v2_cost,profile_cost} (321-style), live_cost{}/candidate_cost{} (346-style). A naive reader inflated 321 costs 4.4x and mislabeled a +2.83 candidate as +1.33 with gate-5 violations. repo_lint check_staging now resolves all three and flags >5% profile gaps.
- **staged_submits is a LIVE surface while workers run.** Dirs appear mid-scan; no-manifest warnings on fresh dirs are transient (manifest written seconds later). Re-read immediately before acting on any audit snapshot; never archive/move dirs while workers are active.

## 2026-06-07 (afternoon ship wave) — captain lessons

- **PROFILE==CREDIT law: 12-for-12 exact today** (180×2, 321, 039×2, 265, 209×2, 346, sweep-12-stack, 225, 246-reswap). Static + zero-gap profile = the score is known before Kaggle returns; polls are bookkeeping, not verdicts. Treat them that way (owner called this out).
- **Byte-equal stacks scale**: 12 tasks in ONE submission landed +10.24 exact. For gold-class (byte-equal + profile==static) candidates there is no decode value in splitting; stack freely, spread only when pricing or correctness is actually uncertain.
- **Re-arm verdicts must ship with the prompt text inline** — a board note saying "push armed" is not a deliverable; the 321 worker was retired before the prompt reached them. Now standing: verdict + inline prompt in the same message, file copy only as backup.
- **The anti-quit window pays twice** (265, 346): rounds after the headline rung produced the multi-formulation floor proof and 7 extra kills — close-out discipline is worth as much as the rung. Corollary tier rule: formulation-break audits earn their hour on 5-9k tasks; at 710-class the ceiling (~+0.5) does not cover the spend.
- **Fire-and-poll chains: use ";" not "&&" between submit and poll** — a Kaggle CLI burp after a successful upload short-circuited the poll (265). The submissions-list check is the dup-guard, not the chain operator.

## 2026-06-07 (evening, 048 close + wave wrap) — captain lessons

- **The class-check evidence can come from the candidate's own ab-attribution**: 048 was NOT on the filtered-clean roster — the bet authorization came from the k12 rung's dual-truth attribution showing LIVE itself misses 84/200k (ambiguous-class by direct evidence, no separate LIVE-truth run needed). k6 at 99.374% + visible-floor then won per 079-doctrine. Roster grows to 8; the depth record (079 @98.56%) still stands.
- **Spread shape for a 3-rung ladder with one gold rung**: bets cheapest-first, gold safe-LAST, reswap the winner — all three held, k6 banked +2.54 with k12 as the certain floor the whole time. Submission cost: 4 (incl. reswap).
- **5-architectures-in-one-session is what "drive to floor" means**: 048's worker full-rewrote 4 times (2,411 -> 2,360 -> 2,149 -> 1,153 -> 699); rungs 4-5 (-62%) exist because the anti-quit pings refused an early plateau. Same pattern as 265/346 — the window converts plateaus into idea-class jumps.
- **xlsx auto-sync (2026-06-07)**: `tools/sync_task_xlsx.py` (idempotent CSV->xlsx volatile-column sync + Summary stamp; --check mode) is wired into the pre-commit hook — any commit touching task_state.csv auto-syncs and stages the spreadsheet (fail-open with a loud warning). Tracked hook copy: `tools/hooks/pre-commit` (.git/hooks is untracked — re-copy after clone). Roundtrip + both hook branches dry-run-tested per the monitor rules.

## 2026-06-07 (task359 close) — three reusable doctrines from a formulation break

- **The family/transformation LABEL is a hint, not the rule — the generator is ground truth.** task359 was classified "Symmetry Completion"; reading `external/ARC-GEN/tasks/task_e26a3af2.py` showed it is DENOISING stripe-bands (input ~10% corrupted by random_pixels, output clean — every pixel present-but-noisy, nothing missing). Trusting the label sends workers down a geometry dead-end (mirror/transpose/rotation) that CANNOT work because there is nothing to complete. My own batch-16 prompt gave a "symmetry" technique hint off the label — the worker correctly overrode it by reading the generator. PROMPT FIX: when a family label is given, add "the label is a hint; if the generator contradicts it, the generator wins."
- **Falsification harness = how to prove a floor is ABSOLUTE, not just formulation-relative.** Read the generator, identify the mandatory statistic, then falsify every cheaper formulation on >=3000 FRESH generator pairs. 0/3000 is decisive; a small-sample 100% is luck (median hit 100% on small N but was 99.967% / ~44-fail over 3000 = not exact). task359: per-line-mode 3000/3000, all geometry 0/3000, single-line-broadcast ~14%. Harness pattern saved at `task359/prototypes/falsify_geometry_v15.py` — reuse it.
- **Generator-proven-irreducible is the STRONGEST floor class** (stronger than multi-formulation-skeleton). It proves from ground truth that the mandatory tensor is information-theoretically required: task359's input is present-but-noisy on every cell => the clean value is recoverable ONLY by aggregation over the perpendicular axis => the 2x F32 ReduceSum histograms (2,400 B) are irreducible across ALL formulations, not just the histogram family. Same shape as task260's info-theoretic >=1,249 floor. When a formulation-relative floor's mandatory tensor can be shown info-theoretically necessary from the generator, upgrade the proof to absolute and close permanently.

## 2026-06-07 (task127 close) — the free-output separability trap

- **Algebraic rank-1 != score-cheaper for a free-output conv.** task127's 900-param dense final conv (output free, #67) is rank-1 separable, but the split materializes a [1,5,30,30] F32 intermediate (18,000 B) -> 18,300 total, 20x worse. Factorizing a free-output/final conv trades a charged param tensor for a charged MEMORY intermediate; it only pays if the intermediate < the param saving (almost never at full HxW). Separability pays on INPUT-side convs (intermediate already exists), not the free final. -> techniques #87 FREE-OUTPUT SEPARABILITY TRAP. Standard dense-final-conv floor kill-order: 1x1-collapse (off-center taps ~0) -> separability (only if intermediate < saving) -> groups (partition holds).

## 2026-06-07 (task291 close) — the anti-quit window has an OFF condition

- **Anti-quit self-pings are insurance against premature plateau-quitting, NOT a mandate to burn the full window.** Discriminator observed across 4 closes today: when the floor proof is INCOMPLETE ("I could not find cheaper"), the extra ping rounds earn their keep — 265 (11-decomposition multi-formulation floor), 346 (+7 kills after the headline), 048 (5 architectures, rungs 4-5 from late pings). When the floor proof is COMPLETE in one pass (generator-source-read makes the rule exact + all final-op families killed with numbers), further pings only re-confirm a closed floor = busywork. task291 hit the proven floor in one pass (1756, both emission+detection families killed, einsum-moment route falsified 7/8000) and the worker correctly declined to burn 2h re-confirming it. Captain rule: judge the PROOF completeness, not the clock — a complete kill-table close is done whether or not the window remains; an incomplete one keeps going. Re-arm only to adversarially re-challenge a proof you doubt.

## 2026-06-07 (batch-16 wave wrap) — 8 closes, +8.56, every one a proven floor

- The whole batch came back byte-equal + profile==static + floor-proven-with-kill-table, decoded off the ARC-GEN generator source. Captain work reduced to: ship (gold every time) + the occasional thin-edge-register vs re-arm call. The protocol is fully internalized by the workers.
- Biggest single: task207 2387->200 (+2.48) — a from-scratch occupancy->u8-bitand rewrite. The pattern holds: the largest gains are always full rewrites off the generator, never tail-swaps.
- Three closes were ABSOLUTE floors (not just formulation-relative): 127 (separability/groups/1x1 all killed with numbers), 359 (generator-proven info-theoretic), 207 (sub-200 scorer-rejected). Absolute-floor proofs are the permanent-close standard.
- My own batch-16 prompt-tailoring gave 359 a "symmetry" technique hint off its family label — which was a MISNOMER (it is denoising). The worker overrode it via generator-read. Prompt templates now must say: family label is a hint; generator wins on conflict.

## 2026-06-07 (task235) — repo-equivalent gates match canonical when ng_gate is off PATH

- task235 worker fell back to "repository-equivalent strict gates" (ng_gate not installed), reported byte-equal 200000 vs LIVE, and the ship credited +2.80 EXACT (2158->131). Confirms the fallback gate path produces the same verdict as canonical ng_gate — workers without ng_gate on PATH can gate safely as long as they run the equivalent byte-equal-vs-LIVE-on-full-corpus check and record the substitution in the manifest. Submit-first remains the final judge; the byte-equal-200k gate is the gold gate regardless of tool.

## 2026-06-07 (BATCH-16 COMPLETE) — 20/20 closed, +24.15, the playbook fully internalized

- 20 boladi tasks + the 142 re-arm, +24.15 (6659.34 -> 6683.49); EVERY ship byte-equal-or-more-correct + profile==static, every prediction exact-or-law-confirmed to the cent. Captain work was pure ship+reconcile+thin-edge-register; the workers decoded from generator source, proved floors with kill-tables, and self-assessed the anti-quit cadence correctly every time.
- Day total 6630.57 -> 6683.49 = +52.92 across two waves (afternoon +28.77, batch-16 +24.15) — the program's biggest day by ~2x.
- THREE family-label misnomers (15%): the DeepSeek classification is unreliable enough that captain-tailored technique hints off it were wrong all 3 times. Only the prompt template's "decode the generator FIRST" carried. ACTION for batch-17 prompt-gen: lead with generator-decode, demote/drop the family-derived hint.
- Production validations banked this batch: #89 scorer-invisible einsum temporary (task188 558 credited exactly — L2 card paid for itself), profile==credit law on a +51% dynamic credit-probe (task065), more-correct-than-LIVE full credit (task188 dual-truth), repo-equivalent gates == canonical (task235).
- New reusable primitives: einsum-separable-mask (#89 use), block-upsampler ConvTranspose (#80-inverse), no-Mul scaled-conv decode (#90), Pad-last reorder (#81 corollary, transferred 083->142), free-output separability trap (#87), small-grid inversion (#81), bbox single-marginal (#86).

## 2026-06-07 (task142) — an inherited FLOOR proof is as unreliable as an inherited family-label hint

- task142's padlast rung was documented "519, FLOOR for the 6×6 grid (full 8-final-op-family enumeration in task083.md)". It was wrong: the floor proof CITED task083's kill-table instead of recomputing it. task083's einsum-final KILL number (690) was for a 3×4 active region with TWO distinct axis maps ([3,30]+[4,30]=210). task142 is 3×3 **symmetric** (row-map == col-map), so ONE shared [3,30] map (counted once, 90) + a smaller [1,10,3,3]=360 crop → einsum-final **456 < 519**. Closed at 456 (truth 200000/200000, ab byte-equal 200266/200266 vs LIVE, score 456 static, gate-5 profile 456 0% gap). Staged `task142_einsum_cost456/`. New technique #91 (einsum-final vs pad-last crossover; symmetry shares the map and flips the crossover).
- META: this is the SAME failure class as the batch-16 family-label misnomers, one level up — a *quantitative* inherited verdict (a kill-number) rather than a *categorical* one (a family label). Both are "trust a sibling's conclusion without re-deriving for this task's specifics". RULE: a floor proof that cites another task's enumeration is only valid if the geometry is **identical**; cross-task floor transfers must recompute the kill-table (#91 LAW). The padlast→einsum move only surfaced because the owner's anti-quit prod forced a from-scratch re-examination of a "proven floor" — the prod earned its keep.
- Validated again: #89 scorer-invisible einsum pairwise temp gives profile==static EXACT on a fresh static graph (task142 456=456, 4 real inputs) — consistent with task188's credited-exactly result.
- SIBLING FLAGS (einsum-final candidates, same Spatial/Tiling free-output GridSample→Pad family, but RULE UNDECODED — need decode before any transfer): **task211** (1520, GridSample small_grid [1,9,4,2]→[1,10,9,4]F32→Pad, 9×4 active — asymmetric so two maps, but still likely beats 1520; .md Rule == "unknown"), **task001** (1149 OVERRIDE-konbu17, fractal/tiling — different structure). Both sit at un-optimized free-output remaps; a decode + einsum-final/pad-last pass is high-value (task211 alone is ~1520→~600-700).

## 2026-06-08 (task264 silent-submit-failure) — verify the submit succeeded before polling

- task264 v12 was staged at staged_submits/task264_SIEGE_maxpool_epsfold_cost2605/ but the ship command referenced task264_v12_... (wrong name) -> swap_and_submit printed candidate-not-found, 0 Successfully, NO submission -> the poll then waited forever for a Kaggle entry that never existed. The +2.59 was silently unbanked for ~1h until the owner asked to check it.
- ROOT CAUSE: ship commands pipe swap_and_submit to grep -cE Successfully but do not ACT on the count (assume success, start the poll). A failed submit (wrong path, CLI error) becomes a hung poll, not a visible error.
- FIX: after swap_and_submit, if the Successfully count is 0, the submit FAILED -> re-check the exact staged dir name (ls staged_submits | grep taskNNN) and re-fire; do NOT start the poll. Copy the staged path from an ls, never hand-type the rung suffix. The grep-c==1 is the go-signal for the poll.
- Detection: `kaggle competitions submissions` not listing the task's submit message = it never submitted (vs PENDING = submitted, processing).

## 2026-06-08 (task295 phantom-score) — parse the Kaggle CSV, never grep it

- COMPANION to the task264 submit-failure lesson above. The ship-poll one-liners pipe the kaggle submissions CSV through `grep "taskNNN desc" | grep -oE 'SubmissionStatus.[A-Z]+,[0-9.]+'`. Submit descriptions now contain MANY commas + numbers (costs, ratios, gate counts like 200265, 441/441). On task295 the grep matched a stray number (6637.04) inside the still-PENDING row's description and the until-loop exited early -> a PHANTOM -74 'regression' that nearly got reconciled.
- ROOT CAUSE: CSV fields with embedded commas/quotes/newlines cannot be parsed by line-grep. The publicScore is a SPECIFIC column, not 'the first number after SubmissionStatus.'
- FIX: poll with a tiny python that does `csv.DictReader(... errors='replace')`, finds the row by description substring, and reads `row['publicScore']` / `row['status']` BY COLUMN NAME. Template saved approach: tmp/poll295.py. Also: kaggle CSV has non-utf8 bytes -> decode with errors='replace'.
- SANITY GUARD (caught it this time): a single-task ship can only move the LB by that task's points (<~15, usually <3). ANY delta >5 from one swap = a read error or a corrupted zip, NOT a real score -> STOP and verify (parse properly + check zip integrity) before reconciling. Never reconcile task_state to a surprising score without this check.

## 2026-07-04 (batch48 ALPHA) — COW cached-baseline stack probe; 11/16 "gold" candidates were real-stack mirages; + the manifest git-conflict trap

- **MANIFEST GIT-CONFLICT TRAP (silent gate-kill).** `Backend/manifest.csv` sat with `<<<<<<< HEAD` as line 1 (unresolved sibling-repo merge, `git status` = `UU`). `csv.DictReader` then reads the conflict marker as the sole header, every `r["task"]` KeyErrors, and `load_manifest()` returns **0 entries** (swallowed by its bare `try/except`). Effect: the captain's ENTIRE team-best/no-gain gate silently disabled — `best=None` for every task, so `classify()` never flags "no-gain" CONFLICTs and predicted probe deltas go None/0. DETECTION: digest shows no CONFLICTs for known-owned/at-floor tasks, or a probe prints "expected delta +None". FIX/GUARD: never operate on a `UU` manifest; the COW probe's `if best is None: return` fails safe (refuses to fire against a broken manifest). Worth hardening `load_manifest()` to raise if the header isn't `task,creator,...`.
- **COW cached-baseline stack probe (captain_ctl.py, owner-directed).** The scratch home zip (`submissions/submission.zip`) had drifted from the live leaderboard stack (`Backend/submission.zip` == Team_Best_Onnx) in **291/401** slots, so `_fire`'s per-probe "team-best baseline submit" could not be a single cached number and burned 2 submits/probe. New path (`_fire_stack_cow` + a `cow` branch in `probe-collect`, routed from `probe-stack`/`probe-theory`): baseline = the READ-ONLY live stack (reached via the existing root `Team_Best_Onnx` junction convention), submitted ONCE and cached by stack-sha in `_probe/_baseline_cache.json`; each candidate = a COPY-ON-WRITE scratch zip (exactly one onnx swapped, live stack NEVER mutated), one submit. `probe_kind` stays `"stack"` (batch-size-1 → stageable). ~28→15 submits; delta == the old stale-home delta under the additivity the gate already assumes, and `candidate_status_from_kaggle` still backstops any violation. GOTCHA: exclude leading-`_` files from `pending_probe_specs()` or `_baseline_cache.json` gets parsed as a probe spec (`KeyError: 'tag'`). Owner directives that shaped it: batch to save submits; `_fire` must not re-submit the baseline every probe; use a junction, keep the team-best folder read-only (COW, never write through it).
- **RESULT — real-stack probing is not optional.** Draining 16 HELD candidates (both lanes) via COW: **11 hidden-FAILED at −17..−36 pts** on Kaggle's hidden set (task138/286/134/109/160/308/297/273/323/345/394) despite every one showing a tiny +local gain — i.e. an ISOLATED/absolute/batch probe would have FALSE-CONFIRMed ~200 pts of regression into the stack. Only 5 genuine and staged: task050 **+0.104** (its old stale-home probe had returned a garbage +35.75; the COW re-verify vindicated the real +0.10), task143 +0.042, task072/301/343 ~+0.00 cost-trims. Reinforces the STACK-ONLY gate (`_proposals/…isolated_probe_false_confirm_stack_only_gate_2026-07-03.md`): only a per-task real-stack overlay may certify staging.
