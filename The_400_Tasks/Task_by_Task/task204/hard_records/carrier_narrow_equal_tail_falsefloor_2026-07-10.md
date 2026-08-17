# DEAD ROUTE — task204 CarrierNarrowEqualTail (false-floor attack)

**Recorded:** 2026-07-10 by `bo_ng_cld_worker3` (Claude lane), batch58.

## Live bar (refreshed at session start)
- **live_cost = 7682** (`tools/ng_current_bar.py --task 204` → manifest, authoritative, NOT stale).
- **live sha256 = d2f77ac3e6bca0c1e5afdb88378da368ae17805bb70bb62651e5d9d10b150c75**
- LIVE graph name = `task204_v2_topleft_length`; profile (`--arcgen 266`, 268 real inputs) = **7682** (static==profile, no reprice gap).
- LIVE visible = 268/268 PASS. score_v2 = (macs 0, mem 6652, params 1030).

## Attempted route (prompt Direction)
"Carrier-narrow sweep: collapse 10-channel/F32 spatial carrier to U8/color-index
before Conv/Reduce/Where; delete charged carrier/canvas/template." Forced-floor
target ~900 = "one spatial predicate/color-index plane + output tail."

## Why it is DEAD

### 1. The charged component is ALREADY GONE (primary DROP reason)
The Direction targets a "10-channel/F32 spatial carrier." Graph inspection of
the LIVE extract (sha d2f77ac3…) shows **no such intermediate exists**:
- Input is F32 `[1,10,30,30]` — but that is the officials' input tensor, which
  is FREE (inputs are not charged).
- The graph immediately `Slice`s to 1-channel F32 crops (`ch1_det_f32
  [1,1,18,20]`=1440, `ch0_row_f32 [1,1,1,20]`=80) and `Cast`s to U8. Every
  compute plane after that is U8. Largest intermediate = `topleft_len_u8
  [1,8,18,18]` U8 = 2592.
The 10-channel F32 carrier + `[1,10,20,20]` BOOL Concat tail belonged to the OLD
booltail/pre-v1 graphs (cost 64789 / 80379). The v1→v2 rewrite that landed 7682
already deleted them. **The carrier-narrow sweep is already done.** Per the
2026-07-10 workflow update ("if it is already gone … return DROP with the
refreshed floor note"), this alone is a DROP.

### 2. The ~900 floor_est is a FALSE floor
900 ≈ the `label30_u8 [1,1,30,30]` output tail alone. It assumes the two-color
fill can be computed for free from "one plane." It cannot — see below.

### 3. A single color-index plane cannot encode the two-color size-parity split
Rule (ARC-GEN 868de0fa): interior filled `orange(7) if side%2 else red(2)`.
Color is a function of the **enclosing square's side-length parity**.
- Fresh ARC-GEN (300 pairs): **229/300 need BOTH colors**; **243/300 contain
  red**. A single-color-index plane misses ≥81% of examples.
- **GATE (visible):** patched LIVE with paint bank collapsed to one color
  (all fills orange), byte-identical otherwise → `task204_probe_singlecolor.onnx`,
  cost 7682 (identical), **visible 50/268 = 18.66% FAIL** (§VISIBLE-FAIL AXIOM ⇒
  certain hidden 0). The 50 passers are all-odd-square grids. Proves the
  8 length-specific channels (the "repeated Conv masks" the forced-floor wants
  deleted) are load-bearing — they carry the size→parity discrimination.

### 4. The "cheap" crossing-parity route actually prices ABOVE the floor
- Horizontal crossing-parity interior detection is WRONG here: top/bottom
  fully-blue edge rows create false interiors on odd-L squares (fresh-ARC exact
  interior match = 14/300). Correct interior needs edge-orientation separation
  (more ops).
- ONNX `CumSum` has **no U8 kernel** (spec T ∈ {int32,int64,uint32,…}). A
  crossing-parity plane forces INT32 `[1,1,20,20]`=1600-byte planes (blue_i32,
  cumleft_i32, parity_i32, interior_i32 …). Four such planes = 6400 before any
  tail — already ≥ the 8-channel detector+paint (2592+800=3392) it would
  replace. Net LOSE even before the two-color problem.

### 5. Structural floor argument (independent of tried-path claims)
Determining a cell's enclosing-square side-length parity requires a receptive
field spanning the whole square AND per-size discrimination. A single
translation-invariant conv channel cannot emit a per-square-constant parity
value (as (i,j) moves within the interior the outline shifts; a linear kernel's
sum is not shift-invariant unless constant, which loses the size signal). The
only realizations are (a) 8-way template match (current: detect+paint 3392), or
(b) scan/run-length via CumSum/CumMax (INT32 wide planes, more expensive), or
(c) unrolled propagation (much more expensive). (a) is the cheapest → the v2
graph is the efficient realization. Confirms the 2026-06-24 floor proof + the
2026-06-30 zero-sum theorem independently.

## SECOND PASS (2026-07-10, forced re-attempt) — genuinely-different formulation BUILT + MEASURED

Reviewer pushed for a from-scratch formulation with a different dominant
intermediate, not a micro-tweak. Mined `INDEX.md`→`techniques.md` (#50
crossing-parity + F16 triangular-MatMul prefix-sum; #48 output-overlay) and the
BoBrain provider (manifest 7682 authoritative).

**Correctness win, cost wall.** Built `task204_v3_crossparity_span.onnx`
(`prototypes/build_task204_v3_crossparity_span.py`), a fully independent rewrite:
- interior = crossing-parity ray-cast (#50): `H = blue∧shiftL∧shiftR`, column
  prefix-sum via `tril_f16 @ H_f16` (NOT CumSum), `mod 2`.
- color = **vertical-span parity**: nearest-blue-below − nearest-blue-above in
  the column == L−1 (robust: "nearest" is the square's own top/bottom edge).
  Realized as one causal `MaxPool` prefix-max per direction. `L odd ⟺
  (near_up+near_dn) odd`. NO 8-channel detector, NO paint bank.
- Proven correct **500/500 fresh ARC-GEN** (scratchpad probe2) and **visible
  268/268 PASS**.

**Measured cost (full profile):** `static=18636 profile=18636` (memory 18140 +
params 496; 268 real inputs) — **+10,957 vs the 7682 floor (2.43×), a regression.**
`ng_graph_cost_diff` named delta: removed detector+paint (−2592 mem −800 param)
+ `ch1_det_f32` (−1440), added the F16 scan pipeline (blue_f16/H_f16/cum_f16/
up_src/near_up/dn_src/near_dn/span_f16 = 8×800 = 6400), `tril_f16` (800), a
SECOND F32 read `black_f32` (1600), and ~15 U8 planes. Params fell to 496 but
memory ballooned to 18140. Did NOT run `--full` truth/ab: gate hygiene forbids
`--full` on a confirmed cost-loss (WORKER_PROTOCOL wins over the prompt).

**Why (A) beats (B) — forced-component argument.** Two-color fill forces a
per-square side-length-parity measurement spanning the whole square. Exactly two
realizations exist:
- (A) per-length template discrimination (v2): the size axis is ONE dense U8
  tensor `[1,8,18,18]`=2592 (1 B/elem) + 8-ch paint 800. Work ≈ 4600 → 7682.
- (B) scan/prefix (v3): the size axis becomes ~6–8 F16 `[1,1,20,20]`=800 canvas
  planes (2 B/elem, irreducible ≈ 6400) + forced F32 read + logic → 18636 measured,
  ~11k even if every avoidable U8/F32 plane is fused away (still 1.4×+ over floor).
(A) is cheaper because it packs the entire size axis into a single 1-byte-per-elem
tensor; (B) spreads it across 2-byte F16 planes, one materialised per scan step.
Zero-sum theorem: param↔memory reshuffles are neutral; the only sub-floor levers
are deleting a load-bearing tensor or shrinking the forced F32 read — both proven
impossible. v3 empirically confirms the scan family is strictly worse.

## VERDICT: WALL. 7682 (v2 template-match) is the exactly-priced floor and the
## cheapest correct formulation. The genuinely-different crossing-parity+span
## rewrite is correct but profiles 18636. No route beats 7682.

## One remaining hypothesis (only if this ever revives)
The single forced F32 read (`ch1_det_f32 [1,1,18,20]`=1440) is 19% of cost. IF a
future opset/kernel allowed a direct U8 read of one input channel WITHOUT an F32
slice+cast intermediate (e.g. an integer Gather that outputs U8 straight from
the F32 input without materializing an F32 plane — none exists at opset 13/14
today; `Gather` output dtype == data dtype == F32, so it loses), ~1400 B could
drop. Not currently expressible. Everything else is proven load-bearing.
