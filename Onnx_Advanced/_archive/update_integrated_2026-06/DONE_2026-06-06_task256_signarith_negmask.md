# task256 — promotable findings (2026-06-06, worker-7)

Shipped-class result: LIVE 17,062 → **2,306 byte-equal-FULL** (v7, ab 200,266/200,266 div=0,
truth 200k/200k, vis 266/266; staged `task256_v7_combohead_bcastadd_cost2306/`; ladder v2
3,938 / v3 3,627 / v6 2,403 also staged). −86.5%, **+2.00 LB raw-ln predicted**. v7's 2,150
memory sits 4 bytes off the enumerated class bound. Full lineage + class-floor analysis:
`task256.md`.

## Promote to techniques.md — **BROADCAST-ADD OUTPUT ASSEMBLY (new class): the final op is a
## broadcasting Add writing `output` directly — no 2-D plane exists anywhere in the graph**

1. **`out[c,r,w] = A[c,r] + B[c,w] > 0` with A `[1,10,30,1]` (600 B) + B `[1,10,1,30]`
   (600 B) as the FINAL op.** The [1,10,30,30] result is the free `output` tensor; the
   per-cell value never exists as a charged intermediate. Any rule whose per-channel region
   is `{(r,w): w ⋛ f(r)}`-shaped (half-planes, staircases, triangles, rectangles, and
   single-sided intervals via ramp cliffs) assembles this way. task256: beat the best
   plane-formulation (concat 1,352 + 4 planes 1,352 + threshold 338 = 3,042) by 2.5× —
   A+B+concats = 1,544 total.
   - Region calculus: monotone-in-w threshold per row = `A[c,r] = thresh(r)`, `B[c,w] = ±w
     + 0.5-offsets` (half-integer offsets ⇒ never 0 on integer grids ⇒ sign-exact).
   - INTERVALS (e.g. bg = `S−r ≤ w < W`) put a cliff in B: `B0 = Min(w+0.5, 64W−63w−32)`
     — base ramp up to the data-dependent W, then a 64-steep drop, ONE Min off one static
     init (63w+32) and one scaled scalar.
   - Band/row gating on A: `A = base(r) + Min(64·(±k−0.5), 0)` off a pre-scaled arange
     (64r+32). Scale 64 = 2⁶ ⇒ exponent shift, F16-exact (keep magnitudes < 2,048).
   - "Exactly-at-row" gate from the two one-sided gates: `at_off = Sum(above_off,
     below_off, +64)` — variadic 3-input Sum, = 0 exactly on the row (both clamps sit at
     −32 there), ≤ −32 off. No quadratic, no Abs.
   - Dead channels + crop rows: Pad A with constant −BIG (margin > max B), Pad B with 0 —
     `−BIG + B < 0` and `A + 0 ≤ 0` everywhere dead. ONE pad cv init total.
   - Margin discipline: prove `off-band A + max(B) < 0` for every region pair, at the
     F16 values actually taken. task256 ghosts found at scale 32 (bg at w=W for W≥6; red at
     (0,0) for S=9) — 64 is the smallest power of two that clears both.
   - What the class CANNOT do: sign(A)·sign(B) products (Mul-final = rectangle unions),
     two-sided coupling in both r and w simultaneously beyond rank-1+cliffs. Enumerated
     dead: Resize-linear (no cliffs), Where/Min-final (rectangles only), MatMul-final
     (rank-k ≥ 2,400 B), Concat/OneHot/ScatterND-final (plane-priced).

2. **Floor proofs are FORMULATION-RELATIVE — enumerate final-op families before believing
   one.** Two correct-within-class floor proofs fell in one session (plane-class 3,892 →
   negmask 3,627 → broadcast-Add 2,403). The checklist that found the break: "list every op
   family that can legally write `output`, price its minimum input set" — do this BEFORE
   pricing stages inside the formulation you already have. (Reinforces task055/task166;
   adds the concrete family list to check.)

## Promote to techniques.md — sign-arithmetic kit (subsumed pieces, still reusable when a
## plane IS forced)

3. **Sign-correctness-only doctrine (>0 decode)**: channels need the right SIGN, not {0,1} —
   every Relu·2·Min clamp ladder is deletable; bands as `Relu(affine)`/`Min(affine,0)`
   (zero off-band = product-safe and add-safe). **Negated-mask identity**: `t·Relu(x) ≡
   (−t)·Min(x_flipped,0)` — pick the polarity most 2-D consumers want, flip the rank-1
   masks (no Neg ops; swap Sub operands).

## Promote to techniques.md (#46-family extension) — **the COMBO HEAD: every scalar the rule
## needs from ONE [1,30] reduction via dominance-weighted Einsum**

4. **Merge the all-channel marginal and the object-channel marginal into one free-input
   Einsum** with weights `w_c = 1 + α·sel_obj[c]` (α a power of two ≥ max marginal value):
   `combo[r] = rowtot[r] + α·objcount[r]`. Then `R = ArgMax(combo)` (object row is the
   strict global max), `Max = W + αL`, `Σ = HW + αL`; one extra scalar einsum
   `L = Einsum('bchw,c->b', sel_obj)` un-mixes everything: `W = Max−αL`, `HW = Σ−αL`,
   `H = HW/W` (F32 Div; W≥1 structural for any non-empty grid; F32 0-div → NaN not SIGFPE).
   Costs ONE [1,30] F32 (120 B) + ~9 scalars instead of two reductions (−97 net on task256).
   Keep α·max-count F32-exact (task256: α=64, values ≤ 525). In-grid clipping then rides
   rank-1 cliffs (`Min(64(H−0.5−r),0)`) — vs ~4.6 kB for a presence plane.
   Caveat: L is NOT recoverable from combo (`Einsum('bchw,c->b', combow)` ≡
   ReduceSum(combo) — circular); the tiny object-selector scalar einsum is irreducible.
   2-op Einsum is §42-trusted; ORT CPU `Sum(F16, 3 inputs)` execution-verified through the
   full gate set (one more dtype datapoint for ort_reality).

## Trap (self-test harnesses)

5. **Hand-rolled verify harnesses must zero ALL channels outside the active grid on BOTH the
   input encoding AND the expected-output one-hot.** `np.eye(10)[idx]` alone marks padding
   cells ch0=1 → a perfect model reads 0/266 (cost a debug round; model was flawless).
   `ng_gate` (official encoder) is the arbiter; self-tests must mask `[:H,:W]`.

## ROUND-2 ADDENDUM (owner sub-1500 push) — promote to techniques.md: **u8-COMPARE OUTPUT ASSEMBLY (the class below broadcast-Add)**

6. **`out = Less(C [1,10,1,30] u8 codes, T [1,10,30,1] u8 thresholds) → BOOL free output.`**
   A comparison final couples r+w exactly like Add-sign (a<b ⟺ a−b<0) but takes 1-BYTE
   operands: the broadcast pads halve (600+600 F16 → 300+300 u8). Bands share ONE static
   arange code init (0 bytes, reused per concat slot); only interval-channels compute their
   code (`Max(base, 32(w−W)+c)` cliff). Integer-native: strict `<` needs no half-offsets.
   task256: 2,306 (F16-Add floor) → 1,679 (opset 12) → 1,592 (opset 19 + QL).
   - **U8-WRAP ASYMMETRY LEMMA**: Cast F16→u8 wraps mod 256 (probed). Wrapped negatives on
     the CODE side = huge codes = never-fire = SAFE; on the THRESHOLD side = fire-all =
     CATASTROPHIC ⇒ Relu every threshold pre-cast, and keep cliff slopes | 256 so code-side
     residues stay provably above max-threshold.
   - **QuantizeLinear(·, scale=1) = SATURATING F16→u8 cast** (QL-19 supports F16; probed
     −30→0, 858→255; ORT CPU executes): kills the Relu clamps AND the residue analysis.
     Opset 19/20 grader risk already retired by task191 (shipped opset 20, Kaggle-confirmed).
   - Risk class: BOOL output = task080 family. Staged as an owner-sanctioned 3-tier ladder
     (SAFE F16-Add 2,306 / MED u8-Less 1,679 / HIGH +QL 1,592), all visible+truth+ab
     perfect & decode-identical to LIVE — a real submit isolates grader/runtime behavior
     from reformulation correctness.
   - Floor verdicts (enumeration-relative, full list in task256.md): u8-compare bound
     ≈1,45x-1,55x; **sub-900 impossible** — any broadcasting binary final needs ≥ 600B of
     u8 operands + a 120B F32 reduction head + threshold construction.

## ROUND-3 ADDENDUM (sub-1500 achieved) — promote with item 6: **the C13 PAD-FILL refinement**

7. **Compute comparison/additive operands only over the REACHABLE fire-extent; ONE
   multi-axis Pad constant-fills the provably-dead remainder.** task256 v10/v11: bands
   cannot fire past w=8, bg past w=12 (corpus-verified bounds) ⇒ the computed u8 code piece
   is [1,1,1,13]; `Pad([0,0,0,0,0,6,0,17], cv=31)` fills dead channels AND dead columns in
   one op (31 > max threshold 22 = never-fires). −203 cost; 30-long inits become 13-long.
   Margin rule: the single fill constant must clear the never-fire margin against every
   threshold its region can meet. Final ladder: v7 SAFE 2,306 / v11 MED 1,476 (opset 12,
   no QL) / v10 HIGH 1,389 (QL @19) — both sub-1500 rungs full-gates-green and
   decode-identical to LIVE on 200,266; sub-900 impossible (operand pads 600 + head 156 +
   threshold construction, itemized in task256.md).

## ROUND-4 ADDENDUM (final): T2-from-band-mins; ladder closes at 1,334

8. **Exact-row selector from the two one-sided clamps (no Abs)**: `m1 = Min(base, +slope·k)`
   and `m3 = Min(base, −slope·k)` are both exactly 0 at k=0 and bracket ≤−slope / ≤base
   off-point ⇒ `T_at = sat(Sum(m1, m3, wanted_on_value))`. Kills |·| chains wherever a
   saturating cast (or Relu) follows. Final task256 ladder, all gates green + decode-identical
   to LIVE on 200,266: v7 SAFE 2,306 / v11 MED 1,476 / **v12 HIGH 1,334** (mem 1,227 sits AT
   the itemized u8-compare bound ~1,235; sub-900 impossible — proof in task256.md).
