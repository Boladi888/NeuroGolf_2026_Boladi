# task256 — archived history (moved from task256.md 2026-06-06, task233 convention)

Live file: `The_400_Tasks/Task_by_Task/task256.md`. This archive holds the full prose of
superseded rungs and the superseded F16-Add-class floor analysis. The staged manifests
(`staged_submits/task256_*/manifest.json`) and builders
(`task256/prototypes/build_task256_v*/`) remain the canonical per-rung records.

## v7_combohead full prose (SAFE PRIMARY, cost 2,306 = 2,150 mem + 156 params)

`prototypes/build_task256_v7_combohead/build_task256_v7_combohead.py` → `task256_v7_combohead.onnx`
(2,978 bytes, opset 12, 38 nodes, fully static — §39: staged copy re-scored 2,306 exact).
Staged: `staged_submits/task256_v7_combohead_bcastadd_cost2306/`.

**Head (v7's cut): ONE [1,30] reduction instead of two.** A single free-input Einsum
`'bchw,c->bh'` with weights `1 + 64·sel2` gives `combo[r] = rowtot[r] + 64·count2[r]` —
non-red in-grid rows read W, the red row reads W+64L (strict global max, L≥2), empty rows 0:

```
R = ArgMax(combo)                      Max = ReduceMax(combo) = W+64L
Σ = ReduceSum(combo) = HW+64L          L = Einsum('bchw,c->b', sel2)   (scalar, 4B)
W = Max − 64L      HW = Σ − 64L        H = HW / W     (F32 Div; W≥1 structural)
```

All combo values ≤ 525 → F32-exact; scalars ≤ 13 → F16-exact post-cast. (L cannot come from
combo — `Einsum('bchw,c->b', combow)` ≡ ReduceSum(combo), circular — so the tiny sel2 scalar
einsum stays.)

**The assembly class (v4): the final op is a broadcasting `Add(A, B)` that writes `output`
directly** — `out[c,r,w] = A[c,r] + B[c,w]`, A `[1,10,30,1]` (600B), B `[1,10,1,30]` (600B),
and the [1,10,30,30] result is the FREE output tensor. No [13,13]/[30,30] plane exists
anywhere in the graph. The triangle test `w < S−r` is exactly the sign of
`(S−r−0.5) + (0.5−w)`; rank-1 additive separation does natively what the plane formulation
paid 2,704B to assemble:

```
ch1/2/3 (bands):  B[c,w] = 0.5−w  (one shared static init, used 3× in the concat)
                  A[c,r] = (S−r−0.5) + band_off64(r)
                  band_off64 = Min(64·(±k−0.5), 0)  off the pre-scaled arange (ar_rh64 = 64r+32)
                  at-row:  at_off64 = Sum(above_off64, below_off64, +64)   — 0 exactly at k=0
ch0 (bg interval S−r ≤ w < W):
                  A[0,r] = (r+0.5−S) + Min(64(H−r−0.5), 0)      — row cliff at H
                  B[0,w] = Min(w+0.5, 64W−63w−32)               — w-cliff at W in ONE Min
rows 13–29 + ch4–9:  A-Pad constant −64 (margin vs max B = +12.5); B-Pad constant 0
```

All penalties scale by 64 = 2⁶ (exponent shift ⇒ F16-exact; magnitudes ≤ 1,888 < 2,048).
Margins: off-band A ≤ −23.5 vs max B +12.5. ORT CPU `Sum(F16, 3 inputs)` execution-verified
through the full gate set.

**Memory ledger**: combo head F32 156 + casts/scalar-folds 18 + twelve [13,1] vectors 312 +
two [1,30] vectors 120 + A4 104 + **A 600** + B4 240 + **B 600** = **2,150**.
Params: sel2 10 + combow 10 + c64f 1 + ar_r_h 13 + ar_rh64 13 + ar_c30_h 30 + ar_c63p32 30 +
b123 30 + zero_h 1 + big_s 1 + padA_pads 8 + padA_cv 1 + padB_pads 8 = **156**.

**Gates** (DETERMINISTIC class): visible **266/266** · truth full **200,000/200,000** ·
ab vs LIVE **200,266/200,266, 0 divergences, byte_equal=true** — identical gate profile on
every rung v2→v7. Adversarial advisory identical too (divergences only on rule-violating
inputs; 0 in-corpus occurrences).

## F16-Add class floor analysis (superseded as "the floor" by the u8-compare class)

**Epistemic warning (the day's meta-lesson, twice over)**: two successive "floor proofs"
fell IN ONE SESSION — v2's plane floor (3,892) fell to the negated-mask fold (v3 3,627), and
v3's "arithmetic-F16 floor" fell to a **new assembly class** (v4 broadcast-Add, 2,631 → v7
2,306). Both earlier proofs were correct *within their formulation* and blind to a
formulation switch. A floor claim is only as strong as the **final-op family enumeration**
behind it (cf. task055/task166 lessons). (Then the F16-Add floor itself fell to u8-compare —
three times in one day.)

**Final-op family enumeration** (what can write `output` [1,10,30,30] and what its inputs
must cost):
- `Add(A[1,10,30,1], B[1,10,1,30])` — **1,200B** operands, carries r+w coupling via additive
  separation. Sub/equivalents identical. Channel dims must BOTH be 10 (ch0's w-ramp runs
  opposite to ch1-3's, so no [1,1,1,30] shared-B exists; A carries 4 distinct row profiles,
  so no [1,1,30,1] shared-A).
- `Mul(A,B)` — sign(A)·sign(B) = union of two rectangles per channel; triangle ✗.
- `Min/Where(cond,X,Y)` — pure intersections/selections = rectangles per channel ✗.
- `Pad(X)` — X ≥ [1,4,13,13] = 1,352B **plus** the planes to fill it (≥2,704B total) — the
  v2/v3 class, strictly worse.
- `Resize(linear)` from [1,10,30,2] — builds w-affine ramps natively but cannot express
  ch0's w-cliff at W ✗. `Expand`/`Pad(edge)` — w-constant ✗. `MatMul` rank-k — ≥2,400B ✗.
  `ScatterND` — 18,000B base ✗. `Concat`-last — ≥1,800B/plane pieces ✗. `OneHot` — 3,380B ✗.
  `Gather/GatherElements`-last — index tensor ≥ output-shaped ✗. `Tile`/`DepthToSpace` —
  input must already hold all output bytes ✗. Any UNARY final (Relu/Cast/Slice-out) — full
  18,000B input plane ✗. Only broadcasting binary finals compress; Add couples r+w in sign;
  (u8-compare finals Less/Greater couple identically at 1 byte — the v8/v9 class).
- **Class bound, itemized (v7-era)**: A 600 + B 600 + A4 104 + B4 240 + combo head 156 +
  glue (12 r-vectors 312 + 2 w-vectors 120 + casts/folds ~18) ⇒ **≈ 2,146 memory**.
  **v7 delivers 2,150** — 4 bytes off the bound.

**Measured-and-rejected alternates (don't re-price)**: L from combo (`Einsum('bchw,c->b',
combow)` ≡ ReduceSum(combo) — circular; the sel2 scalar einsum is irreducible); dominance-R
à la task247 (needs ch2-restricted row weights = 3-op Einsum §40-caveat or a [10,30] 300p
weight ✗; ArgMax at 14B already optimal); H-gate from Slice(combo,0:13) (+156B vs 72B scalar
chain); A-side +32/B-side −32 shift to kill the A-pad cv (needs a second r-arange: +38 net);
gate-stack via [1,4,1,1]-sign-Mul broadcast (+78); B0-cliff at scale 32 (margin fails at
W≥6: ghost bg at w=W); unscaled at-gate at big=32 (margin fails at S=9: ghost red at (0,0) —
at-gate needs ≥36, hence the Sum(+64) form); B w-begin-pad trick (−15 dust, fragility ✗);
ch0 from input's ch0-plane crop (+824B F32 slice tax); sub-13 crops (ragged concat ✗);
Range-op aranges (output is a counted intermediate ✗ — task055 probe); plane-class
extras from v2/v3: [1,3,13,1] mask-stack Mul (+78), rsum2 'bchw,h->bc' + Div-by-L (−47 old
class, obsolete), at-band via Add(above,below)+shift (equal), Max(p,q_n) |k|-form (equal),
u/v from per-row Slice of rowtot (+94).
