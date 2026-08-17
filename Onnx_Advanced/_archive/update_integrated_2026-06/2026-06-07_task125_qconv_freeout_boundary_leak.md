# task125: QLinearConv-as-output at 28% of LIVE + the free-output boundary-leak trap

**Ship-class result** (12,905 → 3,668, −71.6%, all gates gold): the #67 free-output idea
extended to **QLinearConv emitting [1,10,30,30] u8 directly**, with the whole rule riding
u8 planes (#74). Key facts banked for the library:

## Technique facts (extend #67/#74/#77)

1. **QLinearConv-as-output works and prices at ZERO** — `pads=[1,1,16,16]` style asymmetric
   pads let a 15×15 (or 13×13) X emit the full 30×30; padding cells = x_zero_point ⇒
   dequant-0 ⇒ all channels decode empty. Output dtype u8 decodes fine through `(out>0.0)`
   (task205 precedent). **y_zero_point must be u8 — INT8 y is NOT_IMPLEMENTED in ORT
   (probed). All 8 QLinearConv inputs are MANDATORY** (no omission via empty names or
   truncation — probed both). Scale/zp scalars are shareable: one_f32 (x/w/y scales),
   zero_u8 (x/y zp), zero_i8 (w zp) = 3 params total.
2. **Linear taps inside the output conv are free compute**: a 3×3 ones block on a binary
   channel = the 8-neighborhood count A, computed INSIDE the output node — A never exists
   as a tensor. Integer i8 taps + INT32 accumulation = exact; u8 saturation floors
   negatives to 0 = a free Relu for sign-consuming channels.
3. **MaxPool u8/i8 directional dilations** (kernel [1,K]/[K,1], asymmetric pads) are
   0-param replacements for shifted-ones Conv counts whenever only the indicator SIGN is
   consumed downstream. Inclusive-of-self windows are safe on empty cells (s=0 there);
   shape-cell contamination is suppressed by −k·s center taps. u8 Mul/Sub/Concat all fine
   at opset 14 (probed; extends #74's probe list with Mul(U8), MaxPool(U8/I8), Pad(U8,
   value-init), Cast(F32→U8/I8)).

## THE TRAP: free-output convs with positive off-center taps leak across the X boundary

With content at the grid edge (e.g. a shape cell at row 14), an output cell OUTSIDE the
active grid (row 15) still has the edge cell inside its 3×3 window → any channel with
**positive off-center taps** on a content channel fires a phantom there → official decode
mismatch (ground truth is all-zero outside the grid). Found by synthetic border stress
(500 grids, 277 with border content): task125 v1 passed 204k corpus+visible gates yet
failed 185/277 border-6 grids. **Negative off-center taps are safe** (u8 saturation);
center-only positive taps are safe (output = padded X exactly).

Mitigations, in cost order:
- **Corpus-certify the margin** (§37 style): here 6s touch the border 0/200,000 → ship the
  leaky-but-cheaper rung as primary (hidden = same generator), and
- **Stage a center-only-positive sibling**: move the neighborhood count to an explicit
  `a01 = MaxPool 3×3` channel, output conv 1×1 (W [10,C,1,1]) — task125 v5, +434 cost,
  500/500 border stress. The pair (v4 primary + v5 safe sibling) is the recommended shape.

## Round-2 additions (sub-2,000 push, same day)

4. **Variadic `Min`/`Max` collapse AND/OR chains to ONE tensor** — `Min(dL,dR,dU,dD)` u8
   4-input works in ORT at opset 14 (probed): n-way AND of indicator planes = 1 charged
   tensor instead of n−1. The Boolean cousin of #75's variadic `Sum`. Saved 338 here
   (task125 v6/v7); grep any Mul/And-chain on {0,1 } planes fleet-wide.
5. **OneHot-as-output bank-shot (untapped, for OTHER tasks)**: ONNX OneHot maps
   out-of-range indices to ALL-OFF — so `OneHot(canvas, depth=10, axis=1) == output` with
   canvas = color-index grid and out-of-grid cells set out-of-range gives free
   padding-class handling with NO sentinel-Pad/Equal pair. Trap: ORT OneHot needs
   int32/int64 indices ⇒ [1,30,30] i32 canvas = 3,600 B, beating Equal-tail only for
   tasks whose current cost ≫ 4k and whose canvas build is cheap.
6. **Exact-value canvases cost ~2 channels more than sign-predicates**: priority rules
   (here 6 > 4 > 3 > 8) need interaction channels (aP = a01·P, sP = s·P) to linearize —
   consistent tap-set V = 11s + 8e − 5a01 − 4P + 5aP − sP banked for reuse — but the
   6-channel X + V + pad assembly priced 5,551 vs 3,330 for 4 sign-channels straight to
   the free output conv. Sign-consuming decode ⇒ never build values.
7. **Negative invariant (measured, 200k)**: task125 boxed needs ALL FOUR directions —
   every 3-subset AND has ≥359k false positives, 2-subsets ≥1.8M. Don't burn a session
   retrying directional discounts.

## Spread targets

Any task whose LIVE ends `Concat(sign-planes) → Pad → output` with a per-cell local rule
is a candidate for this rebuild (the 97-task `Pad→output` scan-list from #67 overlaps).
The win here: Concat 4,500 + 16 F16 planes → 1 u8 X (675) + free conv. Magnitude −71%.

*Gates: visible 265/265 · truth 200,000/200,000 · ab 200,265/200,265 byte_equal=true (all
of v1/v3/v4/v5 — v2 visible+truth-20k). Builders: `task125/prototypes/build_task125_v[1-5].py`.
Staged: v4 (3,668 primary), v5 (4,102 safe sibling), v1 (3,882 ladder). Border-stress
script: reproduce via `build_task125_v5.py` docstring pointer.*
