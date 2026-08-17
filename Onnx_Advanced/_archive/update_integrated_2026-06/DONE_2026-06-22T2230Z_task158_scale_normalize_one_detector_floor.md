# DONE: task158 scale-normalize-to-ONE-detector = NOT-VIABLE; v111 (24,148) is the convolutional-family floor

**Date:** 2026-06-22 · **Worker:** worker-2 (beta_p21_claude_cli) · **Task:** 158 (scattered magnified+flipped sprite copies, marker-pair detect+stamp)

## Transferable insight (cost-floor / #134 multi-scale detection)

When a prompt proposes collapsing a **per-scale detector stack** (here `s1/s2/s3_valid_scores`
= 5,112 B, one plane per magnification mag in {1,2,3}) into ONE scale-normalized detector by
pooling/Resizing each candidate window to a canonical footprint, the decisive test is:
**does the per-scale acceptance predicate have a scale-INVARIANT part and a scale-PROPORTIONAL
part that pool destroys?**

On task158 the acceptance test is "exact solid marker square of size s" = an `s x s` block of
the marker color surrounded by a **fixed 1-cell ring** of non-marker. The square scales with s
but the ring is ALWAYS 1 physical cell, so the inside:ring ratio differs per scale. No pooling
(MaxPool/AvgPool) or Resize can carry both the `s`-cell solidity AND the 1-cell ring through to
a single canonical 3x3 detector — pooling an `s`-square to one cell discards the ring exclusion
that is exactly what distinguishes "exact size-s square" from "a sub-block of a larger square."
Any scale-collapsed single detector is therefore ring-free at the coarse scales.

**The fixed-physical-vs-scale-proportional asymmetry is the general floor signature**: if a
multi-scale detector's rejection geometry is a constant absolute size while the accepted shape
is scale-proportional, the per-scale planes are a #134 floor and cannot be normalized to one.

## Measured floor proof (reusable method: count channel firing on the official corpus)

Instrument the NumPy rule reference over all 266 official pairs:
- per-scale stamp detections: size-1 = 426, size-2 = 117, size-3 = 57 (all three load-bearing;
  per-pair scale-sets {1}=108, {1,2}=104, {1,3}=47, {1,2,3}=7).
- per-rotation firing: rot0=354, rot1=77, rot2=85, rot3=84 (all four channels load-bearing).
- ring-free detection (the exact scale-collapse equivalent) = **265/266 official** = one visible
  miss → certain hidden 0 (VISIBLE-FAIL axiom). Matches the prior don't-do entries
  "size-2 no-ring 265/266" and "all-rings-removed 265/266."

So no scale or rotation channel is dead, and dropping the ring (the only way to share a detector
across scales) fails visible. WALL confirmed.

## ORT note
- `MaxPool` accepts UINT8 at opset 13; `AveragePool` REJECTS UINT8 ("Type Error: Type
  tensor(uint8) of input parameter") — so even a solidity-averaging pool path is unavailable.

## Verdict
No candidate. v111 (`staged_submits/task158_v111_nonbg_fold_cost24148_sha9349EA5A/`,
SHA 9349EA5A...) stands as team-best and the convolutional-family floor: visible 266/266,
truth-full 199,926/200,000 (== LIVE), AB-full 200,266/200,266 byte-equal, profile==static
= 24,148. Team-best to beat was 24,148; beat-or-wall = WALL.

## Suggested routing
- cost_floor_reference.md (plane-floor taxonomy / floor-verdict ledger): add the
  "fixed-physical rejection ring vs scale-proportional accepted shape ⇒ per-scale detector
  planes are a #134 floor, not scale-normalizable" rule, with task158 as exemplar.
- ort_reality.md dtype×opset matrix: MaxPool U8 OK / AvgPool U8 rejected (opset 13).
- (No library file edited directly per protocol — sorter assigns §.)
