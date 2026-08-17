# Gated multi-distance detector union — the union plane is NOT mergeable (task042, 2026-06-17)

**Context / where this belongs:** extends #131 (forced-floor for paint-mask-preserve) and #137/#135 (opset dtype walls). Likely a short addendum to #131 or a new "detector-union floor" note; sorter to place.

## Pattern
A "detect-then-paint" rule that switches DETECTOR by a global count gate (task042:
count of color-3 cells ∈ {2,4}→n=1, {8}→n=2, {18}→n=3, each a diagonal pair-detector at a
different scale). The build is `Where`-as-output (#81) over a union of K gated QLinearConv
pair-detectors. Floor = FORCED (#131: 900 Where-cond + 400 F32 input-read) + DETECTION.

## The tempting-but-WRONG shrink (measured, killed)
To cut the DETECTION third, the obvious lever is "collapse K detector planes into fewer".
On task042 all three failed:

1. **Merge two distances into ONE dense conv** (n=2 d=2 + n=3 d=3 → one 13×13 dense kernel
   with both tap-sets, gated by `Or(gate2,gate3)`): priced −181 (2,553→2,372) BUT **visible
   196/266 FAIL**. The merged tap-set false-fires across distances — n3 taps trigger on
   n2-active grids and vice versa, because the shared OR-gate lets BOTH tap-sets fire. A
   2-group `[1,2,k,k]` conv restores correctness but its [1,2,10,10] output (200 B) is not
   cheaper than two separate [1,1,10,10] convs (200 B). **Rule: mutually-exclusive-gate
   detectors must stay HARD-SEPARATED; you cannot share a conv's output plane across
   distances whose taps can coincide on off-distance inputs.** Verify on the full visible set
   — the false-fire is exactly the kind of off-generator disagreement a small sample misses.

2. **Sum gated-u8 planes + one Greater** (replace K Greaters+Ors with Add-then-threshold):
   REJECTED — `Add(uint8)` is unsupported under official strict shape-inference at opset 13
   (`A typestr: T, has unsupported type: tensor(uint8)`). The scorer returns None.

3. **Greater-first on [1,2,h,w] then ReduceMax(bool)** to drop a u8 ReduceMax: REJECTED —
   `ReduceMax(bool)` is unsupported at opset 13 (needs opset 20; ReduceMin(bool) never — #135).

## Takeaway (transferable)
For a K-distance gated-detector union under opset 13: the minimal tail is **K depthwise/grouped
QLinearConv outputs + (per-distance gate-folded) Greater + (K−1) BOOL `Or`s** — all in u8/BOOL,
each [1,1,h,w]. You cannot fold below K detector signals (the gate hard-separates them), and the
two cheap fold ops (u8 Add, bool ReduceMax) are opset-13-blocked. So a gated-union detector's
DETECTION cost ≈ K·(conv plane + gated plane) + (K−1) Or planes is a **forced floor for the
formulation** — a lower number needs a different DETECTOR ALGORITHM (Path-B, #134), not a tail
fold. task042 stands at 2,553 = ~1,300 forced + ~1,254 minimal-detection.
