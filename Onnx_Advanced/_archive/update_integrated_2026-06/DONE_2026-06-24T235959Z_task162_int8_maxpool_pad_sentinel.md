# task162 INT8 QLinearConv architecture — MaxPool pad sentinel and Clip requirement

**Task**: task162 (Fill 3×3 all-black holes with blue)
**Session**: Worker 12, 2026-06-24
**Builder**: `The_400_Tasks/Task_by_Day/task162/prototypes/build_task162_v4_int8.py`

---

## What I discovered

### 1. MaxPool INT8 padding sentinel is INT8_MIN (-128), not 0

When using MaxPool with `pads=[p,p,p,p]` on INT8 tensors, the padded region uses -128 (INT8_MIN) as the fill sentinel, NOT 0. This matters for correctness because:

- If the input tensor has NEGATIVE values (e.g., from a previous QLinearConv with y_zp=-8), the padded -128 values are smaller than most negatives, so the window max can be a negative input value.
- Example: MaxPool at a corner position where the window is mostly padding (-128) plus one input cell with value -3: max(-128, ..., -3) = -3.
- If you then do `Cast(dil_i8, BOOL)` to detect "any primary anchor in window", the value -3 → True (false positive).

**Fix**: ALWAYS use `Greater(dil_i8, 0)` (not Cast) when the upstream tensor can be negative.

**Verified**: ORT 1.24.4, opset 14, INT8 MaxPool with pads=[2,2,2,2] on all-(-100) input → corner output = -100 (padding is -128 < -100, so max is the input value).

### 2. The anchor binarization (Clip) before dedupe is mandatory in INT8 pipelines

When doing a "3×3 all-black detection + dedupe" pipeline with QLinearConv:

1. QLinearConv 3×3 (sum, y_zp=-8) → sum3_shifted ∈ [-8, +1]
2. **Clip(sum3_shifted, 0, 1) → anchor_i8 ∈ {0, 1}** ← CANNOT skip this
3. QLinearConv 2×2 (kernel=[[-1,-1],[-1,+1]], pads=[1,1,0,0]) → nbr_check_i8

If you skip step 2 and feed sum3_shifted directly to step 3:
- For a cell with sum3=0 (→ sum3_shifted=-8) surrounded by other sum3=0 cells:
  - output = (-8) - (-8) - (-8) - (-8) = +16 → SPURIOUS PRIMARY ANCHOR
- This causes false fill in completely empty (non-black) regions.

The Clip to {0,1} binarizes the anchor signal so the 2×2 conv only sees values that are truly anchor.

**Verified**: numpy simulation on 10,000 ARC-GEN samples, 0 errors WITH clip, systematic errors WITHOUT clip.

### 3. INT8 QLinearConv path for 3×3 block detection gives 4,727 cost (vs 8,553 F16)

For task162 specifically, the INT8 path saves 44% over the F16 path:

| path | memory | params | cost |
|---|---:|---:|---:|
| F16/F32 (v2) | 8,500 | 53 | 8,553 |
| INT8 (v4) | 4,672 | 55 | 4,727 |

The savings come from:
- Switching 3×3 sum conv from F32 → INT8: 18×18 instead of 20×20, and INT8 vs F32 (324 vs 1600 bytes)
- The QLinearConv y_zp=-8 trick: bake in the -8 shift (sum-8 gives >0 iff anchor) without a separate Sub node
- Collapse dedupe (2×2 conv) to operate on 18×18 INT8 binary map, then MaxPool to dilate to 20×20

But the INT8 path still exceeds the manifest team-best (4,066) because it requires 3 mandatory 18×18 tensors (sum3_shifted + anchor + nbr_check = 972 bytes) vs the manifest's single 18×18 (324 bytes). The manifest achieves its cost by incorrectly combining anchor+dedupe into one step — it's wrong (265/266 visible, ~98% truth).

### 4. Single linear conv + threshold CANNOT correctly do anchor+dedupe

Proven impossible: no single kernel exists that gives sum_anchor > threshold when current cell is primary anchor AND gives sum_anchor ≤ threshold when current cell is a non-primary anchor (blocked by neighbor). 

Counterexample: when anchor is at (r,c) with a blocking neighbor at (r-1,c-1):
- Both cells are sum3=9, so they're indistinguishable by a linear kernel operating on the same-scale values
- A kernel that "charges" neighbor anchors will charge them REGARDLESS of whether they block or share cells with the current anchor
- The manifest's 5×3 kernel fails in practice at exactly the cases where neighbor columns exist

This is an IMPOSSIBILITY RESULT — no future worker should attempt a single-kernel anchor+dedupe.

---

## Where encountered

- task162, `The_400_Tasks/Task_by_Task/task162/prototypes/build_task162_v4_int8.py`
- All gates verified: 266/266 visible, 200k/200k truth, profile=static=4,727

## Why it matters

- The MaxPool INT8 sentinel trap is applicable to ANY pipeline that: (a) uses MaxPool on INT8 with pads, AND (b) uses Cast(output, BOOL) to detect positive values. The sentinel means negative input values can leak through MaxPool boundaries.
- The Clip-before-dedupe pattern is applicable to ANY convolution pipeline where: (a) step 1 gives a non-binary signal that is positive iff the desired property holds, AND (b) step 2 does a "neighbor subtraction" style dedupe.

## What's missing from the library

- `ort_reality.md`: Add note that MaxPool INT8 pad sentinel = INT8_MIN (-128), not 0. (Check if UINT8 uses 0 or UINT8_MAX.)
- `traps.md` or `techniques.md`: Add "Clip-before-dedupe mandatory" pattern for INT8 pipelines.
- `traps.md`: Add note under MaxPool: "Cast(maxpool_int8_output, BOOL) is dangerous if upstream can be negative — use Greater(output, 0) instead."

## Sorter note: ort_reality.md — add MaxPool INT8 pad sentinel fact (INT8_MIN = -128, not 0; UINT8 TBD)

## Sorter note: traps.md §VISIBLE-FAIL or new trap — add "INT8 MaxPool pad sentinel" trap: MaxPool INT8 pads use -128, so negative upstream values leak through corner windows; always use Greater(output, 0) not Cast(output, BOOL)

## Sorter note: techniques.md — add "Clip binarization mandatory before conv dedupe" (BOOL/INT8 dedupe via neighbor-subtraction conv requires binarized input; non-binary values cause +N - (-N) = 2N spurious hits)
