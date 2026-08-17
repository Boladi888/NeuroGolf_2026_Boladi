# Recipe: task071 — bilateral-symmetry denoise (16x16 → 16x16)

> `[B→A]` handoff — 2026-05-04 (Phase 4a of `2026-05-04_1830_both-20hr-beta-backlog-fill.md`).
>
> **Status:** organic strict-clean as of 2026-05-07 Alpha run.
> Implemented as `bilateral-symmetry-denoise`; strict ONNX verifies 265/265.
> **Output shape:** **fixed (16, 16)** for all pairs.
> **Cost:** memory 260908 + params 1334 = 262242, about 12.5230 pts.

---

## Alpha final rule (2026-05-07)

The missing axis refinement is not the kept-color bbox center. For each input:

1. Identify `noise` as the nonzero color whose bbox density is exactly 1.0.
2. Identify `kept` as the other nonzero color.
3. For each horizontal axis `axis2` in `0..30` (twice the real axis), score:
   `sum(input[r,c] == kept and input[r, axis2-c] != 0)` over valid mirror cols.
4. Pick the first axis with the max score.
5. Preserve kept cells. For each noise cell `(r,c)`, fill it with `kept` iff
   `input[r, axis2-c] == kept`; otherwise make it background.

Verification:

- Numpy rule: 265/265 train + test + arc-gen.
- Strict ONNX: 265/265 via `neurogolf_utils.verify_subset`.
- Dispatcher audit: task071 picks `bilateral-symmetry-denoise` at cost 262242.
- `tools/karnak_xref.py` moved organic coverage to 116 and reduced the
  no-current-coverage queue to task019 and task137.

The older H8-H10 queue below is retained as history only.

## Pair-level characterization

- **265 pairs** total. All input shape (16, 16); all output shape (16, 16).
- Each input has exactly 2 non-zero colors. The "noise" color (less common)
  is dropped in the output; the "kept" color (more common) is preserved AND
  EXTENDED into former noise positions.
- The output's kept-color pattern is bilaterally symmetric (h-mirror around
  some axis), but NOT around the global bbox center (Hypothesis 1 failed).

## Failed hypotheses (don't waste time on these)

**H1 — Global bilateral completion around bbox center.** For each kept-color
cell, mirror it via h/v/both flips around the bbox center of all non-zero
cells. **Result: 0/265.** Symmetry axis is not the global bbox center.

**H2 — Per-blob bilateral completion around noise rectangle center.** Identify
noise as a rectangular region; mirror kept cells around the noise center.
**Result: 4/265.** Noise is not always a single rectangle.

**H3 — "Both endpoints non-zero" gate.** For each cell, set output to kept-color
iff input cell AND its bilateral mirror are both non-zero (regardless of
which color). **Result: 1/265.** The kept/noise distinction matters.

## Progress update (2026-05-05 Beta session)

**Key insight:** **noise = perfect rectangle (density=1.0).** The "noise" color
always forms a compact solid rectangle; the "kept" color is the irregular/spread
pattern. Earlier attempts failed because they used count (majority=kept) which is
WRONG for ~1/3 of pairs.

**H_v3 (2026-05-05): noise=highest_density, h-mirror around kept-bbox-center.**

```python
def get_bbox_density(a, c):
    mask = (a == c)
    rows = np.where(mask.any(axis=1))[0]; cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0: return 0
    r0,r1,c0,c1 = rows.min(),rows.max(),cols.min(),cols.max()
    return int(mask.sum()) / ((r1-r0+1)*(c1-c0+1))

def predict_task071_v3(inp):
    a = np.array(inp)
    colors = sorted(set(a.flatten().tolist()) - {0})
    if len(colors) != 2: return None
    dens = {c: get_bbox_density(a, c) for c in colors}
    noise = max(dens, key=dens.get)   # noise = most rectangular
    kept = [c for c in colors if c != noise][0]
    kept_pos = np.where(a == kept)
    if len(kept_pos[0]) == 0: return None
    c_min, c_max = kept_pos[1].min(), kept_pos[1].max()
    H, W = a.shape
    out = a.copy().astype(int)
    for r in range(H):
        for c in range(W):
            if a[r, c] == noise:
                mc = c_min + c_max - c
                if 0 <= mc < W and a[r, mc] == kept:
                    out[r, c] = kept
                else:
                    out[r, c] = 0
    return out
```
**Result: 228/265 = 86%.** This is the new baseline (up from ~40/265).

## Remaining hypotheses for next Beta session (37 failures)

The 37 failing pairs likely need a refined axis. The current approach uses the
**global** kept-color col bbox center, but some pairs may need the center derived
from a sub-region (per-blob) or from the noise block's position.

**H8 — Use noise rectangle center as axis (not kept-color bbox center):**
The noise block is ADJACENT to the kept pattern; mirror the noise cells about
the noise bbox's center column/row instead. Test: does noise_bbox_center give
better axis for the 37 failures?

**H9 — Try all plausible integer/half-integer axes (brute force):**
For each pair, try h-mirror around each col in [0, 15] and half-cols in [0.5,
14.5]. Count of noise→kept transformations that match expected. Pick best.
Verify that the "correct" axis is unique or deterministic.

**H10 — Per-CC axis determination:**
Each noise block may obscure a DIFFERENT sub-blob of the kept pattern. Find
the sub-blob of kept cells adjacent to the noise block; its own bbox center
is the correct axis for that block's restoration.

**Recommendation:** Start with H9 (brute force to determine what axis is
actually correct for the 37 failures), then derive a deterministic rule.

**Do not try H4-H7 from prior session** — they were based on wrong kept/noise
identification (count-based) and won't apply after the density fix.

## Sample inspection

Pair 0:
```
input:                                output:
[. . . . . . . . . . . . . . . .]    [. . . . . . . . . . . . . . . .]
[. . . . . . . . . . . . . . . .]    [. . . . . . . . . . . . . . . .]
[. . . 6 6 . . 6 6 . . . . . . .]    [. . . 6 6 . . 6 6 . . . . . . .]
[. . . 6 6 . 1 1 1 1 . . . . . .]    [. . . 6 6 . . 6 6 . . . . . . .]
[. . . . 6 6 1 1 1 1 . . . . . .]    [. . . . 6 6 6 6 . . . . . . . .]
[. . . . . 6 1 1 1 1 . . . . . .]    [. . . . . 6 6 . . . . . . . . .]
[. . . . . 6 6 . . . . . . . . .]    [. . . . . 6 6 . . . . . . . . .]
[. . . . 6 6 6 6 . . . . . . . .]    [. . . . 6 6 6 6 . . . . . . . .]
[. . . . 6 . . 6 . . . . . . . .]    [. . . . 6 . . 6 . . . . . . . .]
[. . . . 6 6 6 6 . . . . . . . .]    [. . . . 6 6 6 6 . . . . . . . .]
[. . . . . 6 6 . . . . . . . . .]    [. . . . . 6 6 . . . . . . . . .]
```

Note: in input, rows 3-5 cols 6-9 are noise (1s). In output, those positions
become 6 OR 0 depending on the bilateral mirror. The TOP blob (rows 2-5) has
HORIZONTAL bilateral symmetry around col 5.5; the BOTTOM blob (rows 7-10) is
already h-symmetric around col 5.5. So the rule for the top blob: for each
cell with input != 0, output = 6 iff h-mirror(cell) has 6 in input. (Skip H4-H7
maybe — try this 'h-mirror gated by noise = 6' first.)

## ONNX implementation sketch (assuming H4 or similar succeeds)

If the rule turns out to be per-blob bilateral with axis varying per blob:

1. **Identify blobs**: 4-conn CC labeling via iterative MaxPool dilation
   (see `seed_flood_fill_network` for the dilation pattern). One label per
   pixel. ~30 iterations × 9k MACs each = 270k MACs.

2. **Per-blob axis detection**: For each blob, compute the "h-symmetric
   match score" (sum of cells matching h-mirror) and "v-symmetric match
   score". Pick the axis with higher score.

3. **Per-blob center**: Bbox of each blob → center.

4. **Per-blob completion**: For each cell in the blob's bbox, output = kept
   color iff input cell OR bilateral mirror cell (around per-blob center)
   is kept color. Implementation: per-blob mask × Gather-rotated-input.

This is ~700k-1M MACs; ~10-11 pts.

## Files

- This recipe: `Research/notes/recipe-task071-bilateral-symmetry-denoise.md`
- Numpy rule status: TBD; 3 hypotheses tested + failed (this session).
- Three more hypotheses (H4-H7) listed above for Beta to try.

## When to abandon

If H4-H7 also fail and rule remains elusive after ~3 hr investigation, mark
task071 as DEFERRED to Tier 3. The task may require an opset-10-incompatible
op (like `Loop` for iterative refinement, banned per validator) and not be
feasible at all.

---

*Author: 2026-05-04 20-hr Beta-backlog-fill session. Multiple rule hypotheses
tested + ruled out; recipe captures what doesn't work to save Beta time.*

---

## Opus 2026-05-05 refinement (conf=mid, novelty=refined-recipe)

**Refined hypothesis**:

REFINED RULE: noise = the color whose bbox-density = 1.0 (perfect rectangle); kept = the other non-zero color. The transformation REMOVES the noise rectangle and FILLS it with kept-color cells dictated by symmetry-completion of the kept pattern using BOTH horizontal AND vertical mirroring (bilateral, not just h-mirror). Specifically, for each noise cell (r,c): output[r,c] = kept iff at least one of {input[r, 2*cc-c], input[2*rc-r, c], input[2*rc-r, 2*cc-c]} equals kept, where (rc, cc) is the center of the kept-color bbox; otherwise 0. All non-noise cells pass through (kept stays kept, 0 stays 0). Train pair 0 (colors 6/1, noise=1 at rows 3-5 cols 6-9 forming perfect 3x4 rectangle, kept=6 bbox rows 2-10 cols 3-7 → cc=5): the noise cell (3,6) mirrors h to (3,4)=0 but v-mirrors to (9,6)=6 → output (3,6) should be 0 in given output (matches: output row 3 cols 6-9 = [.,6,6,.] not all 6, so vertical-only mirror is too strong). Train pair 1 (colors 2/3, noise=3 forming perfect 4x4 rectangle rows 4-7 cols 0-3, kept=2 bbox rows 3-10 cols 1-7 → cc=4): the row-3 markers at cols 3,6 mirror h around cc=4 to cols 5,2 — output row 3 stays unchanged with 2s only at orig positions, but rows 4-7 cols 0-3 get filled by h-mirror of cols 5-8. Inspecting pair 1 row 4: input cols 4-7 = [2,2,2,2], output cols 0-7 = [.,.,2,2,2,2,2,2] — so noise cells at cols 0-3 became [0,0,2,2]; h-mirror of col 0 around cc=4 is col 8 (input row 4 col 8 = 0), col 1→col 7=2, col 2→col 6=2, col 3→col 5=2. That gives output row 4 = [0,2,2,2,2,2,2,2] but actual is [0,0,2,2,2,2,2,2]. So col 1 should be 0 not 2 — meaning the simple h-mirror around bbox cc fails here too. The TRUE axis must be derived from the kept pattern's own symmetry: find the column c* such that the kept-color cells (excluding noise region) are maximally h-symmetric, then mirror noise cells around c*. For pair 1, kept cells outside noise span cols 1-8 with apparent symmetry around col 4.5 (half-integer). Mirroring noise cell (4,1) around 4.5 → col 8 = 0 → output 0. ✓ Mirroring (4,2) → col 7 = 2 → output 2. ✓ This matches.

**Verification plan**:

Step 1: For each train pair, identify noise = color with bbox-density=1.0, kept = other color. Step 2: Brute-force search over all candidate h-mirror axes c* in {0, 0.5, 1, 1.5, ..., 15} AND v-mirror axes r* in same range. For each (r*, c*), compute predicted output: noise cells → kept iff any of (h-mirror, v-mirror, both-mirror) lands on a kept cell in the original input AND that mirror position is OUTSIDE the noise rectangle; non-noise cells pass through. Score = number of cells matching expected output. Step 3: Verify the winning (r*, c*) per pair equals the kept-pattern's own symmetry axis: compute the axis by finding c* that maximizes |{(r,c) kept-input : input[r, round(2c*-c)] also kept-input AND (r, round(2c*-c)) outside noise bbox}|. Step 4: Confirm whether axes are always half-integer (between two cols) or sometimes integer; whether v-mirror is needed at all or only h-mirror suffices for all 265 pairs. Step 5: If 265/265, lock the rule. If <265, inspect failures for diagonal symmetry or per-blob axes (H10).

**ONNX path**:

Major opset-10 wall: the symmetry axis is INPUT-DEPENDENT (per-pair c*, r*), so Gather indices cannot be baked at build time. Workaround: precompute ALL 31 candidate axes (16 integer + 15 half-integer) as 31 separate Gather operations along W axis, producing 31 mirrored tensors of shape [1,10,16,16] stacked along batch axis → [31,10,16,16]. Same for H axis → [31,10,16,16]. Compute per-axis symmetry score: for each axis, ReduceSum over (kept_channel * mirror_kept_channel * (1 - noise_mask)) → 31 scalar scores. ArgMax → winning axis index. Use Gather along the stacked-batch axis to select the winning mirrored tensor. Then: noise_mask * (mirror_kept_at_winning_axis) → fills noise positions with kept where mirror lands on kept; combine with non-noise pass-through via Where. Noise-vs-kept identification: per-color bbox-density is hard in pure ONNX; workaround = compute per-color (non-zero count) / (bbox area via row_has*col_has ReduceSum products). Use ArgMax over 10 channels of density score to pick noise channel. Cost-heavy ops: 31x stacked Gathers (~31 * 16 * 16 * 10 * 4 bytes = 317k bytes intermediate — over budget). Trim: only test integer axes 0-15 + symmetric half-axes constrained to plausible range = 16 axes → ~163k bytes. Final cost: ~200k-240k bytes if careful with intermediate reuse and casting masks to uint8.

**Cost estimate**: 230000 bytes

**Source**: archived 2026-05-05 Opus hard-task synthesis.
