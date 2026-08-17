# Recipe: task270 — anchor crosses (1, 2) from row/col-aligned cells

> `[B→A]` handoff — 2026-05-05.
>
> **2026-05-08 status:** `[B->A]` verified. The refined color-partition rule
> reaches **266/266**: train 3/3, test 1/1, arc-gen 262/262.
>
> **Superseded 2026-05-08:** old majority-color rule was **256/266 = 96.2%** with majority-outer-color
> selection. 10 failing pairs have an ambiguous cell at the intersection of
> one center's row and the other center's col — disambiguation rule TBD.
> **Output shape:** preserved.
> **Estimated cost:** ~50-100k → ~14 pts (if rule extended to 100%).

---

## Verified rule (266/266)

The center colors are partitioned globally before any arm is drawn. Collect the
non-center cells of colors 3 and 7. For each color, count only unambiguous cells
aligned with the row or column of each center; cells aligned to both centers are
excluded from this assignment tally. Assign each color to the center with the
larger unambiguous tally, then draw that center's adjacent cross arms only in
directions containing at least one cell of its assigned outer color.

Verifier output from `tools/verify_beta_recipes.py` on 2026-05-08:

```text
task270 unambiguous_color_partition_draw_all_aligned_outer_cells: 266/266
train 3/3; test 1/1; arc-gen 262/262
verdict: B->A
```

Acceptance for Alpha handoff: use this refined rule, not the older per-center
majority rule below.

## Superseded majority rule (256/266)

Input: 15×15 grid containing exactly one cell of color 1 and one of color 2 (the
"centers"), plus scattered cells of colors 3 and 7.

Output:
1. Keep both centers in place.
2. For each center, emit a 4-conn cross of the **center's outer color** at distance 1.
   The cross is partial: only the cardinal directions that have at least one
   scattered cell of that color in the same row or col get drawn.
3. Remove all scattered (non-center) input cells.

**Each center has its own outer color** (one of {3, 7}). Per pair, the pairing is
inferred from input data:
- The 1-center pairs with the color whose scattered cells share its row or col.
- The 2-center pairs with the OTHER color.

In observed train pairs the canonical pairing has been 1↔7, 2↔3, but in
arc-gen variants the pairing flips (e.g., pair 56 has 1↔7 + 2↔3, others may flip).

## Numpy verifier (256/266 confirmed)

```python
def predict_task270(inp):
    a = np.array(inp); H, W = a.shape
    out = np.zeros_like(a)
    cells_1 = np.argwhere(a == 1); cells_2 = np.argwhere(a == 2)
    if len(cells_1) != 1 or len(cells_2) != 1: return None
    c1 = tuple(cells_1[0]); c2 = tuple(cells_2[0])
    out[c1] = 1; out[c2] = 2

    for (cr, cc) in [c1, c2]:
        in_row = [(r, c) for r in [cr] for c in range(W)
                  if a[r, c] not in (0, 1, 2) and (r, c) != (cr, cc)]
        in_col = [(r, c) for c in [cc] for r in range(H)
                  if a[r, c] not in (0, 1, 2) and (r, c) != (cr, cc)]
        outer_cells = in_row + in_col
        if not outer_cells: continue
        outer_colors = set(int(a[r, c]) for (r, c) in outer_cells)
        # Disambiguation: when row+col cells contain BOTH colors, take majority.
        if len(outer_colors) > 1:
            counts = {col: 0 for col in outer_colors}
            for (r, c) in outer_cells: counts[int(a[r, c])] += 1
            outer_color = max(counts, key=counts.get)
            outer_cells = [(r, c) for (r, c) in outer_cells if int(a[r, c]) == outer_color]
        else:
            outer_color = list(outer_colors)[0]
        for (r, c) in outer_cells:
            if r == cr and c < cc and cc - 1 >= 0:  out[cr, cc - 1] = outer_color
            elif r == cr and c > cc and cc + 1 < W: out[cr, cc + 1] = outer_color
            elif c == cc and r < cr and cr - 1 >= 0: out[cr - 1, cc] = outer_color
            elif c == cc and r > cr and cr + 1 < H: out[cr + 1, cc] = outer_color
    return out
```

## The 10 ambiguous failures

Each fail involves a single non-center cell at (r, c) such that:
- `r` matches one center's row (e.g., 1-center at (r, cc1))
- `c` matches the OTHER center's col (e.g., 2-center at (cr2, c))

Example pair 21: `(11, 5)` is in row 11 (== 1-center row) and col 5 (== 2-center col).
The cell's color is 7. Expected output: this cell is 1-center's LEFT-direction
trigger; 2-center stays alone (no 3 cells exist anywhere in input).

The "majority outer color" rule incorrectly assigns it to 2-center because col 5
contains only this one cell, making 7 the majority for that center.

### Tested disambiguation hypotheses (none cracked the remaining 10 yet)

- ❌ **Closer center wins** (Manhattan distance): drops accuracy to 115/266 — too aggressive.
- ❌ **Row-preference over col-preference**: 239/266 — wrong direction for half the cases.
- 🟡 **Pair the cell with the center whose other-color cells are ABSENT from input** if such a center exists: should fix pair 21 specifically (2-center has no other 3-cells → never get an outer). Plausible but untested. **Beta to try.**

### Plausible 100% rules to try (Beta)

1. **Cell goes to the center whose outer-color set already has multiple unambiguous members.** If 1-center has another 7-cell at non-intersection position, prefer 1-center. If both centers are "established" via unambiguous cells, fall back to closer-Manhattan or some other tiebreak.
2. **Per-pair canonical pairing 1↔7, 2↔3 unless data conflicts.** Check input scattered cells: if all 3-cells align with 2-center and all 7-cells align with 1-center (per the canonical), use that. If a cell breaks the pattern, treat it as "noise" — drop or assign by closer center.
3. **The center with FEWER unambiguous outer cells gets nothing if all its candidates are ambiguous.** Pair 21 fits this: 2-center has zero unambiguous 3-cells; treat as "no outer" and drop the ambiguous (11, 5).

The third hypothesis is the most likely to be correct based on pair 21's behavior.

## Detector preconditions

For each pair:
1. Exactly one cell of color 1 and one cell of color 2.
2. All other non-zero cells have colors in {3, 7}.
3. Verify the predict_task270 rule produces output exactly (or with the
   disambiguation hypothesis Beta selects).

## ONNX implementation sketch (opset 10)

Cost ~50-100k for the 100% rule.

### Step-by-step

1. **Find center positions.**
   - `ch1 = Slice(input, axis=1, [1:2])` → (1, 1, 30, 30).
   - `ch2 = Slice(input, axis=1, [2:3])` → (1, 1, 30, 30).
   - `ArgMax` row-weighted then col-weighted to get `cr1, cc1, cr2, cc2` — same trick as task029 / task174 bbox detection (Mul by ascending-row template, ArgMax).

2. **Build "row of 1-center" and "col of 1-center" masks.**
   - `row1_mask[r, c] = 1 if r == cr1 else 0` (1, 1, 30, 30) — derive from `cr1` via Range/Equal.
   - `col1_mask[r, c] = 1 if c == cc1 else 0` similar.
   - Same for center 2.

3. **For each direction (UP, DOWN, LEFT, RIGHT) of each center:**
   - `up1_mask[r, c] = (c == cc1) AND (r < cr1)` — derive from col1_mask + Range comparison.
   - `down1_mask`, `left1_mask`, `right1_mask` similar.
   - Same for center 2.

4. **Per-direction outer-color presence.**
   - For each direction, compute `Sum(input_3_mask * direction_mask)` and
     `Sum(input_7_mask * direction_mask)`. (Where input_3_mask = ch3 layer of input.)
   - If sum > 0, that direction "has cells of that color."

5. **Outer color selection per center.**
   - `count_3_around_center1 = Sum over 4 dirs of (input_3_mask * dir_mask)`.
   - `count_7_around_center1 = Sum over 4 dirs of (input_7_mask * dir_mask)`.
   - `is_outer_3_for_center1 = Greater(count_3_around_center1, count_7_around_center1)`.
   - Same for center 2 (mutually exclusive).

6. **Draw cross cells.**
   - For each direction-mask of each center, compute "should I draw outer color here?":
     - For 1-center, direction d: `draw_mask = (sum_outer_color_1_in_dir_d > 0) AND adjacency_d_to_center1`.
     - Output cell at center1 + d direction = `outer_color_1` if draw_mask else 0.

7. **Output assembly.**
   - ch1_out = ch1 mask (the 1-center).
   - ch2_out = ch2 mask (the 2-center).
   - ch3_out = (cross_3_around_center1 if is_outer_3_for_center1 else 0) + (cross_3_around_center2 if is_outer_3_for_center2 else 0).
   - ch7_out = similar for color 7.
   - Other channels = 0 (input has no other colors).
   - Concat → (1, 10, 30, 30).

### Cost estimate

- Bbox/center detection: ~5k MACs.
- Row/col masks: cheap.
- Direction masks: cheap.
- Per-direction sums: ~10k MACs.
- Output assembly: cheap.
- Memory: ~5k bytes intermediate.
- **Total: ~50-100k → ~14 pts.**

## Cross-references

- task029 (marker rectangle interior) for ArgMax-of-asc-times-row-mask bbox trick.
- task389 (color-swap-5-stays) for runtime-color detection via ReduceMax + Mul.
- The "row/col-aligned cell" pattern may also apply to task320 (vertical stacks).

## Recommendation for Alpha

Implement the 266/266 refined color-partition rule above. The older Option A
(96.2%) detector failed verification on 10/266 ARC-GEN pairs and is retained
only as rejected evidence.

---

*Filed: 2026-05-05 by 20-hr Beta-backlog-fill session. Superseded 2026-05-08:
the refined color-partition rule verifies 266/266; the older majority rule was
256/266 and remains rejected evidence.*

---

## Opus 2026-05-05 refinement (conf=mid, novelty=refined-recipe)

**Refined hypothesis**:

Refined rule: 'Pair each center with its canonical outer color via global cell-set partitioning, not per-cell majority.' Procedure: (a) Collect all non-center scattered cells (colors in {3,7}). (b) For each color k in {3,7}, count how many of its cells lie on row(c1) or col(c1) vs row(c2) or col(c2); a cell at the row/col intersection of both centers counts as ambiguous and is excluded from the tally. (c) Assign color k to whichever center has MORE unambiguous aligned cells of color k. The other color goes to the other center. (d) Once each center has a fixed outer color, draw the cross only in directions where an unambiguous aligned cell of that center's outer color exists. Ambiguous intersection cells contribute ONLY if they match the already-assigned outer color of the center whose row/col they lie on AND the other center has no other source for them. This subsumes failed hypothesis #3 (center with zero unambiguous outer cells gets nothing). For train pair 0: 1-center at (10,11), 2-center at (4,3); 7-cells at (6,11),(10,5),(10,14),(14,11) — 3 align with 1-center row/col, 0 with 2-center → 1↔7. 3-cells at (0,3),(4,0),(4,8),(9,3) — 4 align with 2-center, 0 with 1-center → 2↔3. Train pair 1: 7-cells align with 1-center, 3-cells with 2-center (canonical). Train pair 2: 7-cells (6,1),(6,10),(14,10) — (6,10) ambiguous (1-center row AND 2-center col? actually 1 at (6,10), 2 at (11,6); (6,1) is 1-center row, (14,10) is 1-center col → 1↔7). 3-cells (0,6),(11,14),(14,6) → (11,14) is 2-center row, (14,6) is 2-center col → 2↔3.

**Verification plan**:

Step 1: For each of 266 train+arc-gen pairs, locate c1 (color-1 cell) and c2 (color-2 cell). Step 2: Partition scattered {3,7} cells into four buckets per color: row(c1)-aligned, col(c1)-aligned, row(c2)-aligned, col(c2)-aligned (a cell may appear in two buckets if it sits at an intersection). Step 3: For each color k in {3,7}, compute n1(k) = count of UNAMBIGUOUS cells of color k aligned to c1 (in row(c1) or col(c1) but NOT also on row(c2)/col(c2) of the other center), and n2(k) similarly for c2. Step 4: Assign outer1 = argmax_k n1(k), outer2 = the other color (verify outer1 != outer2; if tie, fallback to canonical 1↔7,2↔3). Step 5: Draw cross arms: for center ci with outer color oi, set out[ci+d] = oi iff there exists a scattered cell of color oi in direction d along ci's row or col (UP/DOWN: same col, r<ci or r>ci; LEFT/RIGHT: same row). Step 6: Compare against expected output; expect 266/266. If <266, log per-pair failure mode (which center got wrong color, or which arm got drawn/missed). Step 7: Cross-check on the 10 originally-failing pair indices to confirm the new rule fixes them.

**ONNX path**:

Opset-10 implementation: (1) Slice ch1, ch2, ch3, ch7 from input. (2) ArgMax-based center localization: for ch1, compute row_sum = ReduceSum(axis=2 collapsing W), then ArgMax over H gives cr1; similarly cc1 via ReduceSum axis 1 then ArgMax over W. Same for ch2. Cast ArgMax outputs INT64→FLOAT for arithmetic. (3) Build row/col equality masks via Range(0,30,1) Cast→FLOAT, then Equal-via-Sub-and-Less-trick: row_eq_cr1 = Less(Abs(Sub(range_h, cr1_broadcast)), 0.5). Note opset-10 Equal requires INT — use Less(Abs(diff),0.5) on FLOAT instead. (4) Compute alignment masks: aligned_to_c1 = Or(row_eq_cr1_2D, col_eq_cc1_2D); aligned_to_c2 similar; ambiguous = And(aligned_to_c1, aligned_to_c2). (5) For each color k∈{3,7}: unambig_k_to_c1 = ch_k * aligned_to_c1 * (1-ambiguous); count_k_c1 = ReduceSum. Same for c2. (6) outer1_is_7 = Greater(count_7_c1, count_3_c1) → cast to FLOAT scalar. outer1_color_template = outer1_is_7 * one_hot_7 + (1-outer1_is_7) * one_hot_3 (shape [1,10,1,1]). outer2 = the complement. (7) For each direction d∈{UP,DOWN,LEFT,RIGHT} of each center: build arm_position mask (1×1 cell offset from center) via Range comparisons; gate on 'does any cell of outer_i exist in that direction along the row/col?' via ReduceSum over the half-row/half-col of ch_outer_i, then Greater(0). Multiply arm_position by gate by outer_color_template, accumulate into output. (8) Add center channels (ch1, ch2 reduced to their single positions). Walls hit: Equal-on-FLOAT not in opset-10 (use Less-of-Abs workaround); GatherElements with runtime indices not available (avoided by using mask-multiplication instead of indexed gather); ArgMax→FLOAT cast standard. No Loop/Scan needed since H=W=15 and all reductions are static-shape.

**Cost estimate**: 180000 bytes

**Source**: archived 2026-05-05 Opus hard-task synthesis.
