# task286 hard record — Einsum-transfer-matrix flood chain floor

- **Live bar at attempt:** manifest/team-best **18672**, owner Sho, sha `20036e365eb9e5a5e2a39cc1b4db599e1f3925d4faee1d35ebbb64392316d4b9`, updated 2026-07-10 (via `tools/ng_current_bar.py --task 286`).
- **New reality (supersedes the 2026-07-02 WALL note and Vamshi 21558):** Sho's 18672 is NOT a bit-packed flood. It is a **24-node Einsum graph**: the maze reachability flood is expressed as **3 chained separable-dilation Einsum transfer matrices** (each operand list = `[reach, v_w0, v_w0, open, ...]` × 25 steps, equation `aA,ab,AB,bB,...->zZ`). Each Einsum charges ONLY its `[25,25]` output tensor — the 75 dilation steps are free. This bypasses the whole "flood forced-component 16,248 B core" floor from worker2's WALL.

## What is dead (do NOT re-explore)
1. **Reducing the 3 Einsum chains to 2 (50 steps).** MEASURED corpus max 8-conn convergence depth = **59** (`prototypes/measure_depth.py`); one input needs 59 steps. 2 einsums = 50 steps < 59 ⇒ fails even *locally* (not just hidden). The einsum alphabet caps 25 steps/axis (26 lowercase rows + 26 uppercase cols), so depth 59 ⇒ ⌈59/25⌉ = **3 einsums forced**. Reducing 75→60 steps saves nothing (still 3 einsums, same 3 output tensors).
2. **Removing the Min clamp (v_t11).** TESTED (`build_v1_nomin.py`): feeding unclamped v_t10 (which holds `inf`) into the 3rd einsum produces `nan` (inf spreads then masks) → 262/265 mismatch. Empirically: an einsum tolerates ONE inf-bearing input in the chain (t9→t10 OK) but a SECOND (t10→t12) nans. So `[E25, E25, Min, E25]` is the forced minimum for 75 steps = 4 chain tensors (3 einsum outputs + 1 Min).
3. **4-conn flood.** corpus max 4-conn depth = **77** ⇒ needs 4 einsums (100 steps) = MORE tensors than Sho's 8-conn 3 einsums. Sho's separable (8-conn) dilation is itself a cost optimization (depth 77→59). Rule is truth-100% at 8-conn (2000/2000), so 8-conn is correct here.
4. **Radius-2 dilation (v_w0 band [1,1,1,1,1]) to halve depth.** Jumps over 1-cell walls (mask only checks destination, not path) ⇒ wrong for mazes.
5. **QLinearConv / bucket-detector (prompt hint).** Rule is a genuine transitive-closure maze flood, NOT a threshold/bucket detector. Quantizing the F32 one-hot input for QLinearConv costs a `[1,10,30,30]` U8 = 9000 B intermediate. Dead.
6. **Narrowing the entry v_t0 (F32 [25,25], 2500 B).** Forced by the F32 competition input: casting input to F16 first = `[1,10,30,30]` F16 = 18000 B; Slice-first = `[1,10,25,25]` F32 = 25000 B (old don't-do). The einsum fuses channel-reduce + 30→25 crop into one 2500 B output — optimal.

## What WAS won (2026-07-12, bo_ng_cld_worker3)
**Mask-plane collapse via wall-negative remap** → staged `task286_wallneg_maskcollapse_cost17422` (byte-equal, −1250 B). Sho used 5 charged BOOL planes for the masks; remapping wall color-value 8→−1 lets `Greater(v,0)` be the open mask directly → 3 planes. See that manifest.

## One remaining hypothesis (unproven)
The 3 mask planes (open/bg/seed = 1875 B) and paint tail (v_t19n+v_t20 = 1250 B) each look at floor, but a from-scratch tail that fuses the reached-select + checkerboard-paint into fewer than 2 `[25,25]` U8 planes (e.g. an arithmetic composite) *might* save one 625 B plane — not found here (nested Where / arithmetic both needed ≥2 planes). Entry+flood are proven-forced; any future gain must come from the ~5.6 KB of BOOL/U8 mask+paint planes, not the 10 KB flood+entry core.
