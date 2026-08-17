# task339 hard_record — perfect-25 ≤1 rumor DISPROVEN + RoiAlign 78 is the structural floor

- **Live bar:** cost **78**, sha256 `600ce7fa7698651185e018264fc0beb42693c287c088f6d34e50079b575caca5`
  (source: manifest / Team_Best, owner Vamshi, 2026-07-02). `ng_current_bar --task 339` authoritative.
- **Batch:** bo_ng_cld_worker1 (Claude), 2026-07-07. `worker_attempts_before_review`=1 → likely only pass.

## The task rule (derived, verified 261/261 arc-gen)
Input: ARC 3×3 grid with a single non-zero color `c` appearing `k` times (k∈[1,9]), rest 0.
Output: a **1×k** row filled with `c`. Verified on all 261 arc-gen examples:
`nz_count == output_width` for 261/261; `multi-color inputs = 0`; out shapes all `(1,k)`.
→ Output is **data-dependent in BOTH width `k` AND color `c`**. It is a count→recolor→render
task, NOT a shape-only permutation.

## Route 1 — perfect-25 zero-cost ≤1 (Identity/Transpose/flip/rotate) — DEAD
The prompt ("ZeroopsweepCeilingAudit", floor_est ~1) is a **rumor**. Disproven by data, no gate needed:
a no-payload direct-output op (Identity/Transpose/flip/rotate) cannot produce a `k`-wide, `c`-colored
bar whose width and color both depend on the input. The forced-floor ≤1 argument is for whole-grid
permutation tasks; **this task is not one**. Kill rule satisfied: no ≤1 candidate can be visible-correct.
Do NOT re-open the zero-param family here.

## Route 2 — beat 78 with a cheaper real algorithm — WALL (78 is the RoiAlign-family floor)
Live graph `task339_selected_direct_reduction_roi` (5 nodes): Einsum `bchw,cyx->bcyx` (mask zeroes
ch0) → foreground_counts `[1,10,1,1]` (=k at active channel c) → ReduceMax→count `k` → Reciprocal→`1/k`
→ Concat rois `[0,0,1/k,1]` → RoiAlign renders the k-wide colored bar **directly to the graph output**.

**Why 78 is forced (component-by-component):**
- `foreground_counts [1,10,1,1]` F32 = **40 B** — RoiAlign preserves channel count, so the feature map
  MUST carry all 10 channels to encode color identity spatially. Cannot drop to F16 (Einsum output
  follows F32 input; a Cast leaves the F32 tensor still counted → strictly worse). **40 B forced.**
- `count`(4) + `inverse_count`(4) — computing the ROI width `1/k` needs both. **8 B forced.**
- `rois [1,4]` F32 = **16 B** — RoiAlign's ROI input format is fixed 4-tuple. **16 B forced.**
- memory floor = 40+4+4+16 = **64 B**.
- params: `mask`(10, forced to zero the background channel 0 before the max) + ROI coords x1,y1(2) +
  y2(1) + batch_indices(1) = **14** forced.
- **Total floor = 64 + 14 = 78 = LIVE. The live graph IS the RoiAlign-family floor.**

**Why the RoiAlign family is uniquely cheapest (other formulations rejected analytically):**
- The winning trick: the only large tensor `[1,10,30,30]` is the **graph OUTPUT** (excluded from cost);
  RoiAlign renders it from tiny inputs. Any approach that materializes the k-wide bar as an
  **intermediate** pays for it.
- **Expand/Tile to `[1,10,1,k]` then Pad to 30×30:** the Expand output is a runtime-dynamic tensor;
  grader charges `max(static,runtime)` → max k=9 → `[1,10,1,9]` = **360 B** intermediate alone. LOSE.
- **Ramp `[30]` iota + Less(ramp,k) mask + Mul + Pad:** the iota initializer is 30 params (vs 14 total
  live) or a Range machine that still materializes a `[30]` intermediate. LOSE on params/memory.
- **Resize / ConvTranspose from `[1,10,1,1]`:** fills ALL columns (no partial-`k` crop); getting exactly
  `k` positive columns requires the ROI-crop behaviour, i.e. RoiAlign. No cheaper op renders a
  data-dependent-width partial bar directly to output.

## Verdict
- **WALL at 78.** No candidate strictly below 78 exists in the RoiAlign family (it is the floor) and
  every non-RoiAlign formulation materializes a k-wide intermediate ≥ the live cost.
- perfect-25 ≤1: infeasible (data-dependent width+color).
- Nothing staged. Do not re-explore the ≤1 family or the Expand/ramp/Resize alternatives.
- One remaining (low-confidence) hypothesis for a FUTURE batch: a different grader-accepted output
  encoding that avoids the 10-channel feature (e.g. if the grader accepts a single-channel argmax-index
  map, the feature could drop below 10 channels and shave the 40 B). Requires confirming grader
  compares argmax-grids, not raw one-hot — untested; do not assume.
