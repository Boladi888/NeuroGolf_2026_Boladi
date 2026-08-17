# DEAD ROUTE — task002 "ColorIndexCollapse" false-floor attack (2026-07-10, bo_ng_cld_worker1)

**Live bar:** cost **9413**, sha256 `8c8340bca13986bd90f9831574d54e3e930935eacfec70fa9f830e455179029a`
(manifest authoritative via `ng_current_bar.py --task 2`; owner Boladi; Staged_Boladi/_shipped
`confirmed_public` +0.014 LB). Older in-stack LIVE = 9545 (sha `0c7d82de886f…`). Both are the
same enclosed-fill flood algorithm; the −132 is a byte-equal dedupe. LB fresh 7557.27 (2026-07-10T09:55Z).

**Prompt hypothesis (falsified):** SWEEP class, floor_est ~900 (gap 10.46×). Direction:
"color-index collapse sweep — perform spatial logic on one index channel, use Gather/Equal only at
the boundary; delete charged carrier/canvas/template." Fast probe: "replace the per-color path with
an ArgMax/index-plane prototype."

**Why the premise is wrong:** task002 is NOT a color map. The numpy reference
(`prototypes/task002_numpy_reference.py`) and the live graph both compute a **4-connected flood fill**:
a 0-cell → 4 iff it has NO path through 0-cells to the 20×20 border. Input is binary (0/3); output
adds one color (4). There is **no per-color path** to collapse and no ArgMax/index plane to gather —
the "Color Mapping" family tag mis-describes a topological connectivity task.

**Charged component is still fully paid, but it IS the connectivity computation** (not a deletable
carrier). Live-graph anatomy (`largest_intermediate.py`): Pad BOOL[1,1,30,30]=900 (flood canvas) +
Concat BOOL[1,1,20,20]=400 (output mask) + ~70 bit-packed uint32[1,1,20,1]@80B flood/ray propagation
masks. Deleting these deletes the function (Don't-do log: drop-repair → visible-fail; fewer-iters →
§14e corpus divergence).

**One crack run (probe, no ONNX built — kill rule fired at the premise):**
`prototypes/fast_probe_colorindexcollapse_2026-07-10.py`, ground-truth = task's numpy ref.
- **F1 — per-cell / separable index collapse (the literal Direction):** two grids with identical color
  multisets; input-value 0 maps to output 4 (enclosed center) in one and 0 (border cell) in the other.
  A per-cell/separable Gather-Equal must give them the same answer ⇒ **cannot decide enclosure. DEAD.**
- **F2 — bounded-depth propagation (the most ~900 B buys):** on a width-1 serpentine (true BFS depth
  to deepest exterior cell = **170** on 20×20), a K-step flood false-encloses 165/158/150/130 cells at
  K=5/12/20/40. Enclosure needs propagation depth that scales with grid size ⇒ any fixed small step
  count **falsifies on fresh ARC-GEN-shaped input. DEAD.**

- **F3 — global algebraic / Conv reachability (cost+legality kill, no build):** transitive-closure via
  matrix-power needs a 400×400 F32 operator (~640 KB); Conv-dilation reachability is ORT-illegal on
  integer dtypes (AveragePool rejects UINT8; Conv F16-out on F32 impossible) and in F32 costs
  ~1600 B/mask ([1,1,20,20]) × ~40 iters ≫ 9413. So the **bit-packed uint32 log-shift flood (the
  current LIVE graph) is already the cost-minimal ORT-legal family** — and it is audited to 9413. DEAD.

**Forced-floor argument vs ~900:** connectivity is neither per-cell (F1) nor fixed-shallow (F2) nor
cheaper under any legal global operator (F3); the
cheapest ORT-legal representation is bit-packed log-shift propagation — uint16 has no ORT kernel, so
each of the 20 packed rows is a uint32[1,1,20,1]@80 B, replicated across the iters×directions needed
to reach ~170. The propagation state alone is thousands of bytes; the 9413 floor proof audited every
tensor as load-bearing and is Kaggle byte-equal-confirmed. **~900 is a family-complexity heuristic
(9413/10.46), unmoored from the flood structure — infeasible.**

**Exact failed gate:** none run past the probe (kill rule: premise falsified before build; truth/ab/
battery/full-profile on a non-existent candidate = pure burn, per gate hygiene + "one crack only").

**Verdict: DROP / WALL.** Effective best stays 9413. No color-index-collapse route exists.

**One remaining hypothesis (only lever left, low odds, NOT this angle):** any future gain must be a
further **byte-equal** micro-fold of the existing bit-packed flood graph (e.g. another
redundant-initializer or removable-remask pass like the 9545→9413 dedupe), found by static graph
audit — never a "collapse the color path" reformulation. Do not re-open ColorIndexCollapse.
