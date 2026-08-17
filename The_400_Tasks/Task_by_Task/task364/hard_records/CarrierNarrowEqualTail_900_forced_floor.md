# task364 — DEAD ROUTE: CarrierNarrowEqualTail (~900 forced-floor false-floor attack)

**Batch:** bo_ng_cld_worker3 (Claude lane) 2026-07-11
**Live bar at run time:** **12,319** (manifest, owner **Sho**, updated 2026-07-09T01:09:17Z),
sha256 `224d157edef9bbb1f98e83345ca1ed6a44610c97d0b18edbd0e58b45a3119808`,
`Team_Best_Onnx/task364.onnx`. profile_real_cost `--arcgen 266`: **static==profile==12,319**
(memory 12,214 + params 105, fully STATIC). Graph name `task364_qcode_h25_v25_p7`.

**Prompt target:** ~900 (EV +2.62). Direction: "collapse 10-channel/F32 spatial carrier to
U8/color-index; one spatial predicate plane + Equal tail; delete charged carrier/canvas/template."

## WHY DEAD — three independent walls

**1. The charged carrier is ALREADY GONE (workflow-update DROP trigger).**
Sho's live 12,319 graph already performed the carrier-narrow the Direction asks for. The old
4-channel `W4` U8[1,4,20,22]=1,760 detector bank (in the retired 14,162 graph) no longer exists;
it is replaced by a **1-channel** `nbr_center_code` QLinearConv(3×3) + a **2-channel** `pair2`
[2,1,5,5] that immediately `ReduceMax`es to 1 channel. Largest live tensors:
`mask_f` F32[1,1,20,22]=1,760 (input read), `task364_scalar30` U8[1,1,30,30]=900 (Equal-tail
canvas), `pair2_score_u8` U8[1,2,20,22]=880. **No 10-channel F32 carrier exists.**

**2. PRICE WALL by construction: 900 < 1,760.**
Input is F32 [1,10,30,30]; active grid 20×22. Any per-cell read of the F32 input materializes an
F32 tensor ≥ H·W·4 = **1,760 B** (dtype-locked; Slice/Pad of F32 stays F32 — techniques #88
input-bridge floor). The ~900 target is *below the cost of reading the input once*. Cheaper reads
don't exist: Cast-first `[1,10,30,30]→U8` = 9,000 B (worse); ReduceMax-first `[1,1,30,30]` = 3,600 B
(worse). No algorithm reaches 900.

**3. FORCED FLOOD — the recolor is component-global, not per-cell-local.**
Fast probe on **3000 fresh ARC-GEN pairs** (seed_offset 900000; numpy_reference 3000/3000 = 100%
vs generator; output colors {0,1,2,6} = bg/L/H/U). Best-possible *local, no-flood* per-cell
classifier accuracy by kernel reach:

| receptive field | best per-cell acc | ambiguous cells |
|---|---|---|
| R=1 (3×3)  | 56.9%  | 80.7% |
| R=2 (5×5, = live graph's max kernel) | **77.9%** | 53.9% |
| R=3 (7×7)  | 95.3%  | 14.7% |
| R=4 (9×9)  | 100%   | 0% |

A local carrier at the live graph's kernel reach (≤5×5) tops out at **77.9%** → the "one predicate
plane + Equal tail" formulation **falsifies**. Only a 9×9 receptive field disambiguates — that is
"see the entire ≤5×5 sprite" (component-global), and a 9×9 multi-class Conv+LUT carrier costs MORE
than the current seed + 5-round masked flood (4,400 B), not less. The flood is both forced and the
cheap way to propagate a sparse feature label across a component.

## Formulations refuted (3, independent — all hit walls 2 & 3)
- **A. local color-index plane + Equal tail** (prompt fast-probe): falsifies (≤77.9% at 5×5) and
  priced (plane + 1,760 input read > 900).
- **B. big-receptive-field single-Conv classifier** (9×9 sees the whole sprite): 100% acc but a
  9×9 multi-class conv carrier + argmax/LUT tail costs more than the flood, and still pays the
  1,760 input read. Priced far above 900.
- **C. per-component global reduction** (bbox-edge / junction-count = numpy_reference's own
  classifier) + scatter label back: needs connected-component labeling (a propagation) + scatter
  (a propagation) ≥ current flood cost, plus the 1,760 read. Priced above 900.

## Remaining hypothesis (epsilon-shave, NOT the structural win; unverified, correctness-risky)
`pair2_score_u8` U8[1,2,20,22]=880 is the ONLY multi-channel tensor. IF a single corner-orientation
channel sufficed (it likely does not — the 2 channels give rotation coverage for U-vs-L before the
ReduceMax), a 2→1 collapse would save 440 B → 11,879 (a 3.6% shave). This is a rung-epsilon shave,
explicitly NOT the ~900 false-floor target, and risks breaking U/L discrimination; it would need a
fresh-ARC-GEN truth-full gate to even test. Deferred — do not treat as a floor route.

## Verdict
**WALL / DROP.** 12,319 is at/near the algorithmic floor for the forced idea-class
(F32 input read 1,760 + narrowed feature seed + BFS-diameter masked flood 4,400 + minimal
broadcast-Equal tail 900). The ~900 forced-floor target is infeasible (below the input-read floor;
the recolor is component-global). The prompt's ~900 derives from the stale Euler/hole-count premise
— L/U/H are all line shapes with 0 enclosed holes, so Euler does not discriminate them.

Probe script: `prototypes/probe_local_carrier.py` (re-run: `... probe_local_carrier.py 3000`).
