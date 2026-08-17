# task158 floor + dead routes — after v128 win (22,550), the remaining structure is forced

**Prior LIVE:** Sho 23,201 (manifest, sha `cd08cc20468a…`; actual ONNX score_v2 23,194).
**This batch (bo_ng_cld_worker2, 2026-07-09):** staged **v128 = 22,550** (−651), byte-equal to LIVE.
See `staged_candidates/task158_where_gated_locator_cost22550_sha5FCB99B3/` + `prototypes/task158_v128_worker2_notes.md`.
Ambiguous class → ship gate = byte-equal-to-LIVE on the AB corpus.

## ⚠ The local task158.md docs describe a SUPERSEDED graph
task158.md's "current best v125 23,460" (F32 output, dense `s1/s2/s3` valid-score planes) is NOT live.
LIVE was Sho 23,201, a different formulation: BOOL one-hot output (free `Equal` tail), 26×25 crop, one
`DET=QLinearConv[3,2,11,11]` + `CP=ConvInteger[1,3,9,9]` gather-render. The prompt's `floor_est ~11000`
("three dense valid-score planes are the family wall") is premised on the stale v125 family; those
planes do not exist in LIVE. Decode the LIVE ONNX first, always.

## LIVE graph (Sho, 62 nodes) — decoded, and where v128 cut
Input one-hot `[1,10,30,30]` → free BOOL one-hot output. All planes cropped to 26×25 (650 B).
- Input read: `cidc=Conv(input,[0..9])` F32 2,600 → `C8c=Cast` U8 650.  (F32 read tax — forced)
- **Locator:** `eqbg,Z,gtZ,s11,Z2,nbU,ZM,Z2M,aZ,gW,g16,nbsum,score`. score=`nbsum`(3×3 non-bg count)
  + `g16`(=16 iff ≥2 non-bg colors) → `ReduceMax`+`ArgMax` → legend centre.
  **← v128 dropped `g16` + `c16`:** score feeds only argmax, and the +16 is a rank-only bonus, so
  `score=Where(gW,nbsum,0)` is argmax-identical (0/200,266). −650 B mem −1 param.
- Marker masks: `XYeq=Equal` BOOL 1,300 → `XYu=Cast` U8 1,300 (forced 2-channel).
- Detection: `DET` U8 [1,3,24,23] 1,656 (3 scales, all load-bearing).
- Placement: `CP=ConvInteger` INT32 [1,1,26,25] 2,600 (per-cell stamp code 0..76).
- Render: `gat=Gather(TF[77],CP)` 650, `pos=CP>0` 650, `OC=Where` 650, `OCpad=Pad→30×30` 900, `Equal→output` free.

## Three from-scratch routes — one WIN, two DEAD

### Route 1 — cheaper locator.  PARTIAL WIN (v128, −651).
`score` feeds only `ReduceMax→ArgMax`, so any locator with the same (rc,cc) on all 200k → byte-equal
output. Bench (`prototypes/_locator_bench.py`, numpy replica byte-validated vs ORT on 5k — `_locator_lab.py`):
- `nbsum_only`: **48,818/200,000 mislocate** → multicolor bonus is load-bearing, cannot drop.
- `masked`(nbsum·mc) & `rangepool`(10-complement): 0/200,000 — same argmax. `masked` needs
  `Mul(U8,U8)` (ORT-banned) BUT the identical values are legal via `Where(gW,nbsum,0)` → **that is v128**.
The 2-maxpool max+min-over-non-bg multicolor test itself is tensor-minimal (every one of its planes is
consumed+forced). Only the g16 *magnitude* was slack. No further locator tensor is removable.

### Route 2 — sparse / static-kernel paint-mask render (drop CP INT32).  DEAD.
The legend paint-cell PATTERN is **data-dependent**: 18 distinct patterns over 4,000 corpus legends
(plus/X/L/diagonal). A static conv cannot spread detections to data-dependent paint offsets → the render
must look colors up at runtime via `Gather(TF,CP)`, and Gather indices are INT32/64 only → **CP INT32
2,600 forced** (#132 conv/gather floor). "Paint only paint-cells" ≡ Sho's "paint whole stamp" (verified:
painted cells take 1 distinct color/grid over 7,996 pairs) but that does not remove the data-dependence.
A dynamic-count coordinate render (2–4 placements/input) is a dynamic-shape build that reprices UP (#109).

### Route 3 — byte-equal fold / dtype narrow.  DEAD (beyond v128).
`prototypes/_redundancy_hunt.py` over 1,499 pairs, all intermediates exposed: **0 always-equal same-shape
tensor pairs**; only `cidc`(F32 0..9) and `CP`(INT32 0..76) are dtype-slack, both **dtype-forced**
(`Conv` float-only; `ConvInteger`→`Gather` int-only). No redundant plane; no legal narrowing.

## New floor argument (why ~22,550 is the floor for this formulation)
After v128, memory 21,364 = 2,600 (cidc F32) + 650 (C8c) + ~7,800 (locator, now 12 tensors) + 2,600 (2
marker masks) + 1,656 (DET) + 2,600 (CP INT32) + 1,950 (render) + 900 (OCpad) + ~608 (misc). Every tensor
is consumed and its dtype/shape forced by an ORT op constraint or by data-dependence. Output is free;
all planes cropped to 26×25. No remaining slack found.

## One noted-but-not-a-cost-win hypothesis
LIVE and v128 both CRASH on ~74/200k pairs (`Gather 'gat' idx≥77 OOB`) — the official ab/truth gates
ABORT on these (confirmed LIVE crashes identically). Extending the 77-entry template to ≥122 would make
the graph MORE-correct (dual-truth) but ADDS params and never removes memory → prices ≥ current, not
below. Dead for cost; possible separate correctness improvement only.

**→ FIXED 2026-07-10 (worker-7, v129):** cause = overlapping detection code-stamps ADD inside the CP
ConvInteger 9x9 window (observed idx=121). Fix = `CPc=Min(CP,76)` feeding only the Gather (CP≥0 provable:
all-nonneg u8 operands, zero-points 0). +2,601 forced (CPc INT32 plane + c76 param); byte-equal to LIVE
on all non-crashing boards (offline AB 2,266 pairs div=0). NOTE: table extension to a PROVABLE bound is
dead — provable max CP = 339,150 (DET only provably ≤255). Staged
`staged_candidates/task158_v129_cp_clamp_crashfix_cost25802_shaA88E0E7A/`. The same 1-node clamp composes
with v128 (~25,151 combined) if the captain wants crash-free at the lower rung.

## ⚠ 2026-07-10 run-2 CORRECTIONS (v130 = crash-free 23,187, BELOW the 23,201 bar)
Two claims above are now disproven; the "~22,550 floor" was NOT the floor:
1. **"Mul(U8,U8) ORT-banned" is FALSE at LIVE's opset (17).** That was an opset-13-era fact; at opset
   14+ `Mul`/`Add`/`Sub` run on UINT8 (library ort_reality §41 correction). Verified in this ORT build.
2. **The 2-maxpool multicolor test is NOT tensor-minimal.** `multicolor ⟺ window_sum(Z) != ZM*nbsum`
   (`SUMc=QLinearConv(Z,ones9)`, `MulZN=Mul(ZM,nbsum)`, `eqs=Equal`, `score=Where(eqs,0,nbsum)`; values
   ≤81, no u8 overflow) is byte-identical to the gW-gated score and deletes gtZ/s11/Z2/Z2M/aZ/gW while
   adding only 3 planes: **−1,950 B**. Plus Concat axis 0→1 (−6) and scalar corner indices i0/i2/i6/i8
   + cg-Reshape delete (−8; the cE0 ReduceL1 axes must be retargeted [3]→[0]).
Net: 22,550 − 1,964 + 2,601(clamp) = **23,187 crash-free, static==profile, −14 vs bar** — a real point
gain AND the crash fix in one graph. Gates: visible 266/266; truth n2000 no-error 1999/2000 (miss =
LIVE's crash board); offline crash-aware AB 10,266 pairs div=0/fixed=6/cand_crash=0; safety WARN-parity.
Staged `staged_candidates/task158_v130_sumtest_clamp_cost23187_sha84BF2DB2/`.
Also probed this run (dead): color-weights ConvInteger render to delete the plane-Gather entirely
(`WC=Gather(TF,Wc2)` then `ConvInteger(DET,WC)`) — DEAD because DET's VALUE 1..4 is the rotation code
(kernels: X-slots 50, Y-slots 51..54, per-scale bias −100/−404/−909) and conv contributions are
`d*W[offset]`, not `TF[d*w]`; binarizing DET loses the rotation, 12-channel DET costs +4,968.
Probe: `_probe_det_cp_w7.py`, `_probe_det_perchan_w7.py` (also: overlap-without-crash = 0/8,266 boards,
so every overlap board is a LIVE-crash board — no divergence class exists for sum-preserving rewrites).

## VERDICT: v130 23,187 staged (crash-free, byte-equal-where-LIVE-runs, −14 vs bar). Routes 2 & 3 WALL.
Breadcrumbs: `_locator_lab.py`, `_locator_bench.py`, `_redundancy_hunt.py`, `_offline_ab.py`,
`build_task158_v128_where_gated_score.py`, `build_task158_v130_sumtest_clamp.py`.
