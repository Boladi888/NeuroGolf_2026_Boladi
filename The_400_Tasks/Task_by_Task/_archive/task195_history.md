# task195 — superseded rung history

Pointer source: `Task_by_Task/task195.md` `## COMPRESSED HISTORY`. Current best is `task195_f16kron` cost 489 (full prose in task195.md). Below are the superseded rungs (full prose retained for provenance).

### SUPERSEDED: `task195_hybrid_slicekron.onnx` cost **525** (memory 297 + params 228)

**Worker-195, 2026-06-22.** The hybrid that breaks the (false) WALL. Two independent wins combined:
1. **Aditya's strided-slice anchor** (read from his actual 921 onnx — the construction the WALL audit "didn't have documented"): `Slice(input, ch=5:6, h=3:16:3, w=3:16:3) -> x5f [1,1,5,5]`. Cast F32->U8 (25 B). `ReduceMax` rows/cols -> `[5]`, `ArgMax` -> first strided lattice row/col `top`/`left`, `Add [0,1,2]`, two plain `Gather` -> `P [1,1,3,3]` U8. The gray lattice is 3-spaced and `r0∈[1,7]`,`c0∈[1,9]`, so the strided window `{3,6,9,12,15}` always contains it and ArgMax-of-first-strided-hit + offsets reads exactly P. Verified 200000/200000 incl. the r0=7,c0=9 corner (1073 cases). **This anchor head charges ~150 B vs the freeoutkron head's 576 B** — no F32 `[1,30]` marginals, no binary-clamp chain, no INT64 GatherND index grid, plain `Gather` (INT32 indices OK — not GatherND).
2. **Our free-output separable Einsum** (#109/#119), verbatim from freeoutkron, fed by that cheap P. Aditya's OWN graph does NOT have this — he still materializes `out6ch [1,6,9,9]` U8 = 486 B + `kron` 81 + `out5` 81 + `Where`/`Pad`. So the hybrid dissolves Aditya's 648 B output tail too.

Charged: `x5f` F32[1,1,5,5]=100 + `x5` U8=25 + tiny `[5]`/idx vectors + `P_u8`=9 + `Q_u8`[1,2,3,3]=18 + `Q` F32=72 = mem 297; params Hb 90 + Lb 90 + Ksel 20 + ones33_u8 9 + slice/offset inits 19 = 228. `output` free-output Einsum = 0 B. `profile==static==525`. Gates: visible 265/265, truth 200000/200000, profile 525 over 266 arcgen. SHA256 `d3189b02e75ea3943dbca2305b8c8f095b882b5704b979e1c501f9a6dfd6b51d`. Staged `staged_submits/task195_hybrid_slicekron_cost525/`.

**Next lever (if pushed further):** `x5f` F32[1,1,5,5]=100 B is the dominant charge, forced by F32 Slice dtype. `Q` F32[1,2,3,3]=72 B is the Einsum operand. Both are small; the head is near its floor for this formulation. (Realized 2026-06-24: `Q` shrinks to F16 = 36 B — see f16kron 489 in task195.md.)

### SUPERSEDED: `task195_freeoutkron.onnx` cost **1,011** (memory 760 + params 251)

The 810 B `out9_oh` patch was NOT forced. `kron(P,P)[r,c] = P[r//3,c//3]·P[r%3,c%3]` is a SEPARABLE self-product, so the entire `[1,10,30,30]` one-hot output BECOMES a single free-output Einsum (#109/#119) — the `output` tensor is never charged (scorer/profiler skip the tensor named `output`). One node:
```
output = Einsum('gtij,gtad,tk,ir,ar,jc,dc->gkrc', Q, Q, Ksel, Hb, Lb, Hb, Lb)   # FREE
```
- `Q = concat(ones(3,3), P)` → `[1,2,3,3]`; the term axis `t` carries window (t=0, Q=ones) and kron (t=1, Q=P). P appears TWICE (#119 self-reuse) so the self-product is in-kernel.
- `Ksel[2,10]`: term0 → channel 0 (e0); term1 → e5−e0 (kron paints ch5, and is subtracted back out of ch0 so ch0 = window − kron = NOT-kron inside the 9×9). This exactly reproduces the LIVE semantics verified 2026-06-18: ch5=kron, ch0=¬kron INSIDE the 9×9 only, both zero outside, all other channels zero.
- `Hb[3,30]` (=`r//3==i & r<9`) and `Lb[3,30]` (=`r%3==a & r<9`): `Hb[i,r]·Lb[a,r] = (r==3i+a)`. This factors the dense `[3,3,30]` placement basis into two `[3,30]` matrices REUSED on both the row and col axes (#118 basis-matrix-as-Einsum-factor). Bases = 180 params total (was 540 for the dense `[3,3,30]×2`).

Counted cost is now the cheap P-detection head (two `[1,30]` F32 Einsum marginals 240 B, GatherND index `[1,3,3,3]` INT64 216 B, the cast/argmax chain, the `[1,2,3,3]` Q stack) + the bases/selector params. **No 9×9 BOOL patch, no kron Reshape ladder, no Where/Equal/Pad tail.** Output is F32 {0,1}; the scorer + ng_harness both binarize via `>0`, so visible/truth/ab compare correctly (task295 free-output decode). `profile==static==1011` (#109: the Einsum contraction temp is uncharged at runtime — no reprice). −796 vs v3.

**Floor note (formulation-relative, #68).** The remaining 760 B memory is dominated by the GatherND index `[1,3,3,3]` (216 B) and the two `[1,30]` F32 marginals (240 B) — both part of the P-detection head, not the output. Eliminating GatherND by slicing channel 5 first births a `[1,1,30,30]` F32 plane (3,600 B) — strictly worse, so GatherND is the cheap read. The output itself is fully free; further wins must come from a cheaper P-detector, not the output formulation. Staged at `staged_submits/task195_freeoutkron_cost1011/`.

### BEAT-OR-WALL audit vs team-best 921 (Aditya) — worker-195, 2026-06-20 (**WALL REFUTED 2026-06-22** — see hybrid_slicekron 525 above)

> **2026-06-22 CORRECTION:** This "WALL" was an artifact of optimizing only WITHIN the freeoutkron head. Its 576 B anchor (two F32 `[1,30]` marginals + clamp/ArgMax + INT64 GatherND index) was never the floor — a **strided Slice** anchor (read from Aditya's own onnx) charges ~150 B. Lesson (#68 reinforced): floor proofs are formulation-relative; before declaring a WALL vs a known team-best, READ THE TEAM-BEST GRAPH — the cheaper primitive it uses may be directly transplantable. The original audit text below is retained for provenance but its WALL verdict is VOID.


Target was strictly < 921 (team-best Aditya, confirmed `Agenters/Orchestrator/runs/beta_p8b_claude_cli/team_best.csv` row `task195,Aditya,...,921`). Our best is freeoutkron **1011** (mem 760 + params 251); the output is already a free-output Einsum (0 B), so the entire 1011 is the P-detection head. Exhaustive per-tensor decomposition (every charged byte, profiled `static==profile==1011`):

| charged tensor | bytes | verdict |
|---|---:|---|
| `gather_idx` [1,3,3,3] INT64 | 216 | FORCED. The 9-cell read. INT64 mandatory (strict-shape rejects INT32 GatherND indices — Don't-do). 27 elts = 9 cells × [c,h,w] is minimal. |
| `row_sum`/`col_sum` F32 [1,30] | 240 | FORCED. Einsum-marginal output dtype = input dtype = F32; width 30 follows input H. Cannot narrow without materializing a plane. |
| `row_b`/`col_b` BOOL + `row_u8`/`col_u8` U8 [1,30] | 120 | NEEDED. Must clamp the per-row count to binary so ArgMax returns the FIRST active row, not the max-count row. ArgMax rejects BOOL (`ort_reality.md:62`) → U8 is the cheapest ArgMax-able dtype. 2 tensors/axis is minimal. |
| `Q` F32 [1,2,3,3] + `P4`/`P3` | 144 | `P4` (36 B) removable by emitting `P` as `[1,1,3,3]` directly from a `[1,1,3,3,3]` GatherND index (same 27 elts) → ~975. `Q`/`P` otherwise forced. |
| `offset3` INT64 [3] + `r0_i64`/`c0_i64` | 40 | NEEDED. |
| params (Hb 90 + Lb 90 + idx_base 27 + Ksel 20 + ch5_sel 10 + ones33 9 + sh_p3 4 + zero 1) | 251 | FORCED. Hb/Lb are the minimal #118 `[3,30]` factorization (already 180 vs 540 dense); param count is element-count so no dtype win; sparse inits rejected by scorer (core.md). |

**Forced-floor of the P-detector (this formulation):** read = 216 (GatherND, the cheapest read — proven below), anchor = 240 (two F32 marginals) + 120 (clamp→ArgMax) = 360. Reading the 9 lattice cells + locating the anchor costs ≥ 216 + 360 = 576 B memory + ~108 (Q/P) + 251 params ≈ **975** at the absolute tightest (only `P4` 36 B is removable). That is still **+54 above 921** — the formulation cannot reach the target.

**One-lever check the prompt named (fold the lattice READ into the output Einsum, #119):** REJECTED, strictly worse. Folding the read makes `input` a free operand but requires data-dependent selectors `Hsel[i,r]=(r==r0+3i)` / `Wsel[j,c]=(c==c0+3j)` as `[3,30]` F32 operands ≈ 4×90×4 = 960 B (plus one-hot `r0`/`c0` planes), vs GatherND's 216 B. The window term (`ones`, no input) and the kron term (4 input reads) have mismatched operand-arity, so they cannot share one term-axis Einsum without re-materializing. Slicing ch5 to a single-channel read = `[1,1,30,30]` F32 plane (3,600 B). All three read-fold variants lose to the 216 B GatherND. GatherND(batch_dims=1) IS the floor read.

**Why 921 is reachable but not by us:** Aditya is only 54–90 B below our tightest. The gap is plausibly a micro-construction we don't have documented (e.g. an anchor read that avoids one full `[1,30]` F32 marginal, or a 90-param half-placement) — but every input-reduction + sparse-read detector in our technique set bottoms out at ~975. No candidate staged. Verdict: **WALL** — freeoutkron 1011 remains our best; the cheaper P-detector that reaches < 921 is not expressible with the marginal/GatherND/free-output primitives we have.
