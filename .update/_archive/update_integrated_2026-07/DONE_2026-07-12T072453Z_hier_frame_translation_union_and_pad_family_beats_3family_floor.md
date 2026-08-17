# Transferable insight — a data-dependent translation/tiling UNION can be built cheaper by HIERARCHICAL FRAMING (early copies in a small frame), and the Pad-translation render family beats the Conv/Gather/coord "3-family floor" for diagonal spreads

**Source**: task370 (`bo_ng_cld_worker1`, Claude lane, 2026-07-12). Live team-best **5,446**
(Sho, sha `34d89d01…`, separable diagonal Pad-translation, log-doubling Or-union). Staged **5,059**
(−387, PROBE_READY): output-BYTE-EQUAL to the 5446 bar on all 266 visible+arcgen (0 mismatches) ⇒ same
99.95% truth by construction; visible 266/266; truth --n2000 1999/2000 (==bar); profile static==profile==5059;
ng_safety PASS; ng_graph_cost_diff NET −387 reconciled. Builder:
`Task_by_Task/task370/prototypes/build_task370_hier_double.py`.

Sibling of `DONE_2026-07-12T004531Z_sparse_param_lever…` (§3 floor_est mis-model) and
`Task_by_Task/task370/lessons/LESSON_2026-06-30_datadep_translation_family_floor.md` (the 3-family analysis
this packet CORRECTS).

## 1. HIERARCHICAL FRAMING: build the early copies of a translation union in a small frame
When the render is `U = OR_{k=1..K} shift(window, k·v)` with a data-dependent step `v` (|v|≤S) and a small
`W×W` window — the classic **log-doubling** union materializes K copies all at the FULL canvas size
(`N×N` BOOL = N² B each). But the first doubling {k=1,2} spans only `W + S` cells, so it fits a
`(W+S)×(W+S)` frame, NOT `N×N`. Build {k=1,2} there (cheap), THEN place into the full `N×N` and do the
remaining doubling(s) where the union genuinely spans N.
- task370 numbers: window 5×5, step d≤5 ⇒ {k=1,2} fits **10×10 (100 B)** not 20×20 (400 B). Doubling
  `s1,s2b,U1,s2,U2` (5× 400 = 2000) → `s1s,s2s,U1s`(3×100) + `s1big,s2big,U2`(3×400) = **1500**. −500 gross,
  −~150 overhead (the local-place offset scalars + one extra Concat pad-spec) = **−350 net, output-byte-equal**.
- **Geometry that keeps it EXACT (no divergence):** place the k=1 copy at local offset `off = max(0,−v)`
  (leave room for a negative shift), local after-pad `= S−off`; then place the local union into the big
  canvas at `before = base − off`, big after-pad `= (N−frame) − before`. Both pad-sums are CONSTANT
  (`off+(S−off)=S`, `before+((N−frame)−before)=N−frame`) so static value_info stays inferrable, and the
  big-canvas placement crops at the N-boundary IDENTICALLY to the direct scheme. The small frame must be
  ≥ `W+S` so no copy is cropped locally that would survive at N — then U is byte-identical.
- **Triage rule:** worth it when `K≥4` and `W+S ≤ N/2` (early copies shrink quadratically). Prove
  output-byte-equality vs the parent over visible+arcgen (dual-session run) → inherits parent truth for free,
  ship as a pure cost trim even on an ambiguous-class (<100% truth) task.

## 2. Pad-translation is a 4th render family that BEATS the "3-family floor" for data-dependent diagonal spreads
`LESSON_2026-06-30` (task370, then cost 8868) declared a "3 implementable families" hard floor:
A=dynamic-Conv/Scatterband kernel (won at 841 kernel), B=Gather-shift (14,400 B index, dead),
C=coord free-output (3,200 B grids, dead) — and concluded **"8868 is a hard floor, dynamic-conv is the
minimal family."** That proof was WRONG: Sho then shipped **5446** with a **Family D = Pad-translation**
(Slice a small window → `Pad` to place → `Pad`(before=v,after=−v) to shift, keeping size → `Or` to union;
log-doubling gives 2^m copies in 2m tensors). Pad-shift materializes only the `N×N` BOOL union tensors
(400 B) and NEVER the Conv kernel (841) or a K·cells index (14,400) — so it dominates all three prior
families here. **Correction to the triage arithmetic:** when the step is a rigid translation of a *small*
window (not a per-cell-variable spread), price Family D = `(#copies as doubling tensors)·N²·1B` and compare
FIRST; it usually wins. A floor proof that enumerates "the N families" is suspect if Pad-translation is not
one of them.

## 3. Third confirmation: "TopKCompaction floor_est ~1200" is a prompt mis-model when the output is dense
task370's card = "TopK/ArgMax compaction, floor_est ~1200". Actual rule = diagonal zero-mask translation
recolor; OUTPUT is a dense 30×30 edit ⇒ a `Where(mask30, marker, input)` forces a **900 B BOOL paint mask**
alone, and the separable analysis (marker/zero-box/in-grid projections 720 + bounds 800 + union 1500) puts
the real formulation floor near **5000**, not 1200. Same mis-model as task182/task208 (`DONE_…004531Z §3`,
`DONE_…041239Z`): when a SWEEP/STRUCTURAL card's `floor_est` assumes a select-K-winners shape but the output
is a per-pixel dense edit, the compaction framing is wrong and the output mask + per-position score set the
floor. **Do not chase `floor_est` down to the TopK number on these.**

## Sorter notes
- techniques.md (ray/translation family, near #132/#110 and the 2026-06-30 K·cells-vs-(2D+1)² arithmetic):
  add Family D (Pad-translation) as the usual winner for rigid small-window diagonal tiling, + the
  hierarchical-framing sub-technique (§1) with the exact-crop geometry.
- by_symptom.md ("union of shifted copies too expensive" / "log-doubling all at full canvas"): point to §1.
