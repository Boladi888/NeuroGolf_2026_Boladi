# task018 — DEAD ROUTE: Pad/Concat "template sweep → predicate tail → ~900" (false floor)

**Worker:** bo_ng_cld_worker3 (Claude lane) · **Date:** 2026-07-10 · **Verdict: DROP**

## Live bar (re-measured this session)
- **live_cost = 19125** — `ng_current_bar.py --task 18`: authoritative_source=`manifest`,
  sha256 `d6567a94331bf690c8e85c3361b13713e4f77e327fef0d552070f9dfe76da3fb`, updated 2026-07-02.
- Re-profiled: `profile_real_cost --task 18 --arcgen 266` → **static=19125 profile=19125**
  (memory 18486 + params 639; 266 real inputs). No grader-gap; exact-static.
- The live model == staged `task018_anchor4_cost19125_probe` == **byte-equal to the shipped
  submission** (`ab --full`: match 200266/200266, byte_equal=True). Path-B solver (truth 98.9%),
  so byte-equal-vs-submission is the gold gate, not 100% truth.

## Attempted route (from prompt Direction / Fast probe)
"Pad/Concat/template sweep: avoid stored/materialized canvas, generate a predicate/index and let
the final output op materialize; delete the charged carrier/canvas/template. Target ~900."

## Why it's dead (priced wall, not a stale sweep)
The prompt is a generic SWEEP-class hypothesis triggered by `Pad` + `Concat` appearing in the op
signature. The **live graph is a genuine 2-creature colored-object relocation solver**
(`task018_structural_v54`, 242 nodes), NOT a template fill. The charged component the Direction
wants to delete **does not exist**:

- **The single `Pad` (node 235) IS the forced output canvas** — `raw_label_masked_padded [1,1,30,30]`
  = 900 B. This is the grader-read output, explicitly "forced" per the prompt's own floor note.
  The prompt's "~900 target" is literally just this output-canvas size.
- **The 6 `Concat`s are all load-bearing**, not repeated template planes: they build the D4
  rigid-transform coordinate vectors (`tr_flat_sparse_T1/T2`, `tc_flat_sparse_T1/T2`) and merge the
  two creatures' colors/positions (`all_col_safe`, `all_pos_i16_masked`) for the final scatter.
- **`ScatterElements(zeros577, all_pos, all_col_safe) → canvas`** (node 232) is already a minimal
  576-B flat output; there is no oversized materialized carrier to delete.

**Fast-probe result (analytic falsification — kill rule triggered, did not grind):** the output is
data-dependent object reconstruction — per-input creature positions, D4 rotation ∈{1,2,3,4}, colors,
and the CC-separation of two *color-identical* creatures. A predicate-only Pad/Concat-free tail paints
an input-independent canvas ⇒ **falsifies on fresh ARC-GEN** (near-0 accuracy). To be correct it must
re-add the flood→detect→transform→warp pipeline ⇒ prices back at ~19k. Either branch hits the kill
rule (falsifies OR prices ≫ 900).

## Where the 18,486 charged bytes actually go (largest_intermediate, top items)
| bytes | dtype | tensor | forced by |
|---:|---|---|---|
| 2,304 | F32 | `color_idx_full_f32` (Conv) | F32-forced — Conv F16-out on F32 input impossible (Cast=18k) |
| 1,152 | F16 | `nonbg_score_f16` (TopK score) | F16-forced — UINT8 TopK is ORT NotImplemented |
| 900 | U8 | `raw_label_masked_padded` [30,30] | the forced 30×30 output canvas |
| ~30 × 576 | U8/BOOL | flood rounds (3× MaxPool+Min), color/filler/anchor/region masks, seed scatter, per-template transform coords, canvas | each a distinct semantic mask; visible-tight |

## Independent formulations tried (all wall)
1. **Predicate-only template tail** → falsifies (input-independent output).
2. **Dtype-narrow the two biggest tensors** → color Conv F32→F16 impossible on F32 input; TopK
   score F16→U8 ORT-blocked (`v125_u8_topk`: TopK(11) UINT8 NotImplemented).
3. **Reduce the flood CC-separation** → ONNX has no sparse-CC op (NonZero/Unique banned); wider/single
   -round kernel bridges the two color-identical creatures → `ScatterElements idx=576 out of bounds`
   crash (`v126_k10`) or visible fail. 3-round clamped MaxPool flood is structurally required.
4. (Prior Codex forced 2nd pass) constant-shave floor: anchor cap 6→5→4 works (19175→19150→**19125**),
   **3 fails visible 53%** (`v129_anchor3`); body K=12 / nonbg K=30 visible-tight. 19125 is the floor
   of the shave route.

## The "8154 ceiling" is not a valid bar
Documented in both staged manifests as a **zero-public-weight fullcount/metapack lookup artifact**
(memorized lookup, scores 0 on hidden). Explicitly "not used as the current structural reviewer-ready
bar." The task's cost→public-score link is broken for the lookup family (see task018.md don't-do log).
The real structural bar is 19125.

## One remaining hypothesis (for a future batch, LOW priority)
Only a fundamentally different *algorithm class* could beat 19125 — not a Pad/Concat dissolve and not a
constant shave. Every charged component here is dtype-locked (F32 color Conv, F16 TopK) or structurally
forced (no-sparse-CC flood, forced output canvas). No sub-19125 structural route survived. Treat 19125
as the effective floor; do not re-run the Pad/Concat template-sweep probe on this task.
