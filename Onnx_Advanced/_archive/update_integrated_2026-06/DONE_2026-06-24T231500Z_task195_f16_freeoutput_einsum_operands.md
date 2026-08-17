# F16 operands for a free-output Einsum that does NOT read the F32 graph input

**Where:** task195, worker w2 (beta_p29_claude_cli), 2026-06-24.
- Builder: `The_400_Tasks/Task_by_Task/task195/prototypes/build_task195_f16kron.py`
- Staged: `staged_submits/task195_cheaperkronpdetector_cost489/` (cost 489, sha256 955c4b25...)
- Prior best == team-best Boladi 525 (`task195_hybrid_slicekron`, sha256 d3189b02 — identical graph).

## What I discovered

The task351 carrier-dtype rule (ort_reality.md:55, techniques.md #109 carrier-dtype tax) says a free-output Einsum that takes the **F32 graph input as an operand** is F32-locked — its selectors can't shrink to F16 (shape-inference `ShapeInferenceError: inconsistent type`).

The **inverse** is the lever: if the free-output Einsum's operands are ALL initializer- or Cast-derived (it does NOT take `input` directly), then there is no F32 carrier to lock against, and **every operand can be F16**. ORT runs the F16 multi-operand Einsum end-to-end and the scorer accepts it.

task195's output node is:
```
output = Einsum('gtij,gtad,tk,ir,ar,jc,dc->gkrc', Q, Q, Ksel, Hb, Lb, Hb, Lb)
```
Operands: `Q` (Cast from a U8 Concat of ones + the gathered P), `Ksel`/`Hb`/`Lb` (initializers). NONE is `input`. Casting `Q` to F16 and declaring `Hb`/`Lb`/`Ksel` as F16 inits shrank the only remaining F32 head tensor:
- `Q` F32[1,2,3,3] = 72 B  ->  F16[1,2,3,3] = 36 B  (**-36 memory**)
- Params unchanged (params are element-count, not bytes).
- Output declared F16; decoder thresholds `>0.0` -> decodes identically (core.md).

Result: **525 -> 489**, single change. Gates: visible 265/265, truth 200000/200000, ab byte_equal=True 200265/200265 vs LIVE, profile==static==489 over 266 arcgen.

## Why it's decode-safe here (the task191 caveat does NOT bite)

techniques.md:3778 warns "F16 multi-operand einsum is NOT byte-equal (task191)". That caveat is about RAW float byte-equality. It does NOT apply when:
1. The output decodes through the NeuroGolf `>0.0` threshold (core.md), and
2. The Einsum values are exact small integers in F16. Here kron(P,P) ∈ {0,1}, the window term ∈ {0,1}, and `Ksel` does e5 - e0 (values in {-1,0,1}) — all exactly representable in F16, and all comparisons are against 0. So the {0,1} decode is bit-stable. Confirmed by the full truth + ab gates (200265/200265).

**Rule of thumb to add:** for a free-output separable Einsum producing a {0,1} (or small-integer) one-hot output that decodes via `>0.0`, prefer F16 operands UNLESS one operand is the F32 graph input. The F32-lock is operand-specific, not Einsum-wide. This is a near-free ~2× shrink on the float operand tensors of the head.

## What's missing from the library

The carrier-dtype tax (#109) is documented for the LOCKED case (input-as-operand). The UNLOCKED inverse — "if no operand is the F32 input, go F16" — is not stated as a positive lever anywhere I grepped (techniques.md #109, ort_reality.md:55, the F16-safe-class note at techniques.md:3271). Worth a one-line addition next to the carrier-dtype tax: the lock is per-operand; a free-output Einsum fed only by inits/casts is F16-eligible and decode-safe for {0,1} outputs.

## task195 floor note (for task_techniques_index / task195 row)

After f16kron 489, the dominant single charge is `x5f` F32[1,1,5,5]=100 B — the strided-slice anchor read. FORCED: Slice of the F32 input inherits F32; the 5x5 strided window is minimal for ArgMax-first-hit + [0,1,2] offsets (top/left ∈ {0,1,2}, need 3 consecutive of 5). Casting input->U8 before slicing materializes a [1,10,30,30] U8 plane (9000 B). This is the same F32-slice floor as the existing DONE_..._task370_f32slice_floor drop. 489 is the family WALL absent a structurally different anchor.
