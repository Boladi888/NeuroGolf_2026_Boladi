# Transferable insight — the "color-index collapse to ONE ArgMax index plane" sweep is a FALSE FLOOR: ArgMax output is INT64 (7200 B), and the live best usually already collapsed the per-color path

**Source**: task064 (`bo_ng_cld_worker3`, 2026-07-10). Audited LIVE **8695** (Sho, `t064v3`,
sha `7562db13…`); this is a **DROP / floor re-confirmation**, not a new win. Sibling of
`DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md`
(task254) — same "SWEEP prompt targets a component the live best already dissolved" trap,
different collapse verb (color-index/ArgMax here, free-output Einsum there).

## The trap it kills
The EV/prompt-writer instinct: *"the per-color path is the charged carrier; collapse to ONE index
channel (`ArgMax(input,axis=1)`), do spatial logic on it, `Gather/Equal` only at the boundary; delete
the carrier/canvas/template; floor_est ~900 (just the output tail)."* Two independent reasons it fails:

1. **ArgMax output is ALWAYS INT64 — the "one index plane" is a COST TRAP, not a saving.** ONNX
   `ArgMax` has no output-dtype attr. A full-grid index plane is INT64 `[1,1,30,30] = 7200 B` (or
   `[1,1,24,24]=4608`), and it is a charged INTERMEDIATE (`calculate_memory` charges every tensor not
   literally named `input`/`output`). Measured: a 2-node `ArgMax → boundary-Equal` probe scores
   `mem 8100 = 7200(idx) + 900(mask)`. So the route prices **≥7200 before ANY detection or ray fill** —
   ~8× the ~900 est and at/above the live 8695. You cannot cheaply get a 24×24 index plane either:
   ArgMax@30×30 then Slice charges BOTH (7200+4608); Slicing the F32 input first = `[1,10,24,24]` F32 =
   23040. Cropping/casting never lowers the ArgMax route floor.
2. **The live best has usually ALREADY collapsed the per-color path — without a spatial index plane.**
   `t064v3` handles color as SCALAR indices (`ArgMax over the [1,10] color axis` → `b64`/`d64`, not a
   spatial grid) + per-color MOMENTS (Einsum `cnt/Sr/Sr2` @40 B ea) + ONE F32 projection. There is no
   10-channel carrier/canvas/template left to delete. The Direction's charged component is already gone.

## The mechanism (why ~900 is false — the forced per-cell projection)
`floor_est ~900` counted only the output tail. The rule needs, per in-span row, the leftmost/rightmost
box-or-dot column (the ray extent), which is **data-dependent per row** (dots are scattered). A per-row
reduction REQUIRES a per-cell plane; moments give only global box scalars, never per-row dot extents.
Cheapest per-cell plane from an F32 one-hot input is an **F32 `[1,1,24,24]` projection = 2304**
(live `SD`, via `Conv(input, box+dot onehot)`); the INT64 ArgMax alternative is strictly worse (7200).
Add the multi-rank ray canvas (`fullH`/`fullV` BOOL 576×2, non-separable — near-end is per-row min/max
dot col) + the forced `B30` output-extent mask (900). Refreshed floor ≈ **2304 + 1152 + 900 + walls/
detection ≈ live 8695**, not 900.

## Rule of thumb
Before taking a **"color-index collapse / single-index-channel / floor_est ≈ best/9"** SWEEP prompt at
face value, `inspect_graph.py` the LIVE best first. (a) If it already reasons about color via **scalar
color-axis ArgMax + moments** (no spatial `[1,1,H,W]` INT64/F32 index plane), the per-color path is
already collapsed → the prompt's charged component is gone → **DROP**. (b) Never "collapse to a spatial
`ArgMax` index plane" as a *cost* play — ONNX ArgMax is INT64, so the plane costs 4608–7200 and is a
regression. (c) When the render needs a **data-dependent per-row/per-column reduction**, a per-cell
projection plane (F32 [H,W]) is FORCED and its cost dwarfs the output tail — so `floor_est =
output-tail-only` is false. Treat these HIGH-confidence `best/N` "SWEEP" estimates as unproven when the
best already collapsed color to scalars and the rule needs a per-cell reduction.
