# U8-wraparound interval band beats And(GE,LE) inside separable-Einsum component stacks

## What I discovered
The #148 U8-wraparound interval primitive (`band = Less(Sub(idx_u8, lo_int), span)`)
is a drop-in **30 B/band** saving for the 1-D band components that feed a deep
free-output separable Einsum (#119 / Aditya gap-merge family), not just for the
canvas/Equal-tail families it was first noted on.

A closed band `lo <= idx <= hi` over an integer index is normally built as
`And(GreaterOrEqual(idx, lo), LessOrEqual(idx, hi))` = **3 charged BOOL[1,N]** (the
ge, le, and). The U8 form is **2 charged tensors**: `Sub(idx_u8, lo_int)` (U8[1,N],
unsigned subtract wraps `idx<lo` to >=256) then `Less(., span)` (BOOL[1,N]). The
wraparound makes the lower-bound test free. Per band: 90 B -> 60 B.

Integer-bound conversion that is byte-exact for half-integer wall bounds:
`lo_int = ceil(lo) = Cast(lo + 0.999, U8)`, `hi_int = floor(hi) = Cast(hi, U8)`
(F32->U8 Cast truncates toward zero = floor for nonneg), `span = hi_int - lo_int + 1`.
Verified byte-exact vs the F32 ge/le band on all 1960 task284 generator cases x 10 bands
BEFORE building the ONNX. Requires **opset 14** (U8 `Sub`; opset 13 rejects U8 Sub with
"Type tensor(uint8) ... is invalid"). At opset 14 `ReduceSum` axes move to an input tensor.

## Where
- task284 (BracketFold). Builder: `The_400_Tasks/Task_by_Task/task284/prototypes/build_task284_gapmerge_u8band.py`.
- Replica of Aditya's 2818 (for ablation): `build_task284_gapmerge_opt.py`.
- Staged: `staged_submits/task284_bracketfold_cost2634/`.

## Why it matters
task284 team-best (Aditya `gapmerge`, free-output separable Einsum) **2818 -> 2634
(+184)**, gated: exhaustive 1960/1960, truth --full 200000/200000, ab --full byte-equal
200266/200266, score_v2 2634, profile==static 2634. The win is purely the band primitive
(10 bands x -30 B = -300 B memory) minus a small U8-ramp/scalar-overhead params bump
(73 -> 109 params; the 30-elem U8 idx ramp dominates the +36).

## Transfer / when to reach for it
Any graph that materializes several closed 1-D interval masks `And(GE,LE)` over a shared
integer index ramp (band/component stacks for separable free-output Einsums, staircase
families, marginal-profile architectures). Net win = `30B x n_bands - 30 (U8 ramp param) -
~8B/band ceil/span scalar overhead`. Worth it for n_bands >= ~2.

## Floor note recorded on task284
This family's honest static floor is ~2600: row/col F16[6,30] components = 720 B (F16 is the
cheapest ORT-Einsum-legal dtype; INT8/INT16 Einsum has NO kernel -> score None), + BOOL[6,30]
concats 360 B, + F32[1,30] active-grid ReduceSum 240 B (grid extent is rand and NOT marker-
derivable). K=6 components is minimal (arm-merge blocked by 2 distinct colors). The 3.7k
dynamic Range/ScatterND lane is BANNED (dim_param -> calculate_memory None -> scores 0).
Sub-2400 is not reachable in a guaranteed-correct static graph here.

## What was missing from the library
techniques.md #148 documents the U8-wraparound interval fill for U8-canvas/Equal-tail
families; it did not call out that the same primitive transfers to the BOOL band-component
stacks of the #119 free-output separable-Einsum family (Aditya's gap-merge nets). Adding that
cross-reference under #148 and/or #119 would route future workers to it.
