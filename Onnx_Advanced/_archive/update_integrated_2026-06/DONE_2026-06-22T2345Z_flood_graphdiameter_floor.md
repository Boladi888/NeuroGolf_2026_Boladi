# Connectivity-flood cost floor: the round count is a graph-diameter floor, not a knob

**Where:** task048 (box-path flood / lane-squeeze connectivity, output 1×1 {0,8}).
Builder `The_400_Tasks/Task_by_Task/task048/prototypes/build_task048_v8.py`; staged
`staged_submits/task048_v8_fullbox_k5_cost642/`.

## What I confirmed (re-validation session, w-048)

The mature row-byte #88 bit-packed flood for grid connectivity has its cost dominated by
`72·k` bytes (k = flood rounds). The temptation is to keep shaving k. I proved k is a hard
**graph-diameter floor**, not a tunable:

1. Built v8 at k=4 (cost 570) → **266/270 visible-fail** (certain hidden 0).
2. Ran an INDEPENDENT ground-truth BFS (8-conn, full-box seed → full-box target) on the
   failing visible pair (task048 train#2): box A top-left rows0-1, box B bottom-right
   rows4-5 — the path through the cyan noise genuinely needs **exactly 5 dilation rounds**.
3. So k=5 is irreducible for that pair. No seed/target geometry helps: pre-dilating the seed
   by one step IS one extra round of cost. You pay the box-to-box distance somewhere.

**Generalizable rule for the library (flood/connectivity floor law):** before trying to
drop a flood-round count, run a per-corpus BFS for the *maximum* box-to-box dilation
distance under the chosen seed/target framing. That max IS the minimum k. The only legit
k-reductions come from a *framing* change that shrinks the diameter (e.g. single-seed→
full-box dropped task048 from k=6→k=5 because both endpoints widen by ½ step), and those
saturate at the true geometric diameter. Confirms #134 "forced-component, not theorem"
discipline: the wall is for the routed idea-class.

## The F32 encoder-bridge wall (worth a one-liner in cost_floor_reference.md)

The first charged tensor in any one-hot-input graph is the extraction op's output, and it is
**F32-locked** because the NeuroGolf encoder hands you `input` as F32 [1,10,30,30] and Conv
(and every other op) preserves input dtype. task048's pack-conv output `packed_f` F32
[1,2,8,1] = 64 B cannot become 32 B: ONNX has no dtype-changing Conv attr, and a trailing
Cast(→F16) keeps the 64 B F32 alongside the new 32 B ⇒ net worse. Any non-conv extract
(Slice/Gather a channel) materializes a [1,1,30,30] F32 = 3,600 B plane. So a fused
extract+crop+pack Conv producing the smallest possible F32 tensor is the floor for the
input bridge. (Cross-checks #88 row-byte pack and the "crop to 8×8 first" advisory.)

## zeros-canvas params wash (minor, but stops a wasted probe)

A ScatterElements all-zeros `data` canvas as an N-element U8 initializer (N params) cannot
be made free by generating it at runtime (`Mul/BitwiseAnd(x, 0)` or `ConstantOfShape`):
the generated tensor is an N-byte value_info intermediate, so params −N == memory +N. Exact
wash whenever the canvas size equals its element count (U8). Don't reach for it.

## What's missing from the library

`cost_floor_reference.md` has flood floor laws but (as of this session) no explicit
"round-count = corpus-max BFS diameter under the seed/target framing" statement, and no
"F32 encoder-bridge tensor is dtype-locked at the extraction op" note. Both are
general across the connectivity/flood family. Suggest the sorter add them there (and a
by_symptom.md route: "I see a k·constant flood and want fewer rounds → BFS the diameter").
