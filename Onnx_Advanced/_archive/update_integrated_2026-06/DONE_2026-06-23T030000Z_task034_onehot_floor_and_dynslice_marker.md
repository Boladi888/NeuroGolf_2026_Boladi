# task034 — one-hot output floor + free-input-Einsum-locate + dynamic-2x2 marker read

**Where:** task034 (DissolveDistanceFieldOneHot). Builder
`The_400_Tasks/Task_by_Task/task034/prototypes/build_task034_einsum_v2.py`.
Staged `staged_submits/task034_einsum_dynslice_cost2488/` (cost 2488, profile==static,
visible 267/267, truth 200000/200000). Beat LIVE 3474 by -986 (28%).

## 1. CORRECTION to a recurring prompt premise: a data-dependent 10-ch BOOL one-hot output is NOT dissolvable by free-output overlay

Several task034 prompts (and the brief here) assumed the 810B pre-Pad one-hot
`out9 [1,10,9,9] BOOL` could be killed with a "free-output Where(mask,color,input)"
overlay (#67/#78). This is FALSE when the output is the 10-channel one-hot itself
(official encoding: cell color k -> channel k = 1) AND the color is data-dependent:
- The tail MUST be `Equal(value_canvas, channels)` (Equal-tail, #67).
- The pre-Pad `out9` [1,10,9,9] = 10*81 = **810B**. The only way to make Equal free
  is to make it the graph `output` directly, which forces the value canvas padded to
  [30,30] (U8 = **900B** intermediate). 810 < 900, so the channel-small one-hot +
  spatial-Pad-as-free-output is already CHEAPER.
- This IS exactly INDEX #67 "active-region boundary (task046)": fold to full canvas
  wins only when `C * active_cells < full_grid_cells`. Here 10*81=810 < 30*30=900,
  so keep the channel-small one-hot. The live graph was already optimal on the tail.

**Lesson for routing:** before promising a "kill the one-hot" win, compute
`C*active_cells` vs `full_grid_cells`. If the one-hot is already channel-small AND
< full canvas, the 810 (or whatever C*active) is the FLOOR — redirect the attack to
the INPUT reads and the mask machinery, not the output.

## 2. Two reusable cost cuts that DID work (apply to any "locate-a-block + read-a-few-cells" rule)

a) **Locate (r0,c0) via free-input Einsum marginals, not an F32 channel slice.**
   `Einsum('bchw,c->bh', input, nonbg_sel[10])` + ArgMax gives the block's top row
   for 0 memory in the Einsum (probe: (0,0,10)); the only counted tensor is the
   [1,30] F32 marginal (120B). Killed `bg9` 324 + nonzero4 81 + nonzero_f16 162.
   (Generalizes §42 task047 marker-locate to a block-locate.)

b) **Read K specific cells at a DATA-DEPENDENT position via ONE dynamic Slice with
   static channel + dynamic h/w**, output shape K-small. For the 4 corner red-flags:
   `Slice(input, starts=[0,2,r0,c0], ends=[1,3,r0+2,c0+2])` -> [1,1,2,2] F32 = 16B,
   replacing the static `marker9` [1,1,9,9] (324) + flat_marker + Gather chain.
   TRAP I hit first: slicing the channel statically in a SEPARATE node leaves a
   [1,1,30,30] = 3600B plane — you MUST fold channel+hw into the SAME Slice.
   Dynamic-slice output has unknown shape after strict shape-inference -> inject
   explicit `value_info` for it (survives infer_shapes; it only fills unknowns, never
   overrides concrete dims). profile_real_cost confirmed runtime==static (no gap).

c) Minor: express `|a-b|<=1` band as two BOOL broadcast compares
   (`a<=b+1 AND b<=a+1`) instead of `Sub->Abs->Le` — drops two [9,9] F16 planes
   (324) for one extra BOOL plane. And U8 value-canvas + Equal(U8) halves the
   pre-onehot canvas (162 F16 -> 81 U8).

## What's missing from the library
INDEX #67 has the task046 boundary rule but it's buried; a one-liner "data-dependent
10-ch one-hot output: 810 is the floor when C*active < full canvas — attack inputs,
not output" would have saved this worker the dead-end exploration. Also no catalog
entry for "block-locate via Einsum marginal + dynamic-2x2 cell read" as a paired
pattern for the common ARC "find the small object, read a few of its cells" family.
