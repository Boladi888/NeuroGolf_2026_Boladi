# Folding an explicit `Pad(neg_big)` + `MaxPool` chain into one `MaxPool(pads=...)` — free cost cut

## What I discovered
A graph that pads a feature map with a large-negative constant and then `MaxPool`s it
(`Pad(x, [.,.,1,1,.,.,1,1], neg_big) -> MaxPool(kernel=[3,3], strides=[1,1])`) is
**byte-identical** to a single `MaxPool(x, kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1])`.
ORT's MaxPool intrinsic padding ignores out-of-bounds cells in the max reduction (it behaves
as if padded with `-inf`), so an explicit `Pad` with any value `<=` the real minimum is redundant.
You can delete the `Pad` node, wire `MaxPool` directly to the pre-pad tensor, and add the
`pads` attribute. **Note the MaxPool `pads` attribute is spatial-only**: for 2D it is the 4-value
`[h_begin, w_begin, h_end, w_end]`, NOT the 8-value NCHW form Pad uses (passing 8 values =
`Attribute pads has incorrect size`).

The saved cost is the entire padded-tensor footprint: each eliminated pad tensor is
`(H+2)*(W+2)*itemsize`. Also remember to drop the now-unused `pads`/pad-value initializers
(the scorer counts every initializer element in `params`).

## Where I encountered it
task277 (component-label / size-rank route). The team-best did 5 rounds of
`Pad(neg_big) -> MaxPool` F16 label propagation. Each pad was `[1,1,12,12]` F16 = 288 B.
Folding all five into `MaxPool(pads=[1,1,1,1])` removed 5*288 = 1440 B + the unused `pads_hw`
init (8 params). Cost 6698 -> 5250 (-1448), AB vs team-best 200266/200266 byte_equal=True,
visible 266/266, profile --arcgen 266 static==profile==5250.
Staged: `staged_submits/task277_beat6698_cost5250/`.
Builder/artifact: `The_400_Tasks/Task_by_Task/task277/prototypes/task277_maxpool_pads.onnx`.

## Why it matters
This is a pure, lossless cost refactor whenever a graph builds an explicit constant-pad solely
to feed a MaxPool (common in iterative flood / dilation / label-propagation stacks). The current
library (`cellular_automata.md`, `techniques.md` dilation recipes) shows the *fresh* form
`MaxPool(..., pads=[1,1,1,1])` but does not flag that an **already-built** `Pad(neg_const)+MaxPool`
pair can be collapsed for free — workers inheriting a teammate graph may not realize the pad tensor
is removable. The `-inf` semantics only hold for MaxPool's max; the same fold does NOT apply to
AveragePool (padding affects the average / count_include_pad) or to Conv (pad value is summed).

## What's missing from the library
A one-liner in `techniques.md` (near the §41 MaxPool-pads material around line 2655) or
`cellular_automata.md` (near the dilation recipe) saying: "if you inherit `Pad(neg_big)+MaxPool`,
delete the Pad and use `MaxPool(pads=spatial4)` — saves the full pad-tensor footprint; MaxPool
intrinsic padding == -inf. Spatial-only 4-value pads attribute for 2D."
