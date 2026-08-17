live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route (second-pass, genuinely-different TAIL assembly): the reviewer-
cited task083 "PAD-LAST REORDER COROLLARY" (techniques.md:3380) — build the
one-hot at the small pre-pad extent then Pad-as-output, instead of Pad-ing the
1-channel scalar index up to [1,1,30,30] then Equal-last.

Current tail: merged_u8[1,1,13,13] -> Pad -> color30_u8[1,1,30,30] (900 B) -> Equal -> output.
Variant:      merged_u8[1,1,13,13] -> Equal(color_bank) -> onehot13[1,10,13,13] (1690 B BOOL) -> Pad -> output.

Build: `prototypes/bo_ng_cld_worker1_20260711/task089_freeoutput_pad_tail.onnx`.
score_network_v2 = (0, 4768, 100) -> profile cost **4868** (static==profile).
`ng_gate visible` -> **267/267 (100%) PASS** — the variant is fully CORRECT.
`ng_graph_cost_diff` named delta: +1690 B `onehot13` (added) − 900 B `color30_u8`
(removed) + pad-init churn = **+794 B net (REGRESSION)**.

Why dead: NOT a correctness failure — a byte regression. task083's crossover is
pre-pad h*w < 90; task089 is 13x13 = **169 > 90**, so the 10-channel pre-pad
one-hot (10*169 = 1690 B) is strictly larger than the 1-channel padded index
(900 B). The reordering is exactly the wrong direction for a 13x13 grid.

Consequence: the 900 B `color30_u8` tail is now bracketed on BOTH sides and is
proven the tail floor:
  - shrink it (13x13-native output) -> 0/267 visible (output shape hard-locked
    to [1,10,30,30]; see output_13x13_shape_lock.md).
  - reorder to pre-pad one-hot (this route) -> correct but +794 B.
Any spatially-2D-varying [1,10,30,30] output needs a >=900 B carrier; the
1-channel scalar index + Equal-last is the minimum. No sub-900 tail exists.

Remaining hypothesis: none for the tail. Do not re-try free-output/Pad-last one-
hot reordering on any grid with active h*w >= 90.
