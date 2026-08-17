live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route (the single biggest cost lever): shrink the graph output from
[1,10,30,30] to a 13x13-native [1,10,13,13], deleting the final `Pad`
(merged_u8 -> color30_u8) and the 900-byte `color30_u8 [1,1,30,30]` canvas.
All 267 task089 grids are exactly 13x13 (train 4 / test 1 / arc-gen 262), so a
13x13 output looked plausible.

Build: `prototypes/bo_ng_cld_worker1_20260711/task089_13x13_output_probe.onnx`
(sha256 99af903ca46a1794b4c14d5588307e8dbd6efdbe473e0c19fa8b806e4e9095a0).
score_network_v2 = (0, 3078, 91) -> cost 3169 (-905 vs 4074): removes
`color30_u8` 900 B memory + `pad_spatial`(4) + `sentinel10_u8`(1) params.

Failed gate: `ng_gate visible --task 89` -> 0/267 (0.00%) FAIL. Certain hidden 0
(traps §VISIBLE-FAIL). Truth/ab/profile correctly skipped.

Why dead (source-read proof, not just empirical): the LOCAL gate compares raw
tensors. `ng_harness/modes.py::mode_visible` builds the expected tensor with
`encode.encode_grid(output)`, and `ng_harness/encode.py::encode_grid` ALWAYS
allocates `[1,10,30,30]` (H=W=30 hardcoded). `ng_harness/session.py::run_raw`
returns the model's raw output with no reshape/pad. So
`np.array_equal([1,10,13,13],[1,10,30,30])` is False for every pair. The output
shape is HARD-LOCKED to [1,10,30,30] regardless of the task's true 13x13 grids.

Consequence for the tail floor: producing a spatially-2D-varying [1,10,30,30]
output requires the last op's largest input to be a full-canvas tensor. The
cheapest such intermediate is a 1-byte scalar color canvas `[1,1,30,30]` = 900 B
fed to `Equal(canvas, color_bank[1,10,1,1]) -> output`. Equal-then-Pad instead
materializes a one-hot `[1,10,13,13]` = 1690 B (worse); Pad-of-one-hot is 1690 B;
Where/Concat/Scatter tails need an equal-or-larger canvas or an INT index bank.
Therefore `color30_u8` = 900 B is the proven tail floor. The reviewer's prior
"color30_u8 forced" lock is correct and now empirically confirmed.

Remaining hypothesis: none for the tail. A sub-900 tail is impossible while the
harness pins the output to [1,10,30,30] and ONNX has no sub-byte tensor dtype.
