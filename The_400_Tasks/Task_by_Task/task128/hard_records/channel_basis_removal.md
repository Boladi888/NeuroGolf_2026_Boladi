# task128 channel_basis_removal

- Live bar: cost 104, SHA256 `ccdb0c525b1f0ba110466e0c36a9f4eedb963c098794a264c725a71595205798`, source `python tools/ng_current_bar.py --task 128 --json` manifest authority.
- Attempted route: removed the live `channel_basis[2,10]` operand and used only the non-background coefficient slices for all channels in a single free-output `Einsum`.
- Builder: `The_400_Tasks/Task_by_Task/task128/prototypes/build_task128_poly72_no_channel_basis.py`
- Probe: `The_400_Tasks/Task_by_Task/task128/prototypes/task128_poly72_no_channel_basis.onnx`, nominal/static cost 72, SHA256 `1375a5d2dc9d9ac7cedf5ec328c3215b3655dbc2feb6c3bccba610c58444ebab`.
- Exact failed gate: `python tools/ng_gate.py visible --task 128 --cand The_400_Tasks/Task_by_Task/task128/prototypes/task128_poly72_no_channel_basis.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
- Why dead: the background-vs-non-background channel selector is load-bearing; applying the color polynomial to channel 0 destroys every visible case. Per visible-fail axiom, do not run truth/AB or build variants of this selector-removal family.
- One remaining hypothesis: a materially different row/channel polynomial basis, not a deletion of the existing channel selector. Static audit found no rank-1 factorization of the three `2x2x2` coefficient cores.
