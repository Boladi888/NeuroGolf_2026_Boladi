# Selector Unification 245 Probe

- Live/staged bars used: authoritative live cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`, source `python tools/ng_current_bar.py --task 51 --json`; task-local staged reviewer bar cost 251, sha256 `a8052987d7b160721429ddd03f552ffff29d848603e1dab6626ff029f493e23b`.
- Attempted route: delete either `term_t_sel_swapped` or `term_t_sel` by replacing all uses with the other selector, with and without swapping `orientation_basis` order (`[1,sh,sv]` vs `[1,sv,sh]`).
- Exact failed gate: `ng_gate.py visible` on `task051_worker1_v05_drop_swapped_all.onnx`, `v06_orientswap_drop_swapped_all.onnx`, `v07_drop_term_t_all.onnx`, and `v08_orientswap_drop_term_t_all.onnx` each returned `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the terminal tensor network needs both selector maps; orientation relabeling alone just moves the mismatch and breaks every official pair.
- Remaining hypothesis: only a deeper terminal refactor that changes the surrounding row/column/color factors together could remove one selector; do not retry global selector replacement as a standalone 245-cost shave.
