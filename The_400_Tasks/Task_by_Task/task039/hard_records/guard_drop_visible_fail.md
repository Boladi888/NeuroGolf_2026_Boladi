# task039 hard record - guard_drop_visible_fail

Live/team bar used: cost 277, profile cost 277, sha256
`1059d542c013353bd312f22c8368baf43d37da3a0f2e18237427a3a978293b27`,
from `python tools/ng_current_bar.py --task 39 --json` on 2026-07-13.

Attempted route: delete the row/column extra-cell guards from the Sho QLinearConv
color-index graph and feed the base 5-column/5-row emptiness detectors directly
to the row/column start sums. Prototypes:
`task039_worker3_drop_row_extra.onnx`,
`task039_worker3_drop_col_extra.onnx`, and
`task039_worker3_drop_both_extra.onnx`; builder:
`The_400_Tasks/Task_by_Task/task039/prototypes/build_task039_worker3_guard_drop_variants.py`.

Exact failed gates:
- `python tools/ng_gate.py visible --task 39 --cand .../task039_worker3_drop_row_extra.onnx`
  -> `[visible] 255/264 (96.59%)  FAIL`
- `python tools/ng_gate.py visible --task 39 --cand .../task039_worker3_drop_col_extra.onnx`
  -> `[visible] 250/264 (94.70%)  FAIL`
- `python tools/ng_gate.py visible --task 39 --cand .../task039_worker3_drop_both_extra.onnx`
  -> `[visible] 241/264 (91.29%)  FAIL`

Why dead: the extra sampled cells are load-bearing. Some visible inputs place
the first non-background row/column evidence outside the base 5x5 detector
window, so dropping either guard shifts the detected crop origin. Per the
visible-fail axiom, no truth/AB/profile gates were run on these variants.

One remaining hypothesis: keep the extra-cell evidence but merge it before
binarization. The 2026-07-13 maxpool-maxmerge candidate does this with
`Max(UINT8)` plus the existing qneg binarizer and passes visible/truth/AB.
