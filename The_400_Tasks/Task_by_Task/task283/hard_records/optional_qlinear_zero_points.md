# task283 - optional QLinearConv zero points dead branch (2026-07-13)

- Live bar used: **642**, SHA-256 `363c5fe0400628c3bebe342041d5eabef04da2d5d6db9082a9341650f99c483d`, from `python tools/ng_current_bar.py --task 283 --json`.
- Attempted route: keep the 1x1 negative-pad Conv encoder but omit QLinearConv zero-point initializers and declare UINT8 output to save the shared zero scalar.
- Exact failed gate: `score_network_v2` returned `None`; `onnx.checker.check_model(full_check=True)` failed with `Node ()'s input 2 is marked single but has an empty string in the graph`.
- Why dead: QLinearConv's middle optional zero-point inputs cannot be skipped with empty-string placeholders under the official checker when later inputs are still supplied.
- One remaining hypothesis: keep explicit zero-points and attack the 10-param bias or 90-param output kernel instead; the explicit-zero 1x1-negative-pad sibling is staged at cost 612.
