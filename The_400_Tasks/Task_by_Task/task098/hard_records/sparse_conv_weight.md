route: sparse Conv initializer

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: preserve the exact live one-node Conv math but replace the dense `[10,1,3,9]` initializer with an ONNX sparse initializer. The dense kernel has 96 non-zero values, so this would be a cost win if the scorer accepted sparse Conv weights.

Exact failed gate:
- `task098_sparse_exact_probe.onnx`: `score_network_v2 -> None`.

Why dead: the project core/traps warning holds on this scorer path: sparse initializers feeding Conv are rejected before a valid cost can be assigned. No visible/truth gates were run on an invalid-score graph.

Remaining hypothesis: only a future scorer/runtime change that accepts sparse Conv initializers would revive this exact-math compression.
