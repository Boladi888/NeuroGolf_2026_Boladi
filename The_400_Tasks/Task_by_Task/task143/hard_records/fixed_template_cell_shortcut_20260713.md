# fixed_template_cell_shortcut_20260713

- Live bar: manifest cost 502, SHA `363cf49fdad3b06148b7d66276dede93e9db72e873e532bb5e94fc3724dd9abc`, owner Boladi, refreshed with `python tools/ng_current_bar.py --task 143 --json` on 2026-07-13.
- Attempted route: replace the template-color ratio path (`W2idx / Widx`) with a direct read from one fixed top-left template cell, hoping to delete one scalar path.
- Exact failed gate: pre-ONNX corpus invariant check over bundled `task143.json` pairs. No top-left 3x3 position is always the template color; best cell `(0,0)` is template only 263/266.
- Why dead: a fixed-cell read would visibly miss at least 3/266 before hidden data. Per visible-fail discipline this representation family is dead.
- One remaining hypothesis: a cheaper non-fixed template-color selector would need to read a variable nonzero top-left cell without materializing a 3x3/vector mask; no such selector was found.
