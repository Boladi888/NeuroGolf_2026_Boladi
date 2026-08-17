route: single Pool op as free output

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: replace the grouped Conv with one final `AveragePool(kernel=3x3,pads=1)` output node to avoid all Conv params and exploit the output tensor being free.

Exact failed gate:
- `task098_avgpool3x3_output_probe.onnx`: `ng_gate visible`: `[visible] 0/266 (0.00%)  FAIL`.

Why dead: a single pooling output is a monotone neighborhood aggregate. It cannot distinguish same-color interiors from boundaries, cannot emit the ch0 background/interior horizon, and lights near-object cells that must stay zero under the raw one-hot gate.

Remaining hypothesis: only a final op that can express signed local predicates per channel without a charged mask plane can compete with the current 270-param grouped Conv.
