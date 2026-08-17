route: drop ch0 / all-zero background decode

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: output only the 3x3 same-color boundary detectors on channels 1..9 and leave channel 0 all-zero, relying on all-zero cells to stand for color 0.

Exact failed gate:
- `task098_noch0_color3x3_cost90.onnx`: `profile_real_cost.py`: `static=90 profile=90 (memory 0 + params 90; 266 real inputs)`.
- `ng_gate visible`: `[visible] 0/266 (0.00%)  FAIL`.

Why dead: the official expected output is raw one-hot after NeuroGolf's `>0` decode; active color-0 cells and rectangle interiors require channel 0 positive. All-zero means no color, not black, under the raw one-hot equality gate.

Remaining hypothesis: any winning route must explicitly compute/preserve ch0 for active background and zeroed interiors while keeping off-grid padding all-zero.
