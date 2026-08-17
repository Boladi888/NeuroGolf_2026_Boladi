route: direct sub-270 grouped Conv

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: shrink the accepted one-node grouped Conv below the `[10,1,3,9]` dense kernel by dropping the farthest right-look column, then search for any custom dense grouped Conv footprint with `10*area < 270` (and optional bias where total params still beat 270).

Exact failed gate:
- `task098_trim_rightlook_3x8_cost240.onnx`: `profile_real_cost.py`: `static=240 profile=240 (memory 0 + params 240; 266 real inputs)`.
- `ng_gate visible`: `[visible] 221/266 (83.08%)  FAIL`.
- `task098_search_direct_conv.py`: `no sub-270 direct grouped Conv footprint found; checked 9057`; only separable footprints on official pairs were area-27 `3x9` and `9x3`, both cost 270.

Why dead: color channels need local 3-row neighborhood to suppress same-color interiors, while ch0 must see background at offset +7 for the leftmost interior cell of an 8-wide rectangle. A dense one-op grouped Conv therefore needs a 3x9 or 9x3 bounding footprint; smaller footprints fail visible or are not linearly separable on the official local patterns.

Remaining hypothesis: a non-Conv final op with no charged full-plane intermediates, or a future scorer-supported sparse/structured weight encoding, would be needed to beat 270.
