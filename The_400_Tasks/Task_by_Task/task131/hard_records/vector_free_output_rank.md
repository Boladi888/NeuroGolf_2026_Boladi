live bar: 3922, sha256 b5bcdca3c64c903e9c6279cf6d332563d92bff83e8df5a36601f789cdfa10235 (ng_current_bar.py manifest)

attempted route: prompt-requested row/col boundary vector or free-output render probe using fresh ARC-GEN masks.

exact failed gate: 3000 fresh ARC-GEN rank probe. Shifted color-3 output mask ranks: rank1=83, rank2=683, rank3=2045, rank4=189. Cyan separator rank was always 1, but the moved pattern is not rank-1.

why dead: the moved color-3 bitmap is arbitrary connected 8/9-cell structure, so a boundary-vector-only render cannot carry the pattern. A conservative free-output term render needs active + marker + separator + 9 green terms = 12 row/col term pairs; row/col masks alone are 720 elements, bool+F16 cast masks are about 2160 B, and the current green sparse-read floor is 620 B memory + 465 params, for a lower-bound estimate of 3245 before scalar plumbing. This is above the prompt's 1400 target and too close to the 3922 scatter graph to justify a full build.

one remaining hypothesis: only a final op that consumes sparse coordinates directly without INT64 ScatterND indices or row/col one-hot masks could break this floor; no legal ONNX op found in the current routed library.
