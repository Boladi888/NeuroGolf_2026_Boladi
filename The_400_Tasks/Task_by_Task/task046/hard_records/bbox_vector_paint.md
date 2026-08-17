live bar: 2278, sha a5e2b2cd6adcfd3bdb0826b4ce68d557205f6428c410188d4569a57a81300509.

Attempted route: prompt-suggested bbox-vector paint / separable rectangle formulation, checked by rank-testing the colored foreground mask of task046 outputs before building an ONNX family.

Failed gate: representation probe on officials. Command read `data/competitions/neurogolf-2026/task046.json` and computed binary matrix rank of `output != 0`: `pairs=267 rank_gt1=264 max_rank=3`, first miss `('train', 0, (3, 7), 9, 2)`.

Why dead: task046's decoded rule is snake reassembly, not rectangle/bbox painting. The colored output mask is usually non-separable, so row/col bbox vectors would fill cells that must remain background; a visible miss would be certain hidden 0.

One remaining hypothesis: a vector formulation may still exist for the known snake columns/rows, but it must encode the path mask and segment recoloring, not a bbox rectangle paint.
