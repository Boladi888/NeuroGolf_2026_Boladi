# Rank-Reduced Spatial Factor Probes

- live bar: 118 bytes, manifest/team-best SHA-256 `5121f6312e9e3bd6ee13b5b4a91bdd98dd1ec249358a67bf65a63fea07ff3f7e`
- attempted route: replace the current `Q[30,3] + K[2,3,3] + s[10]` one-node `Einsum` with lower-param spatial factors while keeping the same sign rule over all valid 3x3 foreground masks with foreground count 2..8.
- current parent structure: static/profile cost 118 = memory 0 + params 118; `Q[30,3]` 90 params, `K[2,3,3]` 18 params, `s[10]` 10 params.
- failed probe A: direct one-axis matrix `C[30,3] + s[10]`, projected cost 100. Optimized over all valid 3x3 masks; best result retained `3456` sign violations, min margin `-0.06384745446266028`, `a=0.38305841524960865`. No ONNX built.
- failed probe B: current-style `r=2, M=2`, projected cost 78. Analytic-gradient search with 61 starts retained thousands of violations; representative final/best-band results stayed around `bad ~= 3598..3837`, no zero-violation solution. No ONNX built.
- failed probe C: current-style `r=2, M=4`, projected cost 86. Best recorded run retained `3583` sign violations, min margin `-0.16029823635626572`, `a=0.26854024586206204`. No ONNX built.
- failed probe D: current-style `r=2, M=8`, projected cost 102. Best recorded run retained `3595` sign violations, min margin about `-0.14999980000220287`; final best-by-margin retained `3795` violations. No ONNX built.
- why dead: the self-product row/col decision needs at least two separable spatial components and the 3-wide basis used by the current graph. The tested lower-rank families cannot separate the AND-vs-complement sign pattern even before ONNX construction.
- one remaining hypothesis: a different channel-sign fold could save the `s[10]` vector by reusing an existing basis operand, but it must still handle foreground color 9 and keep output rows/cols >=9 nonpositive.
