# task279 hard record: omit_qlinear_zero_cast_bool

Live/team bar: 5,057, sha256 `4f9a0651cbb5bf227feb6d794dab4e84df585bd83cdfb7fa1a7e2928c70574f9`.

Attempted route: omit QLinearConv zero-point initializers, replace `Greater(flood_1, zero_u8)` with
`Cast(flood_1 -> BOOL)`, and delete `zero_u8` for a 1-param shave.

Exact failed gate: build/checker before score.

Failure lines:
`Node(degree_qconv_0) with schema(::QLinearConv:10) has input size 7 not in range [min=8, max=9]`.
After preserving arity with empty slots:
`Node (degree_qconv_0)'s input 2 is marked single but has an empty string in the graph`.

Why dead: in the checker schema used here, QLinearConv zero-point inputs are mandatory single inputs, not optional empty
slots. Keeping `zero_u8` for QLinearConv means the Cast-to-BOOL rewrite saves no params and no memory.

One remaining hypothesis: a same-cost `Greater -> Cast` swap can be used only as a semantic probe; it is not a cost win
unless QLinearConv zero-points become legally omittable.
