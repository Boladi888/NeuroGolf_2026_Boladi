# sparse_initializers dead route

Live/team-best bar: 488 cost, sha256
`10a94404614387089c8f0c6f2a4ea399c3ee0b8f857cc9d178be70d0c3313e85`
from `python tools/ng_current_bar.py --task 126 --json` (`manifest`, owner Vamshi,
updated 2026-07-02T02:35:15Z).

Attempted route: replace dense initializers in the 488 ScatterND graph with sparse
initializers to save params without changing memory:
- `channel_selector` F32[10] one-hot -> one stored sparse value (target 479).
- `prefix_idx` INT64[2,2] -> one stored sparse value (target 485).
- `zero_f` false-branch zero -> empty or length-one sparse zero (target 487).

Exact failed gate: build/checker/shape-inference before scoring.

Failure lines:
`(op_type:Einsum, node name: column_sums): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (1).`
`(op_type:Concat, node name: indices): [ShapeInferenceError] axis must be in [-rank, rank-1].`
`Sparse tensor (zero_f) must have a dense-rank > 0`
`(op_type:Where, node name: signed_center_updates): Y typestr: T, has unsupported type: sparse_tensor(float)`

Why dead: this scorer/checker stack does not treat sparse initializers as normal
dense tensors for the relevant `Einsum`, `Concat`/ScatterND index path, or `Where`
branch. Adding dense value_info to rescue shape inference would reintroduce memory
larger than the params saved.

One remaining hypothesis: only a dense-op rewrite that removes the `[2,30]` F32
updates or the INT64 `[2,3]` indices can beat 488; sparse initializer accounting
is not available on this graph.
