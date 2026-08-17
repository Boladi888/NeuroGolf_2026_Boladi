# task074 sign-relation timeout probe

Live/team bar used: 202, profile 202, SHA-256
`fdf348b26c90ffa1ee70ba94f20d7b329a571788f881c6a6780e68e3a35f5018` from
`python tools/ng_current_bar.py --task 74 --json` (`authoritative_source=manifest`,
owner Boladi, updated 2026-07-09T04:14:47Z).

Attempted route: replace the current one-hot bit relation `H2[2,2,2]` with sign
features and a 2x2 equality/all relation. Static score improved from 202 to 198:
`task074_sign_relation_worker3.onnx` SHA
`5057add0335693525fe46d5a9739f832d6c085a1d846e128a96a308ef5b122e2`; hybrid
single-control variant SHA
`4e27023bf13b5378aca1bb83aebb31f797c2e6a965698b0efa940a74bd472247`.

Exact failed gate:
- `python tools/ng_gate.py visible --task 74 --cand The_400_Tasks/Task_by_Task/task074/prototypes/task074_sign_relation_hybrid_worker3.onnx --json --out The_400_Tasks/Task_by_Task/task074/prototypes/task074_sign_relation_hybrid_visible_timeout.json`
- `[ng_gate] wall budget 900s exceeded for visible task074; exiting to avoid orphaned gate work`

Why dead: although the graph is still one output-writing Einsum and scores
`(0, 0, 198)`, the sign-dot factorization couples endpoint coordinates too early
for ORT's left-deep path. One dummy ORT run was ~42-45 s versus ~0.04 s for the
202 parent, so visible cannot finish inside the shared gate budget. Do not run
truth/ab/battery on this family.

One remaining hypothesis: sub-202 needs a factorization that preserves the parent
contraction shape: input coordinates must reduce to small bit labels before any
output-coordinate expansion. Sparse Pcoord/channel/H2 variants remain blocked by
the documented sparse-Einsum scorer failure.
