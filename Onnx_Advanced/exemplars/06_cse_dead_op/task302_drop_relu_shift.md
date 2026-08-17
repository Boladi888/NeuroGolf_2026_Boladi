# task302 — drop_relu (shift16 variant)

| Field | Value |
|---|---|
| Task | task302 |
| Pattern | Variant of `dead_relu_post_maxpool` |
| Reference | Worker-staged ~2,496 bytes saved |

A variant of the dead-Relu pattern (technique 16) on a different op chain shape — `Relu` after a `shift16`-style sub-grid alignment + subtraction.

Same principle as task093: when the input to Relu is provably ≥ 0, Relu is identity and can be dropped.

## Cross-ref

- `06_cse_dead_op/task093_dead_relu_post_maxpool.md` — the canonical example.
- `techniques.md` §16.
