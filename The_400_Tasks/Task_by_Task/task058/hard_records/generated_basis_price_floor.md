# task058 generated coordinate basis price floor

Live bar: cost/profile 128, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Attempted route: replace the current stored coordinate basis `[q^i, q^-i, 1]` with zero-param `Range`/`Pow`/`ConstantOfShape` generation before the final analytic `Einsum`.

Failed pre-gate price check: the current stored `basis` costs 90 params. A minimal generated basis must materialize at least `Range(0,30,1)` (`[30]` F32 or INT64 intermediate), two generated power columns (`[30]` each), a generated ones column (`[30]`), and a stacked `[30,3]` basis or equivalent final operands. `zero_param_generators.md` prices `Range` as losing because the runtime output is charged; for F32 columns alone this is at least 120 B for the range plus 240 B for two power columns plus 120 B for ones, before concat/stack memory and scalar inits.

Why dead: deleting 90 stored params creates hundreds of charged runtime bytes, so this cannot beat the current static/profile cost 128. This is the prompt's zero-param basis route under the refreshed 128-byte team-best, not under the stale 19,826-byte graph.

One remaining hypothesis: a generated basis could only win if it fed directly into a free output op without materializing the coordinate columns. The current `Einsum` requires a concrete shared spatial basis operand, so that escape is not available here.
