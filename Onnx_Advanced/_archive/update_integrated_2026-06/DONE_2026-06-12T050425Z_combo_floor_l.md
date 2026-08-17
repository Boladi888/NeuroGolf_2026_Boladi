## Combo-head quotient recovery

**What I discovered:** when a free-input `Einsum` combo head encodes a row as
`base + K * feature`, and the base term is generator-bounded below `K`, the feature can be
recovered from the combo maximum with `Floor(Max / K)`. This can remove a separate selector
reduction for the feature.

**Where:** task256, worker-8 v13.

- Builder: `The_400_Tasks/Task_by_Task/task256/prototypes/build_task256_v13_floorl/build_task256_v13_floorl.py`
- Staged candidate: `staged_submits/task256_v13_floorl_cost1328/`
- Previous v12 cost: 1334
- New v13 cost/profile: 1328

**Why it matters:** v12 used two free-input `Einsum` heads:

- `combo = row_total + 64 * red_count` as `[1,30]`
- `L = red_count` as a scalar selector reduction

For task256, the combo maximum is `W + 64L`, with `W <= 13`. Since `W < 64`,
`Floor(Max_f / 64)` exactly recovers `L`. The rewrite removes the second `Einsum` output
and the 10-element `sel2` initializer, adding one scalar Div and one scalar Floor:

- memory: `-4 + 4 + 4 = +4`
- params: `-10`
- net: `-6`

Gates on staged candidate:

- visible: 266/266
- truth full: 200,000/200,000
- AB full: 200,266/200,266 byte-equal, 0 divergences
- score/profile: 1328 static, 1328 real-input profile

**What's missing from the library:** §42 covers free-input `Einsum` marginals and task256
already documents the combo-head idea, but I did not find a routed note for eliminating a
separate selector by quotienting a biased combo maximum when the residual is bounded below
the scale.

## Sorter note: techniques.md §42 — extend existing free-input Einsum marginals section with combo-head quotient recovery
