# task165 — CarrierNarrowEqualTail priced wall

- Live/team-best bar: **1,641** (`memory=1,538`, `params=103`), SHA-256 `6faf9093ffe11c0fbe8583054205b9ee4ddc36b6f7c4d6c31222055a8d5b43a9`.
- Bar source: authoritative team manifest, reconciled by `python tools/ng_current_bar.py --task 165 --json`; workbook agrees. The `1,634` value attached to the same SHA is stale seven-parameter accounting drift, not a lower model.
- Attempted route: the prompt's one-channel carrier + U8/QLinearConv + Equal-tail fast probe, audited against the current live graph before building.
- Exact failed gate: **fast-probe price/structure gate**. The live graph has already deleted the legacy 10-channel/F32 spatial carrier and repeated Conv masks. Its only full spatial intermediate is `fill30`, BOOL `[1,1,30,30]` = **900 B**, feeding terminal `Where(fill30, marker_vector, input)`. A terminal Equal needs a U8 `[1,1,30,30]` color-index canvas = **900 B** before any detection cost, so it cannot reach the advertised `~720` floor. Per the prompt kill rule, no candidate, truth, AB, full, or safety gate was run.
- Why dead: ARC-GEN UUID `6d58a25d` preserves arbitrary scattered input pixels outside the data-dependent vertical fill. `Where`-as-output is therefore the correct identity-substitution tail. A terminal `Greater` can emit only the fill mask, while a terminal `Equal` cannot also copy the arbitrary ten-channel input. Cropping does not help: a 20x20 mask plus required 30x30 Pad charges `400+900`, and Equal-before-Pad charges a 4,000-B 10-channel BOOL tensor.
- Remaining hypothesis: find a genuinely single terminal ONNX operator that fuses threshold comparison, dynamic marker recolor, and arbitrary raw-input passthrough, or regenerate the scattered input from packed bitsets cheaply enough to remove passthrough. No applicable legal operator is currently known; this is not a carrier-narrow/Equal-tail variant.

Live proof lines:

```text
score_v2: (0, 1538, 103)
profile_real_cost --arcgen 266: task165 task165.onnx: static=1641 profile=1641 (memory 1538 + params 103; 265 real inputs)
visible: 265/265 (100.0%, pass=true)
```
