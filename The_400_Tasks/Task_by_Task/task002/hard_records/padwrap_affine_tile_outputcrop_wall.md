# PadWrapAffineTile / output-crop wall

- Live bar used: 9413 bytes, SHA-256 `8c8340bca13986bd90f9831574d54e3e930935eacfec70fa9f830e455179029a`.
- Bar source: `python tools/ng_current_bar.py --task 2 --json` reported authoritative source `manifest`, owner Boladi, updated `2026-07-02T20:17:39Z`. The workbook/team-best 9397 entry is stale because it has the same live SHA but an inconsistent cost.
- Attempted route: replace the final 30x30 output padding/overlay or use Pad(mode=wrap)/affine tiling to avoid the incumbent `enclosed_full` bool pad.
- Exact failed gate: no new candidate was gated. The route was killed before staging because the only strict-shape variant outputs `[1,10,20,20]`, while `ng_gate visible` compares raw encoded tensors with expected shape `[1,10,30,30]`; alternate 30x30 reconstructions price above the 9413 bar.
- Incumbent sanity lines from this pass: `score_v2=(0, 8962, 451) cost=9413`; `[visible] 268/268 (100.00%)  PASS`; `profile=9413 (memory 8962 + params 451; 268 real inputs)`.
- Why dead: task002 is non-wrapping 4-connected exterior flood. Pad(mode=wrap) or affine tiling connects opposite crop edges and destroys the boundary/exterior meaning. Dropping the final 30x30 pad creates a raw-shape visible miss; preserving raw shape needs either slicing/carrying an F32 `[1,10,20,20]` canvas (16000 bytes) or rebuilding a 30x30 one-hot canvas/mask, both larger than the 900-byte bool pad being targeted.
- One remaining hypothesis: only a byte-equal micro-fold below 9413 remains plausible, such as a new value-CSE/static audit hit not covered by the existing floor proof; no PadWrapAffineTile rewrite is stageable on this function.
