# task261 scratch plan

Worker: bo_ng_cdx2_worker2
Date: 2026-07-04

Authoritative bar used before build: cost 122, SHA-256 `61c58f8fbb942e5a13ed755b3fdc0f7e68c58c405f091240a235d4b0a85359be`, source `python tools/ng_current_bar.py --task 261 --json` / team manifest.

Decoded rule: keep the same active grid size. Every input color-8 cell is translated down by one row and recolored to color 2. Cells shifted beyond the bottom disappear. No other foreground colors appear in the task examples or ARC-GEN samples.

Output-size relation: output has the same H x W active shape embedded at top-left of the 30 x 30 one-hot canvas; padding remains all-zero.

Falsification cases:
- A color-8 cell in the bottom active row must vanish, not wrap and not paint padding.
- A top-row run shifts to row 1 and keeps its columns exactly.
- Sparse disconnected 8s and solid 8 blocks should shift independently with no connected-component dependence.

Candidate families to price:
- Static down-shift/recolor via `Slice`/`Pad`/`Gather`: likely too expensive because reading a F32 channel or full one-hot shift materializes at least one 30 x 30 plane or full canvas.
- Free-output separable `Einsum`: output is written directly, so only initializer params count. Current team-best uses this family at 122 params; inspect for polynomial/factor redundancy and output-channel/ramp factor compression.
- Conv/ConvTranspose/QLinearConv route: dead for this pass unless it can beat 122; even one small kernel plus bias is unlikely to beat the existing output-free separable graph.

Charged structure intended to delete: redundant initializer factors in the current 122-param free-output `Einsum`, especially any row-shift polynomial coefficient blocks or channel sign vectors that can be merged without materializing a charged intermediate.
