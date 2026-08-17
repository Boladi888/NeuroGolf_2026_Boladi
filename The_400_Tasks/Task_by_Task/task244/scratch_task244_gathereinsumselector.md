# task244 scratch - Gather/Einsum selector shave

Current bar used: 85 cost from workbook row and team manifest, sha256 db55e34c9cd94f6da4514281bdd2a377d1b4783c59a83e8d62f09353a60f175e, updated 2026-07-02T02:35:15Z. Static score on the team-best ONNX is `(macs=0, memory=44, params=41)`.

Decoded rule: active inputs are square lattice canvases with widths 8, 11, 14, 15, 17, 19, or 23. Sample the top-left cell color of each lattice cell, horizontally flip the sampled cell grid, and emit it at the top-left of `[1,10,30,30]`. Width 11 has two classes: 3x3 unless the row-2 divider makes it the 4x4 class.

Team-best representation: one scalar high-degree `Einsum` computes a row/color fingerprint, `Greater` against 8 thresholds plus `ArgMin` chooses one of 8 ROI rows, and `RoiAlign` writes the sampled/padded output directly as the graph output. Charged tensors are scalar `r11` (4), bool threshold vector (8), U8 threshold vector (8), bucket scalar (8), and ROI vector (16). Charged params are thresholds (8), ROI table (32), and batch index (1).

Falsification cases:
- visible miss on any official/train/test/arc-gen pair kills the representation family.
- score/profile cost >= 85 is only a probe or tie, not a staged win.
- if a scalar bucket approximation misclassifies width 11 3x3 vs 4x4, stop that bucketizer.

Candidate families to test:
- Legality shave: remove the Bool-to-U8 cast before `ArgMin`, or use F16 ROI rows. Estimate: 77 if accepted, but ORT/type constraints may reject Bool ArgMin or mixed F32/F16 RoiAlign inputs.
- Scalar bucket math: replace the 8 threshold vector with scalar `Log`/affine/Cast bucketization if it separates the 8 fingerprint bands. Estimate: 75-79 if no clamp is needed; risk is hidden misbucketing because fingerprint ranges vary within widths.

Remaining floor if both fail: the 85-byte graph is close to the static floor for this RoiAlign route; one remaining hypothesis is a smaller ROI-table encoding, but any formula must save more params than it adds scalar intermediates.
