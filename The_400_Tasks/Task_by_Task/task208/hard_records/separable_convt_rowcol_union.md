# Dead route: separable ConvTranspose row/col union render

- **Live bar:** 3759, manifest authoritative, owner Aditya, sha256
  `460d7e10b614fb8cd16a82fd7567f63d166c287b56bfc0b5a336f7fb356aa73b`,
  updated 2026-06-14.
- **Attempted route:** keep the live detector through `bg_score17`, reduce the
  two 16x16 placement anchors to independent row and column seed vectors, render
  row-inside/row-border and col-inside/col-border with F16 `ConvTranspose`, then
  reconstruct the 19x19 frame as
  `row_border x col_inside OR row_inside x col_border`.
- **Prototype:** `prototypes/task208_bo_ng_cdx2_worker3_separable_convt_union_probe.onnx`,
  sha256 `4ca09ca723a4f45b3fdb4dc82c1592338e0b2c309be47fe4251bc5b5afe1c62c`.
- **Cost/profile:** static 5794, profile 5794 (`memory 5693 + params 101`);
  delta vs live = +2035, so it is a cost loser even before correctness.
- **Exact failed gate:** `ng_gate visible --task 208 --cand ...` ->
  `[visible] 15/266 (5.64%)  FAIL`.
- **Why dead:** the two cutout placements are data-dependent 2-D pairs. Reducing
  `bg_score17` independently over rows and columns loses the pairing, so diagonal
  placements create cross-products/extra frames. The F16 ConvTranspose factoring
  also materializes three 19x19 F16 outer-product surfaces, so it cannot beat the
  current 722-byte U8+BOOL render pair even if it were exact.
- **One remaining hypothesis:** only an exact paired-anchor formulation could
  revisit this route, but preserving row/col pairing likely requires a 2-D seed,
  TopK/arg split, or per-placement surface, which reintroduces the mask charge.
