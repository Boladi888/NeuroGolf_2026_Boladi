# task066 worker2 notes

## Bar used

Current team-best bar is 2707 from the OnnxlyFans manifest and workbook:
`task066`, creator `Aditya`, SHA-256
`6b0b5fa45a44833d072bae815f1b489cb383ad259b5791001285338c49dc4753`,
updated `2026-06-17T21:58:42Z`. A real-input profile of that ONNX reports
`static=None profile=2707 (memory 2431 + params 276; 266 real inputs)`.

## Rule scratch

Input has background 0, walls 8, one adjacent color-3 source domino, and one
adjacent color-2 target domino. Source and target share orientation. The output
keeps the same grid size and paints a 3-valued orthogonal path from source
toward target. The bend is the unique wall-bounded row/column where the source
leg and target-side end are adjacent to color 8 and the source/cross/target
clearance checks see no color 8.

Output-size relation is identity: active HxW input produces HxW output embedded
in the normal 30x30 NeuroGolf one-hot canvas.

Falsification cases from ARC-GEN/visible:
- Horizontal source/target case where canonical transpose must swap row/column
  logic before drawing.
- 20x20 case with bend at inner coordinate 18; cannot shorten the inner vector
  below 18 without losing wall-at-19 cases.
- Case where target-side column/row clearance is load-bearing; dropping it was
  already documented as visible/arc-gen failing.

## Candidate families considered

- Full structural wall-plane: current historical 11,769 profile; robust but far
  above the 2707 bar.
- Mask/Where tail from segment vectors: needs at least one 30x30 mask or INT32
  outer product, so it loses to the accepted final free-output Einsum tail.
- Byte-equal micro-shave of current free-output graph: best narrow path. It
  avoids changing behavior and removes one duplicate initializer.

Dominant charged tensors in the current team-best profile are the final row/col
selector casts (`rows4_f`, `cols4_f`, 480 bytes each), `rows4_u8`/`cols4_u8`
at 120 bytes each, and the two dynamic wall-line F32 slices at 72 bytes each.
The staged candidate does not reduce memory; it reduces params 276 -> 258.

## Attempts

Staged: `task066_idxshare_cost2689`. Reuses `idx_inner_vec_2d` for the
positive target-side scan and changes `bpos_r2_arg_pos` to `axis=1`, removing
the duplicate `idx_inner_vec` initializer. Profile cost is 2689, an 18-byte
improvement over the manifest bar. Full A/B against the manifest ONNX is
byte-equal over 200,266 inputs.

Failed: first index-sharing save cleared value_info. Visible passed, but
profile scoring returned `score_network returned None`; preserving the parent
dynamic Slice value_info fixed it.

Failed: removing `rows4_f`/`cols4_f` casts and feeding UINT8 row/col selectors
directly to the final `Einsum`. ONNX checker accepted the graph, but ORT load
rejected mixed FLOAT/UINT8 `Einsum` inputs.

## Remaining floor hypothesis

The credible remaining floor is small. Most memory is tied to the final
free-output tail: two 4x30 F32 selector casts are 960 bytes, but replacing them
with a mask/Where tail introduces a 30x30 mask or worse. Further savings likely
come only from param shaving in the selector/rank initializers or a different
free-output factorization that keeps byte-equality to the manifest graph.
