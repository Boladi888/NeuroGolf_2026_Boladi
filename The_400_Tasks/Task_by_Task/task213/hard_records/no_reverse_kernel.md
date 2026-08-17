live bar: 879 @ d4f6ee3d715fd6d664f903ae71cfcd9297d4de6481a48c91e0e0fad910d27c36
route: delete `Slice(step=-1)` before the ConvInteger dynamic kernel to save one 70-byte U8 lane.
failed gate: visible on `prototypes/task213_worker3_no_reverse.onnx` -> `[visible] 25/265 (9.43%)  FAIL`.
why dead: ConvInteger width indexing reads kernel positions in reverse under the required padding, so the dynamic kernel must be reversed; deleting it reverses/garbles the output.
remaining hypothesis: only a different output assembler that avoids ConvInteger's width-index reversal could remove this lane.
