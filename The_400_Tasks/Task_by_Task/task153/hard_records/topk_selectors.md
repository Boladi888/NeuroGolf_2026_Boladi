live bar: manifest cost 564, sha 2dd55a478ba8f14ab444871ada98b3ae7e9c60fe5c32efbbb4b3724c4433d785.

attempted route: replace the current ArgMax+weighted-sum foreground color selector with TopK(k=2), either over pool_max or over pool_idx.

failed gate: pool_max direct `task153_topk2_direct.onnx` scored 545 but visible was `[visible] 0/265 (0.00%)  FAIL`; pool_max reversed `task153_topk2_reverse.onnx` scored 563 but visible was `[visible] 0/265 (0.00%)  FAIL`; pool_idx `task153_poolidx_topk2.onnx` scored 538 but visible was `[visible] 3/265 (1.13%)  FAIL`.

why dead: pool_max has a tie among background and the two foreground colors, so k=2 drops one true color; pool_idx ranks inactive-channel MaxPool tie indices and can select absent high-color channels.

remaining hypothesis: only revisit if the active-channel mask can zero background/inactive channels for less than the bytes saved by deleting the current color-sum selector.
