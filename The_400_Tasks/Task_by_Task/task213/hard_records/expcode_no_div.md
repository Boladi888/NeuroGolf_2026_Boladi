live bar: 879 @ d4f6ee3d715fd6d664f903ae71cfcd9297d4de6481a48c91e0e0fad910d27c36; task-local staged bar: 874 @ 131b8906c36ef401972f1dab4d4475b14a9800eaf245bbf60a4defb97f12e543
route: replace `32**color -> Log -> Div(ln32) -> Cast(UINT8)` with `exp(color) -> Log -> Cast(UINT8)` to delete `Div`, `ln32`, and `id_float`.
failed gate: visible on `prototypes/task213_worker2_expcode_no_div.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
why dead: the anchor-product code includes a variable positive multiplicity `m`; current base-32 division uses `floor(color + log_32(m)) == color`, but no-div exp coding yields `floor(color + ln(m))`, so channel ids shift with the number of live anchor pairs.
remaining hypothesis: a no-div code would need a second compact range/binning mechanism; plain exponent-base changes are dead because Equal needs one stable id per channel.
