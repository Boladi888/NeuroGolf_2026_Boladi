# task245 bounded codebook decode route

- Live bar: manifest cost 940, sha256 `1a7e75f1764b6283cf5a4cb8030bdfd272ffc43bad23737d94f493c73eab3ca5`; local staged rung to beat: 939 (`task245_expdecodeavoidance_cost939`).
- Attempted route: `prototypes/build_task245_codebook_decode.py` / `task245_codebook_decode_probe.onnx`; replaced the four `Log -> Mul -> Sub` position decoders with finite F16 threshold vectors plus `Less -> Cast -> ReduceSum`.
- Gates: visible passed with `[visible] 267/267 (100.00%)  PASS`; profile then lost with `task245 task245_codebook_decode_probe.onnx: static=1160 profile=972 (memory 851 + params 121; 267 real inputs)`.
- Why dead: each bounded decoder saves scalar log arithmetic but adds threshold params plus charged bool/F16 threshold vectors; the net profile is +33 bytes versus the 939 staged rung, so full truth/A-B would be pure burn.
- Remaining hypothesis: a codebook decode can only win if it removes a shared carrier such as `top_weight30_f`/color vectors or reuses one vectorized threshold result without extraction tensors; the per-axis threshold decode is cost-dead.
