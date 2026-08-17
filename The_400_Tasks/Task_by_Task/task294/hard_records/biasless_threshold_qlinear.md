# biasless_threshold_qlinear

Status: retracted/superseded by `task294_biasless_u8_threshold_cost603`.

- live bar: 614 profile/cost, sha256 `826a019cb4f85c90b5e81d087c84478d522248ba25aa47468b72161d06254164`, source `python tools/ng_current_bar.py --task 294 --json` / team manifest.
- prior failed gate: `task294_biasless_u8_qlinear.onnx` failed `[visible] 0/265 (0.00%)  FAIL`.
- correction: the failure was the ch5 kernel, not output-scale thresholding. The old negative-orthogonal ch5 weights painted off-crop cells adjacent to active black. V2 uses the symmetric separator `center + 4*orthogonal - 4*diagonal >= 5`.
- new proof: `task294_biasless_u8_qlinear_v2.onnx` passed visible 265/265, truth 200000/200000, full A/B 200265/200265 byte-equal, profile cost 603.
- remaining hypothesis: below 603 requires reducing the 500-byte F32 crop + U8 carrier memory, not more scalar/bias shuffling.
