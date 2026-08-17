# task347 bias-fuse one-cost shave

- live bar: cost 67, sha256 b8dd2ef6fbfc354734f4146dd5732fa8e576ee8242e622cb4a71335c7dd5fb43, source `python tools/ng_current_bar.py --task 347 --json` / team manifest.
- attempted route: remove first Conv scalar bias by encoding `+1` as `sum(left_all_channels)` in the fixed active 3x6 window.
- candidate: `The_400_Tasks/Task_by_Task/task347/prototypes/bias_fuse_probe/task347_bias_fuse.onnx`, sha256 404b1de9dec85e0d01fb182cb18d15c8cf0b9c242f01d049b0d13b1e06a96bdd.
- gates: visible 269/269 PASS; profile static=66 profile=66; explicit A/B vs `Team_Best_Onnx/task347.onnx` n=2000 byte_equal=True, divergences=0; ng_safety PASS 0 warnings.
- why dead for this prompt: it saves exactly one cost unit by deleting initializer `v_w1`; the task prompt asks for a real floor-break audit and says no 1-byte tie-shave, so this should not be re-staged as the task347 win.
- one remaining hypothesis: find a different output writer that avoids the 10-element final channel-expansion Conv vector or removes/narrows the F32 `[1,1,3,3]` intermediate without materializing a charged small one-hot.
