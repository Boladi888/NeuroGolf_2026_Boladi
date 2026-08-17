# task356 sparse ConvInteger weight dead route

- Live/team-best bar: 1220 profile_cost, SHA `e81463c0cf23b583f80f00c8224100f082a46eeab21f8ea5460ef26e28c3aedc` from OnnxlyFans manifest/workbook.
- Attempted route: keep the exact Vamshi 1220 graph but store the final `ConvInteger` weights as a sparse initializer (`task356_sparse_weight_probe_4866BA6F.onnx`, SHA `4866ba6ff7ca03228958d15d35ad8cc9d1b49ddb6abab7ce99b1bd208fbaeba8`), hoping to reduce final-writer params from 10 to 2.
- Cost/check gate: `python -c "from tools._score_v2 import score_network_v2; ..."` -> `None None`.
- Failed checker detail: `onnx.checker.check_model(full_check=True)` and strict shape inference both return `[ShapeInferenceError] Inference error(s): (op_type:ConvInteger): [TypeInferenceError] inputs are expected to have tensor type and output type should not be null.`
- Failed visible detail after scorer sanitization: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_13' is not a graph input, initializer, or output of a previous node.`
- Why dead: sparse initializers are not treated as legal tensor inputs by the scorer/checker path for `ConvInteger`; adding value_info would bill the dense 10-byte weight shape and still not beat the bar.
- One remaining hypothesis: a non-sparse legal final projection with fewer than 12 params would need an op that broadcasts two signed channel roles into the exempt graph output without materializing a counted two-channel or ten-channel canvas.
