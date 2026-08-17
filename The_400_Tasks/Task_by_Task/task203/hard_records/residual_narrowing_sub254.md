# task203 residual narrowing below 254

Live bar and sha:
- Authoritative live bar: 272 from `python tools/ng_current_bar.py --task 203 --json`, source `manifest`, sha256 `35d5df71d08cdaa3f496cf197358c57ff05bebe1a4670749ceb4271212fb1397`, updated `2026-06-08T23:23:49Z`.
- Effective task-local threshold: staged `task203_u8chain_cost254`, sha256 `ea18293d1f446c09f75d212faffa9cb0826c31600e5feb7e4fd758b2d77195e3`, cost/profile_cost 254.

Attempted route:
- Narrow residual v7 plumbing below 254 without changing the Path-B channel-permutation representation.
- Priced levers: constant-fold `rank_limit`; remove `ax023` by lowering opset and using `ReduceSum` axes attribute; remove the final U8->I32 cast by making `rank_to_color` INT32.

Exact failed gates / probes:
- Corpus invariant probe for constant `rank_limit=10`: failed before build. On 200000 ARC-GEN inputs, `rank_limit` values were `[4, 5, 6, 7, 8, 9, 10]` with counts `{4:28666, 5:28488, 6:28612, 7:28558, 8:28747, 9:28758, 10:28171}`.
- `build_task203_worker2_probe.py` opset attr-axis probe:
  - opset 12: `score_network_v2` failed with `ShapeInferenceError (op_type:Add): unsupported type tensor(uint8)`.
  - opset 13/14: checker failed with `ValidationError: Unrecognized attribute: axes for operator ReduceSum`.
- `task203_worker2_i32_rank_to_color.onnx`: valid but losing, `score=(0, 250, 27) cost=277`, so deleting the final cast through an INT32 inverse table costs +23 vs staged 254.

Why dead:
- The remaining charged vectors are forced by ORT/scorer dtype rules: `counts_f F32[10]`, `scatter_rank I32[10]`, `source_rank I32[10]`, and final `idx I32[10]`.
- `Gather`/`ScatterElements` indices cannot be sub-INT32 (traps §8c), and the only initializer shave (`ax023`) is blocked by the opset needed for U8 Add/Sub/Min.
- Valid cast deletion makes `rank_to_color` and `idx_raw` INT32, which costs more memory than the deleted cast.

One remaining hypothesis:
- A different representation that uses the final free output op directly, not this two-step inverse-permutation vector, would be required to beat 254. Within the v7 U8 rank-chain formulation, no priced residual narrowing is below 254.
