live bar: manifest 1231, sha b2b7db986c3d9fb9b2743deb4589985f7e4c418ad7823b3e2b411baa7034a725; local parent predictor 1115.

attempted route: narrow final bit carrier from UINT32 to UINT16 (`task175_u16_bitcarrier.onnx`), because masks only use bits 0..10.

failed gate: `ng_gate visible --task 175 --cand ...task175_u16_bitcarrier.onnx` -> `ERROR: NotImplemented: Could not find an implementation for BitShift(11)`.

why dead: static predictor priced it at 1073 (-42), but ORT has no UINT16 BitShift kernel in this environment, so it is unshippable.

one remaining hypothesis: a U8/two-lane or non-bitset query encoding would need to overcome the extra mask/table params; do a Boolean-rank/query-mask proof before building it.
