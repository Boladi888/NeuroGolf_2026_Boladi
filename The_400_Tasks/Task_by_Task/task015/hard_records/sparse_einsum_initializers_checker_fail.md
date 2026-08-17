live bar: 431, sha256 94c5e8748d1d756fa21a7094f8212625b3bfb73e7a764f4bec0b77eb370635b7 (ng_current_bar.py manifest, 2026-07-09T01:09:17Z)
attempted route: convert Sho's 431 single-Einsum dense initializers to ONNX `graph.sparse_initializer`; dense params would drop from 431 logical elements to 110 stored nonzeros if legal.
failed gate: `onnx.checker.check_model(model, full_check=True)` before scoring/visible -> `ShapeInferenceError (op_type:Einsum): Rank of input 1 (0) does not match the equation indices (2)`.
why dead: ONNX strict shape inference does not expose the sparse initializer's dense rank to `Einsum`, and NeuroGolf scorer runs strict shape inference before accepting a model.
one remaining hypothesis: only a normal dense-initializer single-Einsum skeleton or a different legal final op can beat 431; sparse initializers are not a legal shortcut for this graph.
