live bar: team-best 1099, sha f1dcb5ae9eb812c8a0da94f6764081cf2af12d38da3a949ae59183b6d7a308cd
attempted route: replace dense 300-param final `class_col_bits` UINT32 initializer with a sparse initializer holding only its 30 nonzero channel/column masks.
failed gate: `python tools/ng_gate.py visible --task 160 --cand The_400_Tasks/Task_by_Task/task160/prototypes/probe_variants/task160_sparse_class_mask_probe.onnx` -> `InvalidArgument: Invalid model. Node input 'safe_name_38' is not a graph input, initializer, or output of a previous node.`
why dead: NeuroGolf sanitization/ORT does not expose the sparse initializer as a legal tensor input for the final `BitwiseAnd`; it fails before visible examples run.
remaining hypothesis: a non-sparse decoder win would need a different final representation that avoids a charged full mask tensor, not the same `BitwiseAnd` with sparse storage.
