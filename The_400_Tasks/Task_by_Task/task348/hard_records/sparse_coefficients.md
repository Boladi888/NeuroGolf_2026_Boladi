# task348 sparse coefficient initializer dead route

- Live bar used: 311, SHA `6c61487d7c0d2b921204a300a6a49db146c23f30fe57f53cec2d5defa135af7f` from `python tools/ng_current_bar.py --task 348 --json`.
- Attempted route: convert the sparse dense coefficient tensors in Sho's one-Einsum graph (`v_w2`, `v_w3`, `v_w4`, `v_w7`, `v_w8`) to ONNX sparse initializers to reduce params.
- Exact failed check: `score_network_v2('The_400_Tasks/Task_by_Task/task348/prototypes/task348_sparse_coeff_probe.onnx')` returned `None`; ORT also emitted "Could not write a profile because no model was loaded."
- Why dead: sparse initializers as `Einsum` operands are not legal through the NeuroGolf scorer path, matching `core.md` sparse-initializer warnings.
- One remaining hypothesis: keep dense initializers but find further selector reuse or a lower-rank terminal `Einsum` factorization.
