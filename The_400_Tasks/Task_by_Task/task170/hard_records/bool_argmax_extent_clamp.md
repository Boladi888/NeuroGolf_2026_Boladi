Live bar: 1623 from `ng_current_bar.py --task 170 --json`, live SHA `a0463f0c05f62fe4c2cc1d0f34a7494a29a5a2517fea9edad643ef8e6ce0a4bd`.

Attempted route: delete `rowu`/`colu` UINT8 clamp tensors by feeding `rowb`/`colb` BOOL directly to the four extent `ArgMax` nodes in the current `t170_probe_poly` graph.

Failed gate: `python tools/ng_gate.py visible --task 170 --cand The_400_Tasks/Task_by_Task/task170/prototypes/task170_bool_argmax_probe.onnx` -> ORT `InvalidGraph`: `Type 'tensor(bool)' ... of operator (ArgMax) ... is invalid.`

Why dead: opset 16 ORT does not accept BOOL `ArgMax`; the 60-byte static saving cannot be realized without a different non-BOOL clamp tensor, and F32 clamp alternatives cost more than the existing `Greater -> Cast(UINT8)` path.

One remaining hypothesis: find a legal non-ArgMax extent detector that consumes the existing F32 `rowsum`/`colsum` without materializing an extra 30-element clamp, but price it before building because any F32 clamp immediately loses the 60-byte target saving.
