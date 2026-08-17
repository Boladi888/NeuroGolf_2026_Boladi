live bar: 1574, sha d785ffe94a9e8b0147f9a7d3ce79f77d4056c4f8a54c8066d2fc268f1ed32276 (`python tools/ng_current_bar.py --task 46 --json`, manifest authority, owner Sho).

Attempted route: metadata/opset trim of the Sho live graph, using older attribute-form `Unsqueeze` axes and, at opset 10, attribute-form `Pad` pads. Prototype builder: `The_400_Tasks/Task_by_Task/task046/prototypes/build_task046_v33_opset_attr_trim.py`.

Failed gate: ONNX checker/shape-inference before visible. `--opset 11` fails `MaxPool(UINT8)` (`unsupported type: tensor(uint8)`). `--opset 10` fails `Equal(UINT8)` (`unsupported type: tensor(uint8)`). Schema check showed the real blocker is worse: task046's UINT8 `Sub` needs opset 14+, while attribute-form `Unsqueeze` is only available through opset 12 and attribute-form `Pad` through opset 10.

Why dead: no single global opset simultaneously supports the current UINT8 arithmetic path and the older no-initializer forms for `Unsqueeze`/`Pad`. Emulating the UINT8 `Sub` under opset 12 costs at least one additional 3x19 UINT8 or QLinearConv path, more than the 3 params saved by dropping `unsq_axes`.

One remaining hypothesis: replace the row-roll multiplexer itself, not the opset metadata. The current schema-level constants are pinned by dtype support rather than by careless graph construction.
