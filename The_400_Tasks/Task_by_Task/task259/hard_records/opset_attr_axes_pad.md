# task259 opset_attr_axes_pad dead route

- Live bar: 907 from `python tools/ng_current_bar.py --task 259 --json`, sha256 `6f3f1ec1a5937f38712575db8f21f6cdbc08d3df0cb2701c221bc986d39d925f`.
- Effective staged bar: 898 from `staged_candidates/task259_bool_tail_pad_axes_cost898/manifest.json`, sha256 `d5f8ce990629e58a08c5a7379dac297debaf864afc14e73a6006a100943c127e`.
- Attempted route: `prototypes/task259_v13_opset10_attr_axes_pad.onnx`, built by `prototypes/build_task259_v13_opset10_attr_axes_pad.py`; keep v11 computation but downgrade to opset 10 so reduction axes, Unsqueeze axes, and Pad pads can be attributes instead of charged initializers.
- Raw ONNX bytes: 3834; sha256 `904b425a5205d052d10b14a9f420fe07be8a4f6af886e0934485791647dd7bde`.
- Failed gate: `score_network_v2 returned None`; ORT load error: `Type 'tensor(bool)' of input parameter (color_patch_b) of operator (ReduceMax) in node (occ_b) is invalid.`
- Why dead: `ReduceMax(BOOL)` is only legal at opset 20 in the local ONNX schema, but opset 20 requires axes inputs for reductions/Unsqueeze and Pad axes for the current cheap final pad. Replacing bool occupancy with F32 occupancy adds about 36 memory bytes and loses the initializer savings.
- One remaining hypothesis: only a non-`ReduceMax(BOOL)` occupancy construction with under 13 added bytes could make an attribute-axis downgrade viable; the obvious F32 occupancy path is too expensive.
