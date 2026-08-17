live bar: 521 bytes, sha256 fe2ea6bc155e42c00f364540ebb1100e650d45b2a2663dbc8dd97bc2a18937bd (ng_current_bar/team manifest, 2026-07-09T01:09:17Z).

attempted route: rebias Sho's final QLinearConv so explicit zero-point initializers could be omitted.

exact failed gate: ONNX checker before scoring. `task189_w1_omit_w_zp`: `ValidationError: Node (output)'s input 5 is marked single but has an empty string`; `task189_w1_omit_x_y_zp_rebias_keep_w`: `ValidationError: Node (output)'s input 2 is marked single but has an empty string`.

why dead: in the pinned ONNX schema used here, QLinearConv x_zero_point and w_zero_point are required single inputs, not omittable optionals. The route cannot save the duplicated UINT8/INT8 zero-point params without failing checker.

one remaining hypothesis: a different final op family, not QLinearConv, may avoid explicit zero-points, but it must retain the 2-D color-code separability that the current bias-bearing QLinearConv provides.
