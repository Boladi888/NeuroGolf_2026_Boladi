# qlinearconv_zero_point_shaves

- live bar: 1162, sha `8ded3453e806a64e32691d9de41bea2422d11c25054df8edc6ac528cca81377c` (`ng_current_bar`, manifest, 2026-07-09T01:09:17Z)
- attempted route: save 1-2 params by omitting zero-valued `QLinearConv` zero-point inputs or sharing the INT8 zero with an INT8 output.
- failed gates:
  - `build_task381_qtail_optional_zp.py`: checker rejects omitting `y_zero_point` with `input size 7 not in range [min=8, max=9]`.
  - `build_task381_qtail_optional_zp.py`: checker rejects empty `w_zero_point` with `input 5 is marked single but has an empty string`.
  - `ng_gate visible` on `task381_qtail_i8_output.onnx`: ORT load fails with `NOT_IMPLEMENTED : Could not find an implementation for QLinearConv(10)`.
- why dead: in this schema/kernel, all three zero-point inputs used by Sho's tail are load-bearing: output zero-point is required, weight zero-point cannot be blank, and signed INT8 output has no CPU kernel for this quantized type combination.
- one remaining hypothesis: a UINT8-weight reparameterization can share neither the needed zero values nor the negative effective weights without adding a replacement scalar, so it is a net tie/loss unless another tail change deletes a larger structure.
