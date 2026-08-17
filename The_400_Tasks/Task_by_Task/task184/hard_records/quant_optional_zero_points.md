live bar: 1137 profile cost, SHA 683bfb6edd7b8e4aa98649cd4dac5600cc0ef30a40e5b4a664a2ec0bc0500f7e (manifest via ng_current_bar.py, 2026-07-10T09:55:00Z)

attempted route: delete QLinearConv zero-point scalar params by leaving x/w/y zero-point inputs empty.

failed gate: `onnx.checker.check_model(full_check=True)` in `prototypes/build_task184_quant_zp_variants.py`.

why dead: QLinearConv zero-point inputs are marked single by the checker in this opset; empty-string omission fails before scoring or visible (`input 2` or `input 5` cannot be empty).

one remaining hypothesis: typed zero-point sharing is still legal; `uint8k_shared_zp` uses UINT8 weights and reuses `zp8`, saving the signed `zpi8` scalar.
