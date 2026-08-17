# task042 — archived history

Superseded early attempt ladder (2026-05-23), moved out of `task042.md` on 2026-07-08 to keep manual content under the size tripwire. All are far above the current best (2,322). The from-scratch descent that reached the shipped QConv detector:

- 2026-05-23 | vec-delta tail from live, preserving original pair logic | verified 266/266 | 20,292 | `task042_vecdelta_from_current.onnx`
- 2026-05-23 | direct channel-3 slice + BOOL output assembly | verified 266/266 | 12,192 | `task042_direct3_boolout_experiment.onnx`
- 2026-05-23 | BOOL internal pair/shift logic | verified 266/266 | 7,911 | `task042_bool_internal_experiment.onnx`
- 2026-05-23 | grouped per-distance masks | verified 266/266 | 7,011 | `task042_bool_grouped_experiment.onnx`
- 2026-05-23 | anchor-crop pair detectors | verified 266/266 | 6,457 | `task042_bool_anchorcrop_experiment.onnx`
- 2026-05-23 | no-clean + 9-channel pre-pad + direct n2/n3 crops | verified 266/266 | 5,895 | `task042_bool_directcrop_n23_noclean_9ch.onnx`
- 2026-05-23 | hybrid QLinearConv n1/n2 + direct n3 | verified 266/266 | 4,919 | `task042_hybrid_qconv12_direct3_noclean_9ch.onnx`

Lesson from the ladder: BOOL shift-AND / anchor-crop realizations of the diagonal detector all cost several thousand bytes of [1,1,10,10] planes; the grouped/dilated QLinearConv (2,943, then the Where-output tail) collapsed the shift-ANDs into conv kernels for free. See `task042.md` for the current best (2,322) and floor proof.
