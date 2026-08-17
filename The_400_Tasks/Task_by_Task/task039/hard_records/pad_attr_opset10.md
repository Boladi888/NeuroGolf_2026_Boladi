# task039 hard record - pad_attr_opset10

Live/team bar used: cost 441, sha256
`dc9191feb18062f9d159affc29e09b0bd8b77a3259cefb129ada0876dd915e0b`,
from workbook and team manifest refreshed 2026-07-03.

Attempted route: lower Vamshi's scalar-moment graph to opset 10 so final
`Pad` can use attribute-form `pads` and delete the 4-element `pad_pads` plus
2-element `pad_axes` initializers. Expected static cost if legal: 435.

Exact failed gate: builder/checker for
`The_400_Tasks/Task_by_Task/task039/prototypes/build_task039_attrpad_v1.py`
failed before ONNX emission:
`ValidationError: No Op registered for Einsum with domain_version of 10`.

Why dead: the scalar-moment detector depends on `Einsum`, which is not available
at opset 10. The model cannot mix Pad-2 attribute form with Einsum under a single
standard ONNX opset. Replacing the two scalar Einsums with opset-10 primitive
Mul/Reduce/MatMul forms would add either full-canvas intermediates or large dense
weights, exceeding the 6-param pad saving.

One remaining hypothesis: a completely different non-Einsum detector that also
keeps detector overhead under 43 params and 32 charged bytes could revive an
attribute-Pad route, but no known task039 route has that profile.
