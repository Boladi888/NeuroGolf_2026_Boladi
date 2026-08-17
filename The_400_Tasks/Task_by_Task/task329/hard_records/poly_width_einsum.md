# poly_width_einsum

- live bar: 308, sha `07341d3234d074e6cf3b0584c466521e6527437db74ad7406cfb8948ce1c8f6c`.
- attempted route: replace the validated 308 factorized size-selector Einsum with a polynomial-in-width terminal Einsum using `center(c,W)=1-0.5*((2c+1)-W)^2`.
- failed gate: `python tools/ng_gate.py visible --task 329 --cand The_400_Tasks/Task_by_Task/task329/prototypes/task329_poly_einsum.onnx`.
- exact failure: `[visible] 0/266 (0.00%)  FAIL`.
- why dead: despite scoring `static=260 profile=260`, the polynomial terminal-Einsum formulation is not visible-correct, so it cannot be a candidate and must not receive truth/AB gates.
- one remaining hypothesis: a non-polynomial byte-equal shave may still exist inside the current 308 size-selector graph by reducing duplicated initializer bases without changing the final selector semantics.
