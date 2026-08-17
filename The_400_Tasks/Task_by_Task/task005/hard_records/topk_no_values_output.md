# task005 dead route — TopK no-values output (2026-07-13)

- Live bar: 2852, sha `d38cdf047af75e26d3bcf919b8b9fe8e70314233612430b87de422317f003d8b`; task-local bar: 2727, sha `485cd9e516a526c8d2c46ceb5481ced01ca5b8a9b48ccad3a7f7ff49a37ec08a`.
- Attempted route: set `TopK` values output to an empty optional output to avoid charging unused `_tk_vals`.
- Failed gate: build/checker in `prototypes/_task005_probe_topk_variants.py` before visible; exact error: `Node (_tk_topk)'s output 0 is marked single but has an empty string in the graph`.
- Why dead: ONNX TopK values output is required in this schema, so the unused values tensor cannot be omitted.
- One remaining hypothesis: only dtype-narrowing the required values output can shave this block; implemented separately as `task005_topk_f16_biased_cost2723`.
