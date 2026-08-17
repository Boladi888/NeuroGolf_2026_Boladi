# int8w_omit_wzero

- live bar: 614 profile/cost, sha256 `826a019cb4f85c90b5e81d087c84478d522248ba25aa47468b72161d06254164`, source `python tools/ng_current_bar.py --task 294 --json` / team manifest; task-local staged bar before this pass was 611 from `task294_sharedscale_yzero_cost611`.
- attempted route: encode the proven signed QLinearConv kernel directly as INT8 weights and omit the charged `w_zero` scalar to save 1 param.
- exact failed gate: build/checker before visible: `ValidationError: Node (int8w_default_wzero_qlinearconv)'s input 5 is marked single but has an empty string in the graph`.
- why dead: ONNX schemas for QLinearConv opsets 10, 11, 13, 14, 19, and 21 all mark `w_zero_point` as `FormalParameterOption.Single`, so this input cannot be omitted.
- one remaining hypothesis: use a different final operator family, or reduce crop/carrier memory; direct `w_zero` omission is schema-dead.
