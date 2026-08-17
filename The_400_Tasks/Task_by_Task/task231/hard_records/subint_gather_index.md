# subint_gather_index

- live bar: cost 168, sha256 `6f13ae96ff65787e0aea9794968ca32c3913dfe9ccbcf55b422ebce75131862d` (`ng_current_bar`, manifest source)
- attempted route: narrow the final dynamic Gather index vector from INT32 to UINT8 to cut the charged `[30]` index tensor from 120 B to 30 B
- exact failed gate: `ng_gate visible --task 231 --cand .../task231_probe_u8_index.onnx` failed before inference with `InvalidGraph: Type 'tensor(uint8)' ... of operator (Gather) ... is invalid`
- why dead: ORT/scorer accepts only INT32/INT64 Gather indices; adding a Cast back to INT32 would reintroduce the charged `[30]` INT32 tensor and lose the intended saving
- one remaining hypothesis: only a non-Gather final-output family that avoids materializing a length-30 index vector can beat 168
