# O4 ConstantOfShape Scorer Confirmation

## Blunt verdict

Not ready for captain submission. The local A/B probes are built and reproduce the R6 scorer behavior at full NeuroGolf output shape: dense final `Constant` tensors score 9000 params, while final `ConstantOfShape` tensors score only the 4-element shape initializer and zero memory. However, no public task admits a scalar-filled final output tensor, so these files are not valid task candidates and must not be submitted.

Package contents:

- `Logs/research/O4_constantofshape_scorer_confirmation_build.py` - builder/evaluator.
- `Logs/research/O4_constantofshape_scorer_confirmation_results.json` - full local results.
- `Logs/research/O4_constantofshape_scorer_confirmation_models/` - generated ONNX probes.

Reproduction command:

```powershell
.venv/Scripts/python.exe Logs/research/O4_constantofshape_scorer_confirmation_build.py
```

Task-context scan result: across tasks 001-400 `train + test + arc-gen`, there are no all-zero final tensors, no all-one final tensors, and no fixed one-hot final tensor. Eleven tasks have per-example uniform ARC-grid outputs, but shape and/or color vary across examples, and none is a scalar-filled `[1,10,30,30]` output tensor.

## Model inventory

| file | construction | output dtype/shape | local params | local memory | checker | ORT run | task validity |
|---|---|---:|---:|---:|---|---|---|
| `dense_constant_f32_zero_1x10x30x30.onnx` | dense `Constant`, F32 zero fill | FLOAT `[1,10,30,30]` | 9000 | 0 | ok | ok; unique `0.0`, threshold sum 0 | no matching task |
| `constantofshape_f32_default_zero_1x10x30x30.onnx` | `ConstantOfShape`, default F32 zero | FLOAT `[1,10,30,30]` | 4 | 0 | ok | ok; unique `0.0`, threshold sum 0 | no matching task |
| `constantofshape_f32_explicit_zero_1x10x30x30.onnx` | `ConstantOfShape`, explicit F32 scalar zero attr | FLOAT `[1,10,30,30]` | 4 | 0 | ok | ok; unique `0.0`, threshold sum 0 | no matching task |
| `dense_constant_u8_seven_1x10x30x30.onnx` | dense `Constant`, U8 seven fill | UINT8 `[1,10,30,30]` | 9000 | 0 | ok | ok; unique `7`, threshold sum 9000 | no matching task |
| `constantofshape_u8_explicit_seven_1x10x30x30.onnx` | `ConstantOfShape`, explicit U8 scalar seven attr | UINT8 `[1,10,30,30]` | 4 | 0 | ok | ok; unique `7`, threshold sum 9000 | no matching task |

## Delta explanation

The F32 zero pair is semantically identical: both graphs return exactly a `[1,10,30,30]` tensor filled with `0.0`. The U8 seven pair is also semantically identical: both graphs return exactly a `[1,10,30,30]` tensor filled with `uint8(7)`.

The tested cost delta is params accounting only. Local `calculate_params` charges dense `Constant` node `value` attributes by element count, so a full NeuroGolf output tensor costs `1*10*30*30 = 9000` params. For `ConstantOfShape`, the shape initializer `[1,10,30,30]` costs 4 params, and the scalar `value` attribute is not charged locally. Because the generated tensor is named graph output `output`, local `calculate_memory` skips it, so memory remains 0.

This is not a valid task output. NeuroGolf targets are one-hot tensors with exactly one active channel per active grid cell and all-zero padding outside the active grid. A scalar fill across all 10 channels decodes either to all zeros or all ones after `> 0.0`, neither of which matches any task target in the local corpus.

## Captain instructions

Do not submit any model from `Logs/research/O4_constantofshape_scorer_confirmation_models/`. They are local scorer probes only.

If external confirmation is still desired, the exact next step is to commission a different task-specific O4b probe: first identify a real correct task graph where a `ConstantOfShape` replacement can preserve the one-hot output and pass `ng_gate visible`, `ng_gate truth --full`, `ng_gate ab --full`, and real-input profiling. Only after both dense and `ConstantOfShape` siblings are task-valid should a captain run a one-task Kaggle wildcard. The current final-output scalar-fill package has no valid submission target.

Safer local probe shape: use a real graph with an intermediate all-zero/all-one mask or canvas-fill helper and compare a dense initializer/`Constant` sibling against `ConstantOfShape` feeding the same downstream nodes. That would test practical replacement value, but it would not test the final-output-free memory case because intermediates are charged.

## Risks

- Validator shape handling: all generated models use static `[1,10,30,30]` shapes with no `dim_param`; strict shape inference and checker pass.
- Attribute accounting: local scorer ignores `ConstantOfShape.value`; Kaggle confirmation is still absent.
- Graph output dtype: F32 default/explicit zero and U8 explicit seven both load and run locally. U8 positive output thresholds to all ones, which is task-invalid.
- Task validity: no generated probe matches any task. A uniform ARC-grid output is not enough because the submitted tensor must be one-hot over 10 channels.
- Cross-task cascade: the probes have zero local memory, but replacing a real task with any of them would be an incorrect submission and could only create noisy LB/cascade effects, not a clean scorer confirmation.

## Do-not-infer list

- This proves only local scorer behavior for final-output `ConstantOfShape` on the pinned local stack.
- It does not prove Kaggle scorer behavior.
- It does not prove `ConstantOfShape` can replace dense one-hot final constants in valid NeuroGolf tasks.
- It does not prove intermediate `ConstantOfShape` tensors are free; intermediates still incur memory.
- It does not prove all scalar value dtypes or all opsets are safe; this package tested F32 zero and U8 seven at opset 10.
- It does not justify editing root knowledge docs without captain review/extraction.
