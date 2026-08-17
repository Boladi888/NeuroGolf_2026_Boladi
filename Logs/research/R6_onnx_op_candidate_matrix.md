# R6 ONNX Op Candidate Matrix Probe

Pinned stack verified locally: `onnx==1.21.0`, `onnxruntime==1.24.4`, `CPUExecutionProvider`, and probe sessions used `ORT_DISABLE_ALL`. Tiny probe models were written under `Logs/research/R6_probe_models/`.

## Blunt Verdict

No broad new ONNX surface should be adopted immediately. The opset 23-26 surface is mostly neural-helper or schema-only material for NeuroGolf. The only new task-level candidate worth keeping warm is `LpPool(p=1)` as a zero-param 2-D window sum for nonnegative float masks. `ConstantOfShape` remains useful but is already covered; the local probe adds one sharper fact: the scorer charged only the 4-element shape initializer for a final-output uniform fill and did not charge the `value` tensor attribute.

Sequence/container paths are dead for normal submissions. `SplitToSequence -> ConcatFromSequence` passed checker and ORT, but the scorer rejected `SplitToSequence` as not permitted. `ReverseSequence` is also scorer-rejected, and ORT additionally rejects spatial `time_axis=3` even though the schema accepts rank-4 tensors.

## Candidate Matrix Summary

CSV: `Logs/research/R6_onnx_op_candidate_matrix.csv`

| family | covered | task_specific | probe_now | dead | total |
|---|---:|---:|---:|---:|---:|
| opset23_26_delta | 1 | 6 | 0 | 7 | 14 |
| sequence_container | 0 | 0 | 0 | 18 | 18 |
| shape_constant_generator | 10 | 3 | 0 | 0 | 13 |
| obscure_spatial | 6 | 5 | 1 | 4 | 16 |
| already_covered | 9 | 0 | 0 | 0 | 9 |
| banned_or_fatal | 0 | 0 | 0 | 3 | 3 |
| total | 26 | 14 | 1 | 32 | 73 |

## Probe Results

| probe | opset | op chain | input shape | dtype | checker | shape inference | ORT load/run | scorer | byte/param delta estimate | verdict |
|---|---:|---|---|---|---|---|---|---|---|---|
| p1 | 23 | `RMSNormalization(axis=-1)` | `[1,2,4]` | F32 | ok | output `[1,2,4]` | ok; sample starts `0.365,0.730,1.095` | ok `memory=0 params=4` | Can fuse an RMS norm chain but pays scale params and only matters for true float normalization | task-specific |
| p2 | 24 | `SplitToSequence -> ConcatFromSequence` | `[1,4]` | F32 | ok | output tensor plus sequence value_info | ORT ok; output equals input | fail; scorer printed `Op type SplitToSequence is not permitted` | Bridge rematerializes tensor; non-final output costs full bytes | dead |
| p3 | 25 | `ConstantOfShape(shape=[1,1,4,4], value=u8(7))` | dummy `[1,1,4,4]` | U8 output | ok | output `[1,1,4,4]` | ok; all 7s | ok `memory=0 params=4` | Dense final constant reference was 16 params; generator was 4 params locally | covered |
| p3b | 25 | dense `Constant` reference | dummy `[1,1,4,4]` | U8 output | ok | output `[1,1,4,4]` | ok; all 7s | ok `memory=0 params=16` | Reference for p3; dense params scale as tensor elements | reference |
| p4 | 22 | `LpPool(p=1,kernel=2x2,stride=1)` | `[1,1,4,4]` | F32 | ok | output `[1,1,3,3]` | ok; 2x2 sums | ok `memory=0 params=0` | Replaces all-ones 2x2 Conv weights for nonnegative masks; float-only | probe-now |
| p5 | 10 | `ReverseSequence(batch_axis=0,time_axis=3)` | `[1,1,2,4]` | F32 | ok | output `[1,1,2,4]` | load failed; ORT requires `time_axis < 2` | fail | Direct NCHW width reversal is blocked by ORT kernel assertion | dead |
| p5b | 10 | `ReverseSequence(batch_axis=0,time_axis=1)` | `[1,4,2]` | F32 | ok | output `[1,4,2]` | ok; axis reversed | fail; scorer printed `Op type ReverseSequence is not permitted` | Legal-axis reversal still cannot pass scorer | dead |
| p6 | 26 | `BitCast(float32 -> int32)` | `[1,4]` | F32 to I32 | ok | output `[1,4]` | load failed; ORT only guarantees ai.onnx through opset25 | fail | Schema exists but opset26 is not loadable here and no CPU kernel is registered | dead |
| p7 | 24 | `Swish(alpha=1)` | `[1,4]` | F32 | ok | output `[1,4]` | ok through ORT function expansion | ok `memory=0 params=0` | Fuses `x*sigmoid(x)` but that idiom is not common in ARC grids | task-specific |

## Recommended Extractions

Add only after captain review:

- `Onnx_Advanced/op_cost_patterns.md`: add `LpPool(p=1)` as a float-only nonnegative window-sum replacement for all-ones Conv kernels.
- `Onnx_Advanced/ort_reality.md`: record `ReverseSequence` local constraints: scorer rejects the op and ORT CPU rejects `time_axis >= 2`.
- `Onnx_Advanced/scoring_internals.md`: record the p3/p3b local scorer delta: final-output `ConstantOfShape` with shape init `[1,1,4,4]` scored 4 params while a dense Constant scored 16.
- `Onnx_Advanced/ort_reality.md`: record that ORT 1.24.4 loads official ai.onnx through opset25; opset26 `BitCast` passed checker but did not load.

## Open Risks

- Kaggle/grader confirmation is still needed before exploiting the `ConstantOfShape` value-attribute accounting beyond ordinary final-output uniform fills.
- Some non-shortlist matrix rows are intentionally marked `not_probed`; they are dead or covered for cost reasons and were not worth model probes in this card.
- `Swish` has no registered CPU kernel but executes through function expansion locally. Treat it as runnable on this stack, not as evidence of a kernel registration.
- `LpPool(p=1)` is exact only for nonnegative inputs. ARC one-hot and mask planes satisfy that; signed deltas do not.
