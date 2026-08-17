# task370 — F32-slice zero-mask is a structural memory floor (no cast-first escape)

## What I discovered
On task370 (diagonal zero-mask projection, dzmp architecture, team-best 8868 = mem 7849 + params 1019), the single largest charged memory tensor is `_v_289_f32` at **1600 B** — an F32 `Slice(input,[0:20,0:20],ch0)` that exists ONLY to be immediately `Cast→UINT8` (`dzmp_zero_u8_20`, 400 B). It looks like obvious waste (1600 B intermediate thrown away), but it is structural:

- ONNX `Slice` output dtype == input dtype. Input is the fixed F32 one-hot `[1,10,30,30]`, so slicing any spatial region of it is F32.
- Casting the input to UINT8 BEFORE slicing inflates to the full `[1,10,30,30]` u8 = 9000 B. **Measured: cast-input-first scores 16268 (vs 8868).** Strictly worse.
- There is no already-materialized spatial U8 tensor to reuse for the zero-channel (the per-channel `ReduceSum`/`Equal` paths are `[1,10]` channel vectors, not `[1,1,H,W]` spatial).

So an F32→U8 boundary at the input is unavoidable when a downstream op (here QLinearConv) needs a U8 spatial plane, and the F32 slice charges at min `H*W*4`. The deferred-cast / input-to-bool whole-graph-rewrite tricks (techniques.md Rule 10 / task376 exemplar) do NOT apply here — those recover the cast cost downstream via MatMul/Gemm; this graph's downstream is a U8 QLinearConv that genuinely needs U8 early.

## Where
- Builder: `The_400_Tasks/Task_by_Task/task370/prototypes/build_task370_i32_microclean.py`
- Team-best: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task370.onnx` (SHA 09fa0cae…)
- Floor analysis written into `The_400_Tasks/Task_by_Task/task370.md` Cost analysis.

## Why it matters
Saves future workers from re-attempting the "obviously kill the 1600 B F32 slice" optimization on task370 (and any graph with a fixed-F32 one-hot input feeding a U8 spatial conv). The F32 input-region slice is a memory floor; cast-first is a known regression. task370 8868 is a byte-equal-to-LIVE floor (LIVE is 99.941% vs ground-truth — the 29×29 kernel distance-15 truncation — so byte-equal-to-LIVE is the gold gate, not 100%-truth).

## What's missing from the library
No explicit note that "F32 one-hot input → U8 spatial conv forces an F32 slice intermediate at H*W*4 that cannot be removed by cast-first." Could extend the cost_floor_reference plane-floor taxonomy or the deferred-cast Rule 10 with this counter-case (cast-first input inflation when downstream needs early-U8). Sorter's call on placement.
