# O3 Attention-23 Candidate Corpus Probe

## 1. Blunt verdict

No real current corpus target.

The live/prototype corpus has no `Softmax` node at all across the scanned ONNX artifacts, so there is no `MatMul -> Softmax -> MatMul`, `Einsum -> Softmax -> MatMul/Einsum`, or charged `[S,S]` softmax score matrix to replace with `Attention-23/24`.

`Attention-23` itself is viable in the pinned stack: checker passes, strict shape inference passes, ORT CPU loads/runs, and the NeuroGolf scorer charges a single-output `Attention` probe as `memory=0 params=0`. That confirms the mechanism, but the mechanism has no current task graph to apply to.

Outputs from this card:

- Scan script: `Logs/research/O3_attention23_candidate_corpus_probe_scan.py`
- Scan JSON: `Logs/research/O3_attention23_candidate_corpus_probe_scan.json`
- Probe script: `Logs/research/O3_attention23_candidate_corpus_probe_probe.py`
- Probe JSON: `Logs/research/O3_attention23_candidate_corpus_probe_probe_results.json`
- Probe models:
  - `Logs/research/O3_attention23_candidate_corpus_probe_attention23.onnx`
  - `Logs/research/O3_attention23_candidate_corpus_probe_explicit_matmul_softmax.onnx`

## 2. Corpus scan method

Protocol read before work: `Onnx_Advanced/WORKER_PROTOCOL.md`, `Onnx_Advanced/INDEX.md`, `Onnx_Advanced/core.md`, `Onnx_Advanced/traps.md`.

ONNX artifact scan:

- `submissions/submission.zip::*.onnx`
- `*.onnx` at repo root
- `The_400_Tasks/Task_by_Task/**/*.onnx`
- `staged_submits/**/*.onnx`
- `work/**/*.onnx`

Scanner behavior:

- Loaded every model with `onnx.load` or `onnx.load_model_from_string`.
- Ran `onnx.shape_inference.infer_shapes(..., strict_mode=True)` when possible.
- Indexed producer/consumer dataflow.
- Queried exact `Softmax` chains and pairwise square `MatMul`/`Einsum` outputs consumed by `MatMul`/`Einsum`.
- Computed static bytes as `prod(shape) * dtype_size`, excluding tensors named exactly `input` or `output`.

Scan counts:

- Models enumerated/loaded: `7162 / 7162`
- Shape-inference failures: `69` (op scan still used raw graph in these cases)
- Total `MatMul` nodes: `2611`
- Total `Einsum` nodes: `4924`
- Total `Softmax` nodes: `0`
- Total `Attention` nodes: `0`
- Attention-compatible chains: `0`
- Pairwise non-softmax near-misses: `5`

Text/prototype queries:

```powershell
rg -n "\bSoftmax\b" The_400_Tasks staged_submits work Logs repo -g *.py -g !Logs/research/O3_attention23_candidate_corpus_probe_*
rg -n "\bAttention\b" The_400_Tasks staged_submits work Logs repo -g *.py -g !Logs/research/O3_attention23_candidate_corpus_probe_*
rg -n "\b(MatMul|Softmax|Einsum|Attention)\b" The_400_Tasks staged_submits work Logs repo -g *.py -g *.md -g *.json
rg -n "qk|query|key|self.?attention|attn|weighted value|score matrix" The_400_Tasks staged_submits work Logs repo Onnx_Advanced -g *.py -g *.md -g *.json
```

The pre-existing `.py` prototype scans for `Softmax` and `Attention` returned no hits. Broader text hits were documentation/history references, not graph builders.

## 3. Candidate table

No true candidates were found. The table below records the only square pairwise aggregation near-misses so the rejection is explicit.

| task/path | chain | shapes | current bytes | Attention-compatible? | expected delta | verdict |
|---|---|---|---:|---|---:|---|
| `task154` / `The_400_Tasks/Task_by_Task/task154/prototypes/task154.onnx` | `MatMul(R_mat,c5_4d)->c5_r_4d; MatMul(c5_r_4d,C_mat)->c5_out_4d` | `c5_r_4d [1,1,30,30] F32` | `1*1*30*30*4 = 3600 B` | No. Reflection/placement matrix multiply, no Softmax, no Q/K/V probability weights. | `0 B` | Reject. Generic spatial transform, not attention. |
| `task088` / `The_400_Tasks/Task_by_Task/task088/prototypes/task088_compress_v1.onnx` | `MatMul(M_up,shape_mask)->shape_up; MatMul(shape_up,M_left)->shape_shifted` | `shape_up [1,1,24,24] F16` | `1*1*24*24*2 = 1152 B` | No. Shape shifting by row/col masks, no Softmax normalization. | `0 B` | Reject. Generic matrix product. |
| `task088` / `The_400_Tasks/Task_by_Task/task088/prototypes/task088_compress_v2.onnx` | same as above | `shape_up [1,1,24,24] F16` | `1*1*24*24*2 = 1152 B` | No. Same non-attention shift chain. | `0 B` | Reject. Duplicate near-miss variant. |
| `task088` / `The_400_Tasks/Task_by_Task/task088/prototypes/task088_compress_v3.onnx` | same as above | `shape_up [1,1,24,24] F16` | `1*1*24*24*2 = 1152 B` | No. Same non-attention shift chain. | `0 B` | Reject. Duplicate near-miss variant. |
| `task088` / `The_400_Tasks/Task_by_Task/task088/prototypes/task088_f16trick_80BA06BD.onnx` | same as above | `shape_up [1,1,24,24] F16` | `1*1*24*24*2 = 1152 B` | No. Same non-attention shift chain. | `0 B` | Reject. Duplicate near-miss variant. |

There are no `[S,S]`, `[H*W,H*W]`, or `[N,N]` charged matrices followed by softmax-normalized weighted value aggregation in the scanned corpus.

## 4. Probe result table

| probe | opset | inputs | checker | shape inference | ORT load/run | scorer | notes |
|---|---:|---|---|---|---|---|---|
| `attention23_self_4d` | 23 | Single NeuroGolf input `input [1,10,30,30] F32`; node uses it as Q/K/V; output `output [1,10,30,30] F32` | PASS | PASS; no `value_info` intermediates | PASS, CPU output `[1,10,30,30] float32` | `macs=0 memory=0 params=0 cost=0` | `scale=1.0`; only graph tensor after input is graph output, so qk/softmax internals are uncharged. |
| `explicit_matmul_softmax_matmul` | 23 | Same input/output shape | PASS | PASS; `k_t`, `scores`, `weights` all `[1,10,30,30] F32` | PASS, CPU output `[1,10,30,30] float32` | `macs=0 memory=108000 params=0 cost=108000` | `3 * (1*10*30*30*4) = 108000 B`; numerically identical to Attention probe, max abs diff `0.0`. |

Probe interpretation: `Attention-23` would be a real cost lever for an explicit attention chain because the explicit baseline pays for transpose, score, and softmax tensors while the single-output fused op does not expose them as charged graph tensors.

## 5. Recommended next action

Close O3 as negative corpus evidence. Do not spend task-worker time on an `Attention-23` rewrite now.

Keep the probe result in mind only as a future trigger: if a later generated or imported graph contains an actual float `MatMul/Einsum -> Softmax -> MatMul/Einsum` with a large charged score/weights tensor, try `Attention-23` with exactly one output and price casts from the current dtype. Until such a graph exists, the opset-23 attention idea is parked, not actionable.

## Final checklist

- [x] Corpus scan completed against current graph locations.
- [x] Local `Attention-23` smoke probe run in pinned `.venv`.
- [x] Every near-candidate has byte arithmetic.
- [x] No task model submitted or root docs edited.
