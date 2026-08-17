# Research Card O3_REPOACCESS - Attention-23 candidate corpus probe

**Deliverable**: repo-access research note and optional tiny probe models. Save outputs under `Logs/research` with basename `O3_attention23_candidate_corpus_probe`.

## Worker contract

You have full repo and tool access. Follow `Onnx_Advanced/WORKER_PROTOCOL.md`. You are not alone in the codebase; do not revert or overwrite unrelated edits. Do not submit models. Do not edit root knowledge docs directly.

## Why this card exists

The N1/R6 research closed opset 23-26 as mostly barren, but left one speculative candidate: `Attention-23/24`. It may help only if it replaces a real large `MatMul -> Softmax -> MatMul` or attention-like chain with charged intermediate tensors.

This card decides whether any current NeuroGolf graph actually has such a target.

## Required work

1. Search current ONNX graphs and prototypes for attention-like chains:
   - `MatMul -> Softmax -> MatMul`
   - `Einsum -> Softmax -> MatMul/Einsum`
   - large pairwise score matrix followed by weighted value aggregation
   - any `[S,S]`, `[H*W,H*W]`, or `[N,N]` charged matrix that is immediately reduced into a value map
2. For each candidate graph, record task id, path, node chain, shapes, current intermediate bytes, and whether Q/K/V semantics match ONNX `Attention`.
3. Build one tiny local `Attention-23` probe using the pinned `.venv`, and verify checker, shape inference, ORT CPU load/run, and scorer behavior if feasible.
4. If a real task candidate exists, sketch the rewrite and cost delta. Do not implement the task rewrite unless explicitly asked later.

## Required report sections

1. **Blunt verdict**: any real corpus target, yes/no.
2. **Corpus scan method**: exact paths searched and query patterns.
3. **Candidate table**:

`| task/path | chain | shapes | current bytes | Attention-compatible? | expected delta | verdict |`

4. **Probe result table**:

`| probe | opset | inputs | checker | shape inference | ORT load/run | scorer | notes |`

5. **Recommended next action**.

## Guardrails

- Do not confuse generic matrix multiplication with attention. The fused op only helps if it can hide a large score/softmax intermediate or fuse a real value aggregation.
- `Attention` is float/f16; price casts from uint8/bool/f32 one-hot grids.
- Multi-output optional present-key/value/qk outputs must not become charged outputs.
- If no current graph has the chain, close this as negative evidence.

## Final checklist

- [ ] Corpus scan completed against current graph locations.
- [ ] At least one local `Attention-23` smoke probe run or a clear reason it could not run.
- [ ] Every candidate has byte arithmetic.
- [ ] No task model submitted or root docs edited.
