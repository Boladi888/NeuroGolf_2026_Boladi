# Critique of Research Card B3 deliverable — FunctionProto + If subgraphs

> **STATUS: DEAD-END for NeuroGolf environment — 2026-05-26**. The research itself is technically sound (correct citations to ONNX IR spec, correct ORT execution model, correct counting-policy taxonomy). The PROBLEM is that **both halves of the card hit hard validator-rejection gates** that the card itself did not mention. Companion deliverable: `Onnx_Advanced/function_proto_subgraphs.md` (renamed from `Todo/deep-research-report.md`). Card moved to `Todo/Declined/`. Future cards on subgraph-bearing constructs should explicitly enumerate the validator gates first.

## Executive summary

Card B3 asks whether `FunctionProto` (custom function defs) and `If` (then/else branch subgraphs) can serve as reusable subroutine mechanisms in an environment where `Loop` / `Scan` are banned. The submitted research correctly establishes:

1. `FunctionProto` bodies have no serialized `initializer` field — only `Constant` nodes.
2. ORT 1.24 stores model-local functions as `FunctionTemplate`s at model load, with optional call-site inlining via `Graph::InlineFunction`.
3. Counting policy is scorer-defined: deduplicated definition-level counting vs. expansion/call-site counting.
4. `If` runtime cost: only the executed branch runs; static cost depends on scorer's branch policy.
5. `SequenceMap` is the only other standard subgraph-bearing op besides `If`/`Loop`/`Scan`.

These findings are accurate against ONNX IR and ORT source. The card recommends conditional use of `FunctionProto` when the scorer is "definition-level" and `If` whenever the scorer is branch-aware.

**The fatal omission**: every concrete recommendation depends on the model passing the NeuroGolf validator. Both constructs trigger immediate rejection:

| Construct | Validator gate | Source |
|---|---|---|
| `FunctionProto` (any `model.functions` entry) | `if model.functions: return None` | `data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py:202` |
| `If` (or any node with `AttributeProto.GRAPH` / `GRAPHS`) | `if attr.type in [GRAPH, GRAPHS]: return None` | `neurogolf_utils.py:207-211` |
| Custom domain (required to register `FunctionProto`) | `if opset.domain not in {"", "ai.onnx"}: return None` | `neurogolf_utils.py:203-204` |
| `FUNCTION` op type | listed in `_EXCLUDED_OP_TYPES` | `neurogolf_utils.py:108` |

All four gates are already documented in `Onnx_Advanced/core.md` "Validator rejection criteria" (lines 49-65). The researcher had no way to discover this from the card alone — the card's environment block says only "Banned ops: Loop, Scan, NonZero, Unique, Compress" and never mentions subgraph attributes or `model.functions`.

## Concrete validator-rejection check

Empirically verified by reading `neurogolf_utils.score_network` → `sanitize_model`:

```
neurogolf_utils.py:194-211 (extract)
def sanitize_model(model):
    ...
    if model.functions: return None                          # ← line 202: FunctionProto KILL
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}: return None  # ← line 204: custom-domain KILL
    ...
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH,
                             onnx.AttributeProto.GRAPHS]:
                return None                                  # ← line 211: If/Loop/Scan KILL
```

A `score_network` returning `None` produces a `0` Kaggle score (or full submission rejection). Therefore the entire B3 design space — every "FunctionProto saves K·(N−1) under definition-level scoring" recommendation, every "If saves on the untaken branch" recommendation — is unreachable.

## What IS true and salvageable

The deliverable's general technical content is correct and stays true in the broader ONNX/ORT ecosystem:

- The `FunctionProto` definition pattern (`make_function` + custom domain + `functions=[...]` in `make_model`) is the correct recipe for any ORT consumer that doesn't validator-reject it.
- The ORT hybrid model (templates at load, optional `InlineFunction` later) is the correct execution mental model.
- The counting-policy split (definition-level vs. expansion-level) is real and applies to any scorer that traverses ONNX protobufs.
- The `If` runtime semantics (unused branch not executed) is correctly stated against the ONNX `If` spec.

So the deliverable is reusable as a reference for ONNX subgraph mechanics in any non-NeuroGolf context. Within NeuroGolf, it is purely a "do not pursue" disposition.

## Process lesson for future research cards

Two improvements to the card-writing template would have caught this before researcher hours were spent:

1. **Every card's "Environment" block must enumerate ALL validator rejection gates, not just banned-op-types**. Add a "Validator rejections" subsection that lists `model.functions`, `AttributeProto.GRAPH/GRAPHS`, non-default opset domains, file-size cap, and any other `return None` path in `sanitize_model`. The current cards only list `_EXCLUDED_OP_TYPES` (Loop/Scan/NonZero/Unique/Compress), which leaves the GRAPH-attribute and `model.functions` gates implicit.

2. **Every card should require a "validator pre-check" answer**: "Will the proposed construct pass `sanitize_model` and `calculate_params`? Cite the relevant `neurogolf_utils.py` line numbers." If the answer is no, the card can stop before the research dives into the topic.

Both are now applied as a TODO for `Todo/README.md` in a follow-on (not this critique's scope).

## Disposition

- Deliverable (this critique companion): `Onnx_Advanced/function_proto_subgraphs.md` — kept for ecosystem reference, with a STATUS preamble noting validator rejection.
- Card B3: moved to `Todo/Declined/` with a one-line disposition note appended.
- Project docs: no update to `traps.md` or `techniques.md` needed — `core.md` already documents both validator gates. Cross-reference in this critique is sufficient.

## Related project docs

- `Onnx_Advanced/core.md` — "Validator rejection criteria" section (lines 49-65)
- `data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py:108,194-211` — validator source
- `Onnx_Advanced/Research/notes/b1_quantized_ops_verification_attempt{1,2}_2026-05-25.md` — same critique pattern (card was "research brief rather than factual memorandum")
