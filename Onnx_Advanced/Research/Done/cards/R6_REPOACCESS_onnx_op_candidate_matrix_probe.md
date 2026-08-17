# Research Card R6_REPOACCESS - Build and probe an ONNX/ORT op candidate matrix

**Deliverable**: repo-access research note plus generated artifacts. Suggested note basename: `R6_onnx_op_candidate_matrix` under `Logs/research/`.

## Worker contract

You have full repo and tool access. Follow `Onnx_Advanced/WORKER_PROTOCOL.md`. Do not submit models. Do not edit the root knowledge base directly except by writing a research note and optional generated CSV/JSON artifacts under `Logs/research/`.

This card exists to turn the internet-only N-series findings into local proof against the real pinned stack and scorer.

## Goal

Generate a machine-readable candidate matrix for ONNX default-domain operators on the local pinned stack, then run minimal local probes for the most promising obscure ops.

Pinned stack expected by the repo:

```
onnx==1.21.0
onnxruntime==1.24.4
CPUExecutionProvider
Graph optimizations disabled
```

## Required artifacts

1. Markdown note under `Logs/research/` with basename `R6_onnx_op_candidate_matrix` - concise research note with verdicts.
2. `Logs/research/R6_onnx_op_candidate_matrix.csv` - one row per candidate op/version.
3. Optional: `Logs/research/R6_probe_models/` for tiny generated ONNX probes, only if useful and small.

## Required matrix columns

`op, domain, since_version, latest_schema_version, ort_cpu_kernel, ort_kernel_version_range, dtype_constraints, output_kind, scorer_risk, candidate_family, verdict, notes`

`candidate_family` should be one of:

- `opset23_26_delta`
- `sequence_container`
- `shape_constant_generator`
- `obscure_spatial`
- `already_covered`
- `banned_or_fatal`

`verdict` should be one of:

- `probe_now`
- `task_specific`
- `covered`
- `dead`
- `needs_internet_source`

## Required local probes

Run checker + ORT CPU session creation + one inference + scorer/shape sanity where feasible for at least these categories:

1. A top opset 23-26 candidate, if any local schema and ORT CPU kernel exist.
2. One sequence/container bridge pattern, such as `SplitToSequence -> ConcatFromSequence`, or prove it is unsuitable without writing a graph if the scorer type would be non-tensor/fatal.
3. `ConstantOfShape` or another generator replacing a large stored constant.
4. One obscure spatial/window op not already routine, such as `RoiAlign`, `LpPool`, `Col2Im`, `MaxUnpool`, `ReverseSequence`, or later `Pad(mode="wrap")`, depending on local support.
5. A negative-control op that passes ONNX schema but lacks an ORT CPU kernel or fails the scorer path.

For every probe, record:

`opset, op chain, input shape, dtype, check_model result, shape_inference result, ORT load result, run result, scorer result if applicable, byte/param delta estimate, final verdict`

## Guardrails

- Do not re-probe the entire existing `ort_reality.md` dtype matrix. Probe only new gaps or claims from N-series cards.
- Do not use banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
- Do not add doctrine to `techniques.md` until a captain extracts the durable finding.
- If a candidate is only useful when it is the final graph output, say that explicitly.
- If an op produces a sequence/string/optional output and needs a tensor bridge, price the bridge.

## Suggested implementation approach

Use local Python introspection:

- `onnx.defs.get_all_schemas_with_history()` for schema/version inventory.
- `onnxruntime.capi._pybind_state.get_all_opkernel_def()` for registered ORT kernels.
- Existing repo scorer helpers only if a tiny model can be scored without task data; otherwise use checker + ORT + symbolic byte arithmetic and mark scorer as `not_applicable`.

## Required final sections

1. **Blunt verdict**: name any op/idiom worth immediate adoption, or say the new surface is barren.
2. **Candidate matrix summary**: counts by family and verdict.
3. **Probe results**: one compact table.
4. **Recommended extractions**: exact target doc/section for any durable finding.
5. **Open risks**: anything needing Kaggle/grader confirmation.

## Verification checklist

- [ ] CSV matrix written.
- [ ] At least five probes or justified non-probes recorded.
- [ ] All probes distinguish checker, shape inference, ORT load, ORT run, and scorer behavior.
- [ ] No banned ops used.
- [ ] No root knowledge-base edits made directly.
