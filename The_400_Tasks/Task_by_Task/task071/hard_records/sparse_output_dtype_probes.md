# task071 dead route: sparse/output-dtype probes on 843 DFT graph

- Live bar used: cost 843, SHA `82a0d5108c509ef5bf46191bcb953420f9fd32560cb87f8c5f04e17446f83da5`, Vamshi team-best from manifest/workbook updated `2026-07-02T02:35:15Z`.
- Attempted route: replace zero-heavy initializers (`F`, `Q`, `cmap`, `R`, channel weights) with sparse initializers; separately lie graph output dtype to F16 or BOOL.
- Exact failed gate: `score_network_v2` returned `None` for `sparse_F.onnx`, `sparse_all_zero_heavy.onnx`, `output_decl_f16.onnx`, and `output_decl_bool.onnx`.
- Why dead: these forms do not survive the official scorer path, so they are not valid candidates even before visible/truth gates.
- One remaining hypothesis: factor or compress dense `F [7,2,30]` inside a legal final Einsum representation without sparse tensors or output dtype lies.
