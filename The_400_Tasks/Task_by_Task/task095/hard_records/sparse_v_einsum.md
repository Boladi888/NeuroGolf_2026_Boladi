# task095 sparse-V Einsum route dead

- live bar used: 208, manifest/team-best sha256 `05651c20d66acf86fd6952b5887f6e9aed7f0de06a2a12ba19bedb4792f3fd73`, owner Vamshi, updated 2026-07-12T15:15:54Z.
- attempted route: preserve Vamshi's single free-output Einsum parent but encode the 21 all-zero rows of `V[30,6]` as an ONNX sparse initializer.
- candidate: `The_400_Tasks/Task_by_Task/task095/prototypes/task095_sparseV_probe.onnx`, sha256 `a31a8a70bf504f02033eeaf3286fd9d2dfa93e6cb782a04450ccd8491427c91b`.
- apparent cost if scorer accepted sparse V: 82 params (`V` nnz 54 + `F` 20 + `C` 8), but official scorer rejects it.
- exact failed gate/check: `score_network_v2(...) -> None`; strict shape inference says `(op_type:Einsum): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`; raw ORT load alone passed, so the scorer/shape-inference path is the blocker.
- why dead: sparse initializers do not carry usable rank/shape into strict shape inference for Einsum operands, matching traps §8e. Adding a value_info bridge would bill a dense `[30,6]` tensor and lose the saving.
- one remaining hypothesis: only a scorer/sanitizer behavior change for sparse initializer shape propagation would revive this; otherwise the dense zero-row tax in `V[30,6]` is forced for this one-node parent.
