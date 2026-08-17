# task050 - sparse-initializer direct164 Einsum probe is dead

- Live bar used: manifest/team-best cost **164**, profile **164**, sha256 `c88d4f740152c3458c3389cf1634f385b2ae8ffee589c78982a1d0b6d6fbf14a`, from `python tools/ng_current_bar.py --task 50 --json` (`authoritative_source=manifest`, owner Boladi, updated 2026-07-10T22:11:58Z).
- Attempted route: convert the current direct-164 one-node `Einsum` tables `P/G/S/D/A` to ONNX `sparse_initializer`s. If accepted, stored nonzero values would be `75` instead of dense params `164`.
- Prototype: `prototypes/build_direct164_sparse_init_probe.py` -> `prototypes/task050_direct164_sparse_init_probe.onnx`, sha256 `7112042d74f9ff50e4b03041108d3c14fac4cb62aee0d6a46cf73b8095b5da9e`.
- Failed gate: `onnx.checker.check_model(..., full_check=True)` -> `[ShapeInferenceError] (op_type:Einsum, node name: output): Rank of input 1 (0) does not match the equation indices (2).`
- Score result: `score_network_v2(...) -> None`.
- Safety result: `FAIL_ORT_LOAD_AFTER_SANITIZE`: `Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
- Why dead: sparse tensors are not valid `Einsum` operands under ONNX shape inference / ORT sanitizer in this scorer path. The apparent param saving is not creditable because the model cannot load/score.
- One remaining hypothesis: only a legal compressed representation of the dense selector tables inside the one output `Einsum` can go sub-164; sparse initializer encoding is not that route.
