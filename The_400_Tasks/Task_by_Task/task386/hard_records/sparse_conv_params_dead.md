# task386 sparse-Conv parameter route dead

- live bar: cost 78, sha256 `9c54eca017da2e3ce23451448e72ac42cbf846b2b13e96e2bad3781978a6cf9e`, source `python tools/ng_current_bar.py --task 386 --json` / team manifest.
- attempted route: convert the current dense Conv initializers to sparse initializers to price only the 4 nonzero detector weights and 2 nonzero output weights.
- probe path: `The_400_Tasks/Task_by_Task/task386/prototypes/task386_sparse_conv_probe.onnx`, sha256 `5298865a4fb908c6093dc426e948d6d0e3301d4cc7749c02362b4e89ce541a10`.
- exact failed gate/probe lines:
  - `score_network_v2(...) -> None`
  - `ng_gate visible`: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
- why dead: scorer sanitizer/checker does not preserve sparse initializer connectivity for Conv inputs; this matches the library sparse-initializer caveat.
- one remaining hypothesis: no safe sparse-weight version is known; only a future scorer-supported sparse Conv or a different legal op that accepts sparse parameters could revive this.
