# task378 hard-record — MaxPool Y output cannot be omitted

- Live bar at session (2026-07-07): **2654**, sha
  `d30e61b3d2e68fd4f30df196ed557c28ca019b7754fc3d613aad0810eec10cfe`
  (`python tools/ng_current_bar.py --task 378 --json`, team manifest). Lower task-local
  reviewer-ready bar: **2614**, sha
  `ed6fef4fe2d3291f396572e15d94e658fb714d72c50ae77de8df24eab48b89f9`.
- Attempted route: set the first `MaxPool` output `pool_values` to an empty optional output
  while keeping `pool_indices4`, aiming to remove the dead charged 40 B F32 tensor.
- Failed gate: `onnx.checker.check_model(model, full_check=True)` →
  `ValidationError Node (pool_values)'s output 0 is marked single but has an empty string in the graph`.
- Why dead: `MaxPool` output 0 (`Y`) is required; only the trailing indices output is optional.
  ONNX optional-output syntax cannot skip `Y` to keep indices.
- One remaining hypothesis: avoid downstream materialized vectors after the required index output
  exists; `pool_values` itself remains mandatory unless the first-index extraction architecture is
  replaced wholesale.
