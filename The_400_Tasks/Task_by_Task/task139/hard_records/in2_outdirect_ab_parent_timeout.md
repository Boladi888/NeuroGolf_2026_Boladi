# task139 in2/outdirect A/B parent-timeout record

- Live bar used: Team_Best_Onnx/task139.onnx / manifest cost 118, sha256 `9a4f8800c4b25dc2147f12d391ac834e47d18ef5f75b72802afc75e3981ee8f6`.
- Attempted route: 116-param split-selector rewrite of Sho's one-node Einsum. Input-side selector drops color 7 because task139 official/arc-gen inputs contain only colors 0 and 4; output uses a direct 2x10 map for channels 0, 4, 7.
- Exact failed gate:
  - `python tools/ng_gate.py ab --task 139 --cand The_400_Tasks/Task_by_Task/task139/prototypes/task139_in2_outdirect116.onnx --ref Team_Best_Onnx/task139.onnx --n 8000`
  - result: `[ng_gate] wall budget 900s exceeded for ab task139; exiting to avoid orphaned gate work`
  - repeated on reordered candidate `task139_in2_outdirect116_fast.onnx` with the same result.
- Why this route is dead: A/B against the current 118-byte parent executes the slow-order parent and exceeds the shared 900s gate budget. Do not relaunch with a larger `ng_gate --max-seconds`.
- Remaining hypothesis: prove future variants as Path-B with `truth --n 8000`, or only attempt Path-A if a fast byte-equivalent parent proof artifact becomes available without re-running the slow parent A/B.
