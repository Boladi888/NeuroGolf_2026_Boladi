# task342 116-byte A/B n8000 timeout

- Live/team bar at probe time: manifest 126 bytes, SHA `9a749314b24fdbf582eee9c05b47dc9361a441c5baf00c18afac8f7c6d499d27`; task-local staged bar 116 bytes, SHA `4f5147c0802213acf5f076d7a4ec3f56aef55c9c076fedac823e38f8c7b04d55`.
- Attempted route: escalate the existing one-op `Einsum` Path-A compression `task342_wherereducesumbridge_cost116` to required `ab --n 8000` proof against `Team_Best_Onnx/task342.onnx`.
- Failed gate: `python tools/ng_gate.py ab --task 342 --cand The_400_Tasks/Task_by_Task/task342/staged_candidates/task342_wherereducesumbridge_cost116/task342.onnx --ref Team_Best_Onnx/task342.onnx --n 8000 --json --out The_400_Tasks/Task_by_Task/task342/prototypes/task342_worker3_ab_n8000_ref_team_best.json` -> `[ng_gate] wall budget 900s exceeded for ab task342; exiting to avoid orphaned gate work`.
- Why stopped: protocol forbids relaunching a per-candidate gate with a larger `--max-seconds` after an `ng_gate` wall-budget timeout. Existing n2000 truth/A-B remains useful evidence, but this is not `LOCAL_GOLD`.
- One remaining hypothesis: a reviewer/captain policy could run a sanctioned long-window A/B, or a different sub-116 graph must avoid this one-op `Einsum` runtime profile while preserving byte equality.
