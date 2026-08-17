# task342 selector-drop retune wall

- Live/team bar at probe time: manifest 126 bytes, SHA `9a749314b24fdbf582eee9c05b47dc9361a441c5baf00c18afac8f7c6d499d27`; task-local staged bar 116 bytes, SHA `4F5147C0802213ACF5F076D7A4EC3F56AEF55C9C076FEDAC823E38F8C7B04D55`.
- Attempted route: keep the 106-byte selector-drop ONNX structures but retune all 8 `branch_core[p,a,b]` coefficients to replace the deleted selector with count/moment classification.
- Failed checks: raw `drop_output_selector` gate failed `[visible] 0/266 (0.00%)  FAIL`; train/test linear feasibility for retuned `drop_output_selector` was infeasible; `drop_local_selector` had direct positive/negative feature contradictions and was infeasible.
- Why dead: the deleted selector removes information that the remaining four branch features per path cannot linearly separate, even before ARC-GEN rows are considered.
- One remaining hypothesis: leave both channel selectors intact and seek a lower-param basis/mode rewrite, or use a materially different graph family rather than selector deletion.
