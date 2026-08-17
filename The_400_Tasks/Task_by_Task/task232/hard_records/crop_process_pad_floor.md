# task232 hard record - crop-process-pad at 205 bar

- Live/team-best bar: cost 205, profile_cost 205, sha256 `f54c7425eeb6a1da3977db56be71af2cadeafdc729064e47ce402f3cc5cf80e5`.
- Bar source: `NG_OnnxlyFans_TaskStatus.xlsx` row task232 and `Team_Submission/Backend/manifest.csv` row task232, both updated `2026-07-02T02:35:15Z`.
- Attempted route: use the 14x14 active-grid bound to crop, process row/column vectors, then pad back to 30x30.
- Exact failed gate: no ONNX gate run; prebuild cost-floor check fails because any materialized `1x10x14x14` output-like tensor costs at least 1,960 F32 bytes or 1,960 params/BOOL elements before padding, already far above the 205-byte one-Einsum bar.
- Why dead: the current team-best has zero intermediates and only 205 initializer elements, so memory-saving crop logic cannot win unless it also avoids materializing the crop tensor entirely.
- One remaining hypothesis: only revisit crop if it is still a single free-output expression with no intermediate crop plane.
