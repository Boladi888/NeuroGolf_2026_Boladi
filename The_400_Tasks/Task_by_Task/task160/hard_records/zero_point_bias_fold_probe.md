# task160 hard record — QLinearConv zero-point / final-bias fold probes

- **live bar used:** manifest cost 1094, sha `10c3a7c017954791986b3b32b85e7b6ae908b2b2eb44e3130383e9d9cc179c50`, owner Sho, updated 2026-07-09T01:09:17Z (`python tools/ng_current_bar.py --task 160 --json`).
- **attempted route:** task-local probes in `prototypes/zero_point_probes.py` against the current 5-node QLinearConv graph.
- **failed gate:** optional `QLinearConv` x/w/y zero-point omission rejected at ORT load/profile: `input 2 is marked single but has an empty string in the graph`; unadjusted `final_b` -> `x_zero_point=1` fold loaded/profiled at 1092 but failed visible `[visible] 0/265 (0.00%) FAIL`.
- **why dead:** QLinearConv `x_zero_point` is not optional in this ORT/scorer path, so the shared `i8zero` scalar cannot be deleted. The naive no-bias final QLC changes the active-grid logits and is a certain hidden zero.
- **one remaining hypothesis:** only a constrained re-fit of the final QLC weights around a nonzero zero-point could revisit this idea; it must be simulator-proven before a single visible gate, because the unadjusted fold already visible-failed.
