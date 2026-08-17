# Output-mask separable free-output route - rank/selector wall (2026-07-12)

- Live bar: `2028`, SHA-256 `da5d9b3cc90ccbc82e3147aef22c969b5f29b6ad7d8b344c5a42274a8ab39596`, from `python tools/ng_current_bar.py --task 187 --json` (`manifest`, reconciled with workbook and `Team_Best_Onnx/task187.onnx`).
- Attempted route: direct #109/#119 separable one-hot output factorization from per-channel output masks; probe script `The_400_Tasks/Task_by_Task/task187/prototypes/probe_output_mask_rank.py`.
- Exact failed gate: pre-build rank/price kill, not a candidate gate. Official visible+arc-gen output masks required rank-sum `min=14, p50=17, p90=18, max=26`; 400 fresh ARC-GEN masks required `min=14, p50=18, p90=18, max=23`.
- Why dead: the masks are data-dependent, so row/column/channel factors would be runtime F32 selectors. A 30x30 term costs `10+30+30=70` elements; runtime F32 selector price is `280*T` bytes. Even the minimum observed `T=14` prices at `3920`, already above the `2028` live graph before detection/extraction.
- Remaining hypothesis: only an implicit-index terminal operator or analytic predicate that avoids materializing runtime row/column selectors can beat the live graph; a generated/factorized runtime selector bank cannot.
