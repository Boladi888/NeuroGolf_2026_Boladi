# task032 compact_vertical_truecolor hard record

- Live bar used: cost 403, sha256 `e3ca4a62f12fe88ff9852a6159c24108f8438d030eff62ecf6a74267146754ce` from `python tools/ng_current_bar.py --task 32 --json` / team manifest.
- Attempted route: replace the 6x6 code canvas plus 1x1 final `QLinearConv` with a vertical padded final kernel over compact true-color column features. The only plausible sub-403 feature set was `m=active_col*(n-k)`, `q=color_code`, and `q2=q*q`; a masked-empty `q=0` variant was also checked.
- Exact failed gate: LP separability gate in `The_400_Tasks/Task_by_Task/task032/prototypes/probe_compact_final_routes.py`.
  - Command: `python The_400_Tasks/Task_by_Task/task032/prototypes/probe_compact_final_routes.py`
  - Result lines:
    - `Vertical features feasible: ('m', 'q', 'q2') mask_empty_q=False: NO (first_fail=ch1)`
    - `Vertical features feasible: ('m', 'q', 'q2') mask_empty_q=True: NO (first_fail=ch1)`
    - `Vertical 4-feature winners from cheap pool: NONE`
- Why dead: the three-feature carrier that could price below 403 cannot linearly separate row-4/color-1 positives from same-color off-row and other-color states under one padded vertical kernel. Adding enough interaction features can separate the rule, but the sanity check only becomes feasible at a feature bank like `('a','n','k','m','q','q2','nk','mk','n2','k2')`, whose final kernel alone needs `10*10*6 = 600` weights before feature-generation cost.
- One remaining hypothesis: a different final op with parameter sharing across rows, or a free-output integer contraction that accepts U8 operands, could make the compact true-color representation viable. Standard `QLinearConv`/`ConvInteger` vertical kernels do not.
