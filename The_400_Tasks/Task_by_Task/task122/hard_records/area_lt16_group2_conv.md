# task122 hard record: area<16 group=2 Conv stencils

- Live/team-best bar: 810, SHA256 `9d256b458e31da4423eee2f1c90ac42204a8e573ea622c1b679c25865828eabf`, source `python tools/ng_current_bar.py --task 122 --json` / team manifest.
- Attempted route: keep the v10 one-node `group=2` free-output Conv formulation but reduce kernel area below 16 so full bias remains legal and total cost is `<810`.
- Failed gate: construction-level separability probe before ONNX emission. Command:
  `python The_400_Tasks/Task_by_Task/task122/prototypes/probe_task122_area15_group2_conv.py --max-dilation 3 --max-side 15 --early-conflict`
- Exact result: `checked=8217 conflict_dead=8217 no_conflict=0 lp_dead=0 feasible=0`; `NO_FEASIBLE_SUB810_AREA_LT16_GROUP2_CONV`.
- Why dead: every area<16 grouped-Conv placement through dilation 3 has at least one identical input patch with conflicting output labels on the 266 train/test/arc-gen pairs, so no linear separator can exist; no visible/truth gate should be run on this representation family.
- One remaining hypothesis: a non-Conv free-output factorization that avoids a charged middle plane, or a generator-proven structural output formula below the documented 960 scalar-rebuild floor. Do not revisit smaller grouped-Conv stencils unless the search space changes beyond rectangular area<16/dilation<=3.
