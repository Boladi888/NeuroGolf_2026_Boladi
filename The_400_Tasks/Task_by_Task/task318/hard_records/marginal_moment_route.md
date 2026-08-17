live bar: cost 100, sha256 0bfa37796a89ad38a7655d8e98dbb256aba81674632133fbae9b8b9a137c49fe (ng_current_bar.py, manifest authority, Vamshi, 2026-07-12T15:15:54Z)

attempted route: prompt's separable row/column/color moment route, tested in `prototypes/probe_task318_sub100.py` with fresh ARC-GEN pairs and explicit generator-source counterexample.

failed gate/probe: `python The_400_Tasks/Task_by_Task/task318/prototypes/probe_task318_sub100.py` wrote `prototypes/task318_marginal_probe.json`; result `marginals_not_sufficient`, with fresh ARC-GEN combined row/column collisions and a legal split-row/column counterexample.

why dead: task_ce4f8723 independently samples arbitrary 4x4 pixels for color 1 and color 2. Two legal top-half masks can have identical row/column moments but different per-cell output masks, so an exact solver must preserve aligned cell state somewhere.

one remaining hypothesis: only an op that performs local per-cell aligned reads at the free output boundary, or a legal sparse-weight/accounting equivalent of that op, can plausibly beat the 100-param Conv bar.
