You are `{worker_id}` continuing `{task}` (forced second pass — required before review).

Your prior worker packet:

```text
{prior_worker_packet}
```

Push harder. Requirements:

- Goal: the LOWEST possible cost by any valid means — the task prompt and your first pass are hints, not constraints. Try a genuinely DIFFERENT from-scratch ONNX formulation than your first pass (different dominant intermediate shape/dtype or assembly), not a micro-tweak.
- Mine `Onnx_Advanced/INDEX.md` -> `techniques.md`/`traps.md` for cited levers, and the BoBrain provider context, for cheaper assemblies.
- Re-gate every candidate at `--n 8000` (visible + truth/ab; raw `--full`/n=200000 is banned) + `profile_real_cost.py`. A win must have profile cost STRICTLY below the team-best bar `{bar}`.
- If you still cannot beat the bar, return an explicit WALL verdict: the cheapest correct formulation you found, its profile cost, and the forced-component / information-theoretic argument for why lower is not reachable by your route.

Return an updated worker packet (same fields as the initial contract), making clear whether you have a strict win or a wall.

## Bar + proof tooling (2026-07-03)

- **Authoritative bar:** start with `python tools/ng_current_bar.py --task NNN --json`; build strictly below its `live_cost` (it reconciles manifest/team-best/workbook and flags stale sources) — not a BoBrain / old-doc cost.
- **Name your saving:** before claiming a win, run `python tools/ng_graph_cost_diff.py --task NNN --live <current-best.onnx> --candidate <yours.onnx>` and put the named charged-structure delta in your packet.
